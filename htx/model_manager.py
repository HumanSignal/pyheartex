import os
import multiprocessing as mp
import logging
import json
import attr
import io
import shutil

from operator import attrgetter
from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry, FinishedJobRegistry
from rq.job import Job


logger = logging.getLogger(__name__)


class QueuedItem(object):

    def __init__(self, project):
        self.project = project


class QueuedDataItem(QueuedItem):

    def __init__(self, data_item, project):
        super(QueuedDataItem, self).__init__(project)
        self.data_item = attr.asdict(data_item)


class QueuedDataItems(QueuedItem):

    def __init__(self, data_items, project):
        super(QueuedDataItems, self).__init__(project)
        self.data_items = [attr.asdict(data_item) for data_item in data_items]


class QueuedWaitSignal(QueuedItem):
    pass


class QueuedTrainSignal(QueuedItem):
    pass


class QueuedFlushAllSignal(QueuedItem):
    pass


class ModelManager(object):

    _MODEL_LIST_FILE = 'model_list.txt'
    _DEFAULT_MODEL_VERSION = 'model'
    queue = mp.Queue()

    def __init__(
        self,
        create_model_func,
        train_script,
        model_dir='~/.heartex/models',
        data_dir='~/.heartex/data',
        min_examples_for_train=1,
        retrain_after_num_examples=1,
        redis_host='localhost',
        redis_port=6379,
        redis_queue='default',
        **train_kwargs
    ):
        self.model_dir = os.path.expanduser(model_dir)
        self.data_dir = os.path.expanduser(data_dir)
        self.create_model_func = create_model_func
        self.train_script = train_script
        self.train_kwargs = train_kwargs
        self.min_examples_for_train = min_examples_for_train
        self.retrain_after_num_examples = retrain_after_num_examples
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_queue = redis_queue

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.model_list_file = os.path.join(self.model_dir, self._MODEL_LIST_FILE)

        self._current_model = {}
        self._redis = Redis(host=redis_host, port=redis_port)

    def _get_latest_finished_train_job(self, project):
        queue = Queue(name=self.redis_queue, connection=self._redis)
        registry = FinishedJobRegistry(queue.name, queue.connection)
        if registry.count == 0:
            logger.info('Train job registry is empty.')
            return None
        jobs = []
        for job_id in registry.get_job_ids():
            job = Job.fetch(job_id, connection=self._redis)
            if job.meta.get('project') != project:
                continue
            jobs.append(job)
        if not jobs:
            logger.info(f'No jobs found for project {project}')
            return None
        jobs = sorted(jobs, key=attrgetter('ended_at'), reverse=True)
        latest_job = jobs[0]
        logger.info(f'Project {project}: latest train job found: {latest_job}')
        return latest_job

    def _stash_resources(self, project, resources):
        self._redis.set(f'project:{project}:res', resources)

    def _try_stashpop_resources(self, project):
        return self._redis.get(f'project:{project}:res')

    def setup(self, project, schema):
        train_job = self._get_latest_finished_train_job(project)
        if train_job:
            resources = train_job.result
        else:
            # in case when jobs are broken, try to stash pop train resources from cache
            resources = self._try_stashpop_resources(project)

        if not resources:
            logger.info(f'Couldn\'t load train resources for project {project} neither from latest jobs, nor from cache.'
                        f' Model is not loaded, and consequent API calls (e.g. predict) will fail. This normally happens'
                        f' if your model training is not started at very beginning. Otherwise it is a bug.')
            return None

        try:
            model = self.create_model_func(**schema)
            model_version = model.load(resources)
        except Exception as exc:
            logger.error(f'Couldn\'t load model for project {project} from resources {resources}', exc_info=True)
            return None
        else:
            if model_version is None:
                logger.error(f'Found resources {resources}, but model is not loaded for project {project}. '
                             f'Consequent API calls (e.g. predict) will fail.')
                return None
        self._current_model[project] = model
        self._stash_resources(project, resources)
        logger.info(f'Model {model_version} successfully loaded for project {project}.')
        return model_version

    def validate(self, config):
        return self.create_model_func().get_valid_schemas(config)

    def predict(self, request_data):
        project = request_data['project']
        if project not in self._current_model:
            # try to initialize model
            model_version = self.setup(project, request_data['schema'])
            if model_version is not None:
                logger.info(f'Model {model_version} is initialized for project {project} in lazy mode.')
            else:
                raise ValueError(f'Model is not loaded for project {project}')

        model = self._current_model[project]
        data_items = []
        for task in request_data['tasks']:
            data_items.append(attr.asdict(model.get_data_item(task, for_train=False)))
        results = model.predict(data_items)

        return results

    def update(self, task, project, schema):
        model = self.create_model_func(**schema)
        data_item = model.get_data_item(task, for_train=True)
        if not data_item.input:
            logger.warning(f'Input is missing for {data_item}: skip using it.')
        elif not data_item.output:
            logger.warning(f'Output is missing for {data_item}: skip using it.')
        else:
            queued_items = [
                QueuedDataItem(data_item, project),
                QueuedTrainSignal(project)
            ]
            self.queue.put((queued_items,))

    def update_many(self, tasks, project, schema, for_train=True):
        model = self.create_model_func(**schema)
        data_items = []
        for task in tasks:
            data_item = model.get_data_item(task, for_train=for_train)
            if not data_item.input:
                logger.warning(f'Input is missing for {data_item}: skip using it.')
            elif not data_item.output:
                logger.warning(f'Output is missing for {data_item}: skip using it.')
            else:
                data_items.append(data_item)
        queued_items = [
            QueuedFlushAllSignal(project),
            QueuedDataItems(data_items, project),
            QueuedTrainSignal(project)
        ]

        self.queue.put((queued_items,))

    def upload_many(self, tasks, project, schema, start_training=True):
        model = self.create_model_func(**schema)
        data_items = []
        for task in tasks:
            data_item = model.get_data_item(task, for_train=False)
            if not data_item.input:
                logger.warning(f'Input is missing for {data_item}: skip using it.')
            else:
                data_items.append(data_item)
        queued_items = [
            QueuedFlushAllSignal(project),
            QueuedDataItems(data_items, project)
        ]
        if start_training:
            queued_items.append(QueuedTrainSignal(project))

        self.queue.put((queued_items,))

    def _run_train_script(self, queue, train_script, data_dir, project):
        project_model_dir = os.path.join(self.model_dir, project)
        job = queue.enqueue(
            train_script,
            args=(data_dir, project_model_dir),  # TODO: only project id is needed to be passed
            kwargs=self.train_kwargs,
            ttl=-1,
            result_ttl=-1,
            failure_ttl=300,
            meta={'project': project}
        )
        logger.info(f'Training job started: {job}')

    @staticmethod
    def _update_counters(redis, project):
        redis_key = f'project:{project}'  # TODO: may be using scopes?
        redis.incr(redis_key)
        total_items = int(redis.get(redis_key))
        return total_items

    @staticmethod
    def _delete_counters(redis, project):
        redis_key = f'project:{project}'
        redis.delete(redis_key)

    @staticmethod
    def _try_save_data(data_items, project_data_dir):
        output_file = None
        try:
            output_file = os.path.join(project_data_dir, 'data.jsonl')
            with io.open(output_file, mode='a') as fout:
                for data_item in data_items:
                    item = {'input': data_item['input'], 'output': data_item['output']}
                    if data_item['meta']:
                        item['meta'] = data_item['meta']
                    jsonl = json.dumps(item, ensure_ascii=False)
                    fout.write(jsonl + '\n')
        except Exception:
            logger.error(f'Failed to store data to {output_file}', exc_info=True)
            return False
        logger.info(f'{len(data_items)} data item(s) successfully saved to {output_file}')
        return True

    def _flush_all(self, project, redis, reqis_queue):

        # Cancel all running & finished jobs for specified project
        started_registry = StartedJobRegistry(reqis_queue.name, reqis_queue.connection)
        finished_registry = FinishedJobRegistry(reqis_queue.name, reqis_queue.connection)
        for job_id in started_registry.get_job_ids() + finished_registry.get_job_ids():
            job = Job.fetch(job_id, connection=self._redis)
            if job.meta.get('project') != project:
                continue
            logger.info(f'Deleting job_id {job_id}')
            job.delete()

        # Delete project keys from Redis
        self._delete_counters(redis, project)

        # Remove project data dir
        project_data_dir = os.path.join(self.data_dir, project)
        if os.path.exists(project_data_dir):
            logger.info(f'Remove {project_data_dir}')
            # TODO: do we need the locks here?
            shutil.rmtree(project_data_dir)

    def train_loop(self, data_queue, train_script):
        redis = Redis(host=self.redis_host, port=self.redis_port)
        redis_queue = Queue(name=self.redis_queue, connection=redis)
        logger.info(f'Train loop starts: PID={os.getpid()}, Redis connection: {redis}, queue: {redis_queue}')
        for queued_items, in iter(data_queue.get, None):
            for queued_item in queued_items:
                project = queued_item.project

                # ensure project dir exists
                project_data_dir = os.path.join(self.data_dir, project)
                if not os.path.exists(project_data_dir):
                    os.makedirs(project_data_dir)

                # one data item -> store it
                if isinstance(queued_item, QueuedDataItem):
                    self._try_save_data([queued_item.data_item], project_data_dir)

                # many data items -> store them
                elif isinstance(queued_item, QueuedDataItems):
                    self._try_save_data(queued_item.data_items, project_data_dir)

                # train signal -> launch training if conditions are met
                elif isinstance(queued_item, QueuedTrainSignal):
                    try:
                        total_items = self._update_counters(redis, project)
                        if total_items >= self.min_examples_for_train and total_items % self.retrain_after_num_examples == 0:
                            self._run_train_script(redis_queue, train_script, project_data_dir, project)
                    except Exception as error:
                        logger.error(f'Failed to start training job. Reason: {error}', exc_info=True)

                # wait signal -> do nothing
                elif isinstance(queued_item, QueuedWaitSignal):
                    pass

                # flush all signal -> completely remove data & models related to specified project
                elif isinstance(queued_item, QueuedFlushAllSignal):
                    self._flush_all(queued_item.project, redis, redis_queue)

                else:
                    logger.warning(f'Unknown queued item type {queued_item.__class__.__name__}')
