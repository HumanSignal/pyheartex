import os
import multiprocessing as mp
import logging
import json
import attr
import io

from operator import attrgetter
from redis import Redis
from rq import Queue
from rq.registry import FinishedJobRegistry
from rq.job import Job


logger = logging.getLogger(__name__)


class ModelManager(object):

    _MODEL_LIST_FILE = 'model_list.txt'
    _DEFAULT_MODEL_VERSION = 'model'
    queue = mp.Queue()

    def __init__(
        self,
        create_model_func,
        model_dir='~/.heartex/models',
        data_dir='~/.heartex/data',
        min_examples_for_train=1,
        retrain_after_num_examples=1,
        train_interval=60,
        **train_kwargs
    ):
        self.model_dir = os.path.expanduser(model_dir)
        self.data_dir = os.path.expanduser(data_dir)
        self.create_model_func = create_model_func
        self.train_interval = train_interval
        self.train_kwargs = train_kwargs
        self.min_examples_for_train = min_examples_for_train
        self.retrain_after_num_examples = retrain_after_num_examples

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.model_list_file = os.path.join(self.model_dir, self._MODEL_LIST_FILE)

        self._current_model = {}
        self._redis = Redis()

    def _get_latest_finished_train_job(self, project):
        redis = Redis()
        queue = Queue(connection=redis)
        registry = FinishedJobRegistry(queue.name, queue.connection)
        if registry.count == 0:
            logger.info('Train job registry is empty.')
            return None
        jobs = []
        for job_id in registry.get_job_ids():
            job = Job.fetch(job_id, connection=redis)
            if job.meta.get('project') != project:
                continue
            jobs.append(job)
        jobs = sorted(jobs, key=attrgetter('ended_at'), reverse=True)
        latest_job = jobs[0]
        logger.info(f'Project {project}: latest train job found: {latest_job}')
        return latest_job

    def setup(self, project, schema):
        train_job = self._get_latest_finished_train_job(project)
        if not train_job:
            logger.info('No one training job has been finished yet.')
            return None

        model = self.create_model_func(**schema)
        loaded = model.load(train_job.result)
        if not loaded:
            logger.error('Model is not loaded.')
            return None
        self._current_model[project] = model
        return train_job.ended_at

    def validate(self, config):
        return self.create_model_func().get_valid_schemas(config)

    def predict(self, request_data):
        project = request_data['project']
        if project not in self._current_model:
            raise ValueError(f'Model is not loaded for project {project}')

        model = self._current_model[project]
        data_items = []
        for task in request_data['tasks']:
            data_items.append(attr.asdict(model.get_data_item(task, for_train=False)))
        results = model.predict(data_items)

        return results, model.version

    def update(self, task, project, schema):
        model = self.create_model_func(**schema)
        data_item = model.get_data_item(task, for_train=True)
        if not data_item.input:
            logger.warning(f'Input is missing for {data_item}: skip using it.')
        elif not data_item.output:
            logger.warning(f'Output is missing for {data_item}: skip using it.')
        else:
            self.queue.put((project, attr.asdict(data_item)))

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

    def train_loop(self, data_queue, train_script):
        redis = Redis()
        redis_queue = Queue(connection=redis)
        logger.info(f'Train loop starts: PID={os.getpid()}, Redis connection: {redis}, queue: {redis_queue}')
        for project, data in iter(data_queue.get, None):

            # data block
            project_data_dir = None
            try:
                project_data_dir = os.path.join(self.data_dir, project)
                if not os.path.exists(project_data_dir):
                    os.makedirs(project_data_dir)
                with io.open(os.path.join(project_data_dir, 'data.jsonl'), mode='a') as fout:
                    item = {'input': data['input'], 'output': data['output']}
                    if data['meta']:
                        item['meta'] = data['meta']
                    jsonl = json.dumps(item, ensure_ascii=False)
                    fout.write(jsonl + '\n')
            except Exception as error:
                logger.error(f'Failed to store data: data_dir={self.data_dir}, project={project}, '
                             f'data={data}. Reason: {error}', exc_info=True)

            if not project_data_dir:
                continue

            # train block
            try:
                redis_key = f'project:{project}'  # TODO: may be using scopes?
                redis.incr(redis_key)
                total_items = int(redis.get(redis_key))
                if total_items >= self.min_examples_for_train and total_items % self.retrain_after_num_examples == 0:
                    self._run_train_script(redis_queue, train_script, project_data_dir, project)
            except Exception as error:
                logger.error(f'Failed to start training job. Reason: {error}', exc_info=True)
