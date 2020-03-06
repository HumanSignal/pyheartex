import os
import logging
import json
import attr
import shutil
import time

from copy import deepcopy
from redis import Redis
from rq import Queue, get_current_job
from rq.registry import StartedJobRegistry, FinishedJobRegistry
from rq.job import Job
from .utils import generate_version
from .base_model import DataItem

logger = logging.getLogger(__name__)


@attr.s
class ModelWrapper(object):
    model = attr.ib()
    model_version = attr.ib()


class ModelManager(object):

    def __init__(
        self,
        create_model_func,
        train_script,
        model_dir='~/.heartex/models',
        redis_host='localhost',
        redis_port=6379,
        redis_queue='default',
        **train_kwargs
    ):
        self.model_dir = os.path.expanduser(model_dir)
        self.create_model_func = create_model_func
        self.train_script = train_script
        self.train_kwargs = train_kwargs
        self.redis_host = redis_host
        self.redis_port = redis_port

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self._redis = Redis(host=redis_host, port=redis_port)
        self._redis_queue = Queue(name=redis_queue, connection=self._redis)

    def _remove_jobs(self, project):
        started_registry = StartedJobRegistry(self._redis_queue.name, self._redis_queue.connection)
        finished_registry = FinishedJobRegistry(self._redis_queue.name, self._redis_queue.connection)
        for job_id in started_registry.get_job_ids() + finished_registry.get_job_ids():
            job = Job.fetch(job_id, connection=self._redis)
            if job.meta.get('project') != project:
                continue
            logger.info(f'Deleting job_id {job_id}')
            job.delete()

    def _start_training_job(self, project, params):
        train_kwargs = deepcopy(self.train_kwargs)
        train_kwargs.update(params)
        job = self._redis_queue.enqueue(
            self.train_script_wrapper,
            args=(self.train_script, self.redis_host, self.redis_port, self.model_dir, project, train_kwargs),
            job_timeout='365d',
            ttl=-1,
            result_ttl=-1,
            failure_ttl=300,
            meta={'project': project},
        )
        logger.info(f'Training job {job} started for project {project}')
        return job

    @classmethod
    def get_tasks_key(cls, project):
        return f'project:{project}:tasks'

    @classmethod
    def get_job_results_key(cls, project):
        return f'project:{project}:job_results'

    def _get_latest_job_result(self, project):
        job_results_key = self.get_job_results_key(project)
        num_finished_jobs = self._redis.llen(job_results_key)
        if num_finished_jobs == 0:
            logger.info(f'No one finished training jobs found by key {job_results_key}. Redis: {self._redis}')
            return None
        return json.loads(self._redis.lindex(job_results_key, -1))

    def _create_model_from_job_result(self, job_result, schema):
        resources = job_result['resources']
        model = self.create_model_func(**schema)
        model.load(resources)
        return model

    def setup(self, project, schema):
        if not hasattr(self, '_current_model'):
            # This ensures each subprocess loads its own copy of model to avoid pre-fork initializations
            self._current_model = {}

        current_model = self._current_model.get(project)
        job_result = self._get_latest_job_result(project)

        # if job queue is empty, leave current model or raise 404 if model doesn't exists
        if job_result is None:
            if current_model is not None:
                logger.info(f'Project {project} has active model version {current_model.model_version}. '
                            f'Can\'t load any newest one since job results queue is empty.')
                return current_model.model_version
            raise FileNotFoundError(f'Can\'t retrieve any job result for project {project}')

        # if current model has the same model version as from latest job, leave current model
        model_version_from_job_result = job_result['version']
        if current_model and current_model.model_version == model_version_from_job_result:
            logger.info(f'Model for project {project} is already up-to-date (version={current_model.model_version}), '
                        f'loading is unnecessary')
            return current_model.model_version

        # if latest job result has newer model version, reload current model
        model = self._create_model_from_job_result(job_result, schema)
        self._current_model[project] = ModelWrapper(model, model_version_from_job_result)
        logger.info(f'Model {self._current_model[project].model_version} successfully loaded for project {project}.')
        return self._current_model[project].model_version

    def validate(self, config):
        return self.create_model_func().get_valid_schemas(config)

    def job_status(self, job_id):
        job = Job.fetch(job_id, connection=self._redis)
        response = {
            'job_status': job.get_status(),
            'error': job.exc_info,
            'created_at': job.created_at,
            'enqueued_at': job.enqueued_at,
            'started_at': job.started_at,
            'ended_at': job.ended_at
        }
        if job.is_finished and isinstance(job.result, str):
            response['result'] = json.loads(job.result)
        return response

    def predict(self, tasks, project, schema=None, model_version=None, **kwargs):
        if not hasattr(self, '_current_model'):
            # This ensures each subprocess loads its own copy of model to avoid pre-fork initializations
            self._current_model = {}

        if self._current_model.get(project) is None:
            # try to load model in lazy mode
            if schema is None:
                raise ValueError(f'You are trying to get prediction for project {project}, but model is not loaded. '
                                 f'We can try to fix it, but you should specify valid "schema" field in request')

            # TODO: instead of just latest, here we can easily retrieve job result with requested model_version
            job_result = self._get_latest_job_result(project)
            if job_result is None:
                raise FileNotFoundError(
                    f'You are trying to get prediction for project {project}, but model is not loaded.'
                    f'We have sought in the latest job results to load model in lazy mode, but results are empty'
                )
            model = self._create_model_from_job_result(job_result, schema)
            version = job_result['version']
            self._current_model[project] = ModelWrapper(model, version)
            logger.info(f'Model {version} is initialized for project {project} in lazy mode.')
            model_version = version

        m = self._current_model[project]
        data_items = []
        for task in tasks:
            data_items.append(attr.asdict(m.model.get_data_item(task)))
        results = m.model.predict(data_items, **kwargs)
        return results, model_version

    @classmethod
    def train_script_wrapper(cls, train_script, redis_host, redis_port, model_dir, project, train_kwargs):
        redis = Redis(host=redis_host, port=redis_port)

        project_model_dir = os.path.join(model_dir, project)
        version = generate_version()
        workdir = os.path.join(project_model_dir, version)
        os.makedirs(workdir)
        data_stream = (
            DataItem.deserialize_to_dict(serialized_item)
            for serialized_item in redis.lrange(cls.get_tasks_key(project), 0, -1)
        )
        t = time.time()
        resources = train_script(data_stream, workdir, **train_kwargs)
        job_result = json.dumps({
            'status': 'ok',
            'resources': resources,
            'project': project,
            'workdir': workdir,
            'version': version,
            'job_id': get_current_job().id,
            'time': time.time() - t
        })
        redis.rpush(cls.get_job_results_key(project), job_result)
        return job_result

    def update(self, task, project, schema, retrain, params):
        model = self.create_model_func(**schema)
        data_item = model.get_data_item(task)

        self._redis.rpush(self.get_tasks_key(project), data_item.serialize())
        if retrain:
            job = self._start_training_job(project, params)
            return job

    def train(self, tasks, project, schema, params):
        model = self.create_model_func(**schema)
        tasks_key = self.get_tasks_key(project)
        self._redis.delete(tasks_key)
        # self._remove_jobs(project)
        for task in tasks:
            data_item = model.get_data_item(task)
            self._redis.rpush(tasks_key, data_item.serialize())
        job = self._start_training_job(project, params)
        return job

    def delete(self, project):
        self._remove_jobs(project)
        job_results_key = self.get_job_results_key(project)
        for job_result in self._redis.lrange(job_results_key, 0, -1):
            j = json.loads(job_result)
            if os.path.exists(j['workdir']):
                shutil.rmtree(j['workdir'], ignore_errors=True)
        project_dir = os.path.join(self.model_dir, project)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        self._redis.delete(self.get_tasks_key(project), job_results_key)
        if hasattr(self, '_current_model'):
            self._current_model.pop(project, None)

    def duplicate_model(self, project_src, project_dst):
        latest_job_result = self._get_latest_job_result(project_src)
        if latest_job_result is None:
            raise FileNotFoundError(
                f'You are trying to copy resources for project {project_src}, but latest job results are not found'
            )

        self._redis.rpush(self.get_job_results_key(project_dst), json.dumps(latest_job_result))
        logger.info(f'Found latest job results with model version={latest_job_result["version"]}: '
                    f'copying it from project {project_src} to project {project_dst}')
