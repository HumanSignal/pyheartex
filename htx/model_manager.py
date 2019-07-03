import os
import logging
import json
import attr
import shutil
import time

from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry, FinishedJobRegistry
from rq.job import Job
from rq.exceptions import NoSuchJobError
from .utils import generate_version
from .base_model import DataItem

logger = logging.getLogger(__name__)


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

    def _start_training_job(self, project):
        job = self._redis_queue.enqueue(
            self.train_script_wrapper,
            args=(self.train_script, self.redis_host, self.redis_port, self.model_dir, project, self.train_kwargs),
            job_timeout='365d',
            ttl=-1,
            result_ttl='1d',
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

    def _create_model_from_finished_jobs(self, project, schema):
        job_results_key = self.get_job_results_key(project)
        num_finished_jobs = self._redis.llen(job_results_key)
        if num_finished_jobs == 0:
            logger.info(f'No one finished training jobs found by key {job_results_key}. Redis: {self._redis}')
            return None, None
        latest_job_result = json.loads(self._redis.lindex(job_results_key, -1))
        resources = latest_job_result['resources']
        model_version = latest_job_result['version']
        model = self.create_model_func(**schema)
        model.load(resources)
        return model, model_version

    def setup(self, project, schema):
        model, model_version = self._create_model_from_finished_jobs(project, schema)
        if model is not None:
            if not hasattr(self, '_current_model'):
                # This ensures each subprocess loads its own copy of model to avoid pre-fork initializations
                self._current_model = {}
            self._current_model[project] = model
            logger.info(f'Model {model_version} successfully loaded for project {project}.')
        return model_version

    def validate(self, config):
        return self.create_model_func().get_valid_schemas(config)

    def job_status(self, job_id):
        try:
            job = Job.fetch(job_id, connection=self._redis)
        except NoSuchJobError:
            logger.error(f'Can\'t get job status {job_id}: no such job', exc_info=True)
        else:
            status = job.get_status()
            error = job.exc_info
            ended_at = job.ended_at
            return status, error, ended_at

    def predict(self, tasks, project, schema=None, model_version=None):
        if not hasattr(self, '_current_model'):
            # This ensures each subprocess loads its own copy of model to avoid pre-fork initializations
            self._current_model = {}
        if self._current_model.get(project) is None:
            if schema is None:
                raise ValueError(f'You are trying to get prediction for project {project}, but model is not loaded. '
                                 f'We can fix it, but you should specify valid "schema" field in request')
            # try to initialize model
            model, model_version = self._create_model_from_finished_jobs(project, schema)
            if model_version is not None:
                logger.info(f'Model {model_version} is initialized for project {project} in lazy mode.')
            else:
                raise ValueError(f'Model is not loaded for project {project}')
            self._current_model[project] = model

        model = self._current_model[project]
        data_items = []
        for task in tasks:
            data_items.append(attr.asdict(model.get_data_item(task)))
        results = model.predict(data_items)
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
        redis.rpush(cls.get_job_results_key(project), json.dumps({
            'status': 'ok',
            'resources': resources,
            'project': project,
            'workdir': workdir,
            'version': version,
            'time': time.time() - t
        }))

    def update(self, task, project, schema, retrain):
        model = self.create_model_func(**schema)
        data_item = model.get_data_item(task)

        self._redis.rpush(self.get_tasks_key(project), data_item.serialize())
        if retrain:
            job = self._start_training_job(project)
            return job

    def train(self, tasks, project, schema):
        model = self.create_model_func(**schema)
        tasks_key = self.get_tasks_key(project)
        self._redis.delete(tasks_key)
        self._remove_jobs(project)
        for task in tasks:
            data_item = model.get_data_item(task)
            self._redis.rpush(tasks_key, data_item.serialize())
        job = self._start_training_job(project)
        return job

    def delete(self, project):
        self._remove_jobs(project)
        job_results_key = self.get_job_results_key(project)
        for job_result in self._redis.lrange(job_results_key, 0, -1):
            if os.path.exists(job_result['workdir']):
                shutil.rmtree(job_result['workdir'], ignore_errors=True)
        self._redis.delete(self.get_tasks_key(project), job_results_key)
