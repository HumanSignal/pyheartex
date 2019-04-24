import os
import multiprocessing as mp
import logging
import json
import attr

from datetime import datetime
from collections import defaultdict


logger = logging.getLogger(__name__)


@attr.s
class ModelItem(object):
    model = attr.ib()
    version = attr.ib()
    schema = attr.ib()

    def reassign_schema(self):
        # TODO: remove schema from model itself and use it in preprocessing step in ModelManager
        new_tag_name = self.schema.get('tag_name')
        if new_tag_name != self.model.tag_name:
            logger.info(f'New tag name={new_tag_name}')
            self.model.tag_name = new_tag_name

        new_source_name = self.schema.get('source_name')
        if new_source_name != self.model.source_name:
            logger.info(f'New source name={new_source_name}')
            self.model.source_name = new_source_name

        new_source_value = self.schema.get('source_value')
        if new_source_value != self.model.source_value:
            logger.info(f'New source value={new_source_value}')
            self.model.source_value = new_source_value


class ModelManager(object):

    _MODEL_LIST_FILE = 'model_list.txt'
    _DEFAULT_MODEL_VERSION = 'model'
    queue = mp.Queue()

    def __init__(self, create_model_func, model_dir, min_examples_for_train=10, retrain_after_num_examples=10):
        self.model_dir = model_dir
        self.create_model_func = create_model_func
        self.min_examples_for_train = min_examples_for_train
        self.retrain_after_num_examples = retrain_after_num_examples
        self.model_list_file = os.path.join(self.model_dir, self._MODEL_LIST_FILE)

        self._current_model = {}

    def get_model(self, project):
        return self._current_model.get(project)

    def get_model_version(self, project):
        curr_model = self._current_model.get(project)
        return curr_model.version if curr_model else None

    def create_new_model(self):
        model = self.create_model_func()
        version = str(datetime.now())
        return model, version

    def _create_new_model(self, version, schema):
        model = self.create_model_func()
        if schema and not self._validate(model, schema):
            error_msg = f'Current scheme {schema} is not valid for model {model}'
            logger.error(error_msg)
            raise ValueError(error_msg)
        model_item = ModelItem(
            model=self.create_model_func(),
            version=version,
            schema=schema or {}
        )
        return model_item

    def load_model(self, model_version, project, schema):
        model_item = self.get_model(project)
        if not model_item:
            logger.info(f'Creating new model for project={project}, version={model_version}, schema={schema}')
            self._current_model[project] = self._create_new_model(model_version, schema)
        elif model_version != model_item.version:
            logger.info(f'Creating new model for project={project}, version={model_version}, schema={schema}')
            self._current_model[project] = self._create_new_model(model_version, schema)
            model_file = os.path.join(self.model_dir, str(project), model_version)
            logger.info(f'Loading model parameters from {model_file}')
            self._current_model[project].model.load(model_file)

    def save_model(self, model, model_version, project):
        dirpath = os.path.join(self.model_dir, str(project))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        output_model_file = os.path.join(dirpath, model_version)
        model.save(output_model_file)
        model_list_file = os.path.join(dirpath, self._MODEL_LIST_FILE)
        with open(model_list_file, mode='a') as fout:
            fout.write(model_version + '\n')
        logger.info(f'Model successfully saved to {output_model_file}')

    def setup(self, project, scheme=None):
        model_list_file = os.path.join(self.model_dir, str(project), self._MODEL_LIST_FILE)
        if not os.path.exists(model_list_file) or os.stat(model_list_file).st_size == 0:
            logger.warning(f'{self.model_list_file} is doesn''t exist or empty')
            if not self.get_model(project):
                logger.info(f'Creating new empty model for project={project}')
                self._current_model[project] = self._create_new_model(version=None, schema=scheme)
        else:
            logger.info(f'Reading "{model_list_file}"')
            with open(model_list_file) as f:
                model_list = f.read().splitlines()
            requested_model_version = model_list[-1]
            logger.info(f'Loading model version {requested_model_version}')
            self.load_model(requested_model_version, project, schema=scheme)

        # TODO: make this explicit in model manager
        self.get_model(project).reassign_schema()

    @classmethod
    def _validate(cls, model, scheme):
        return (
            model.tag_type.lower() == scheme.get('tag_type', '').lower() and
            model.source_type.lower() == scheme.get('source_type', '').lower()
        )

    def validate(self, scheme):
        model = self.create_model_func()
        return model and scheme and self._validate(model, scheme)

    def predict(self, request_data):
        project = request_data['project']
        if self._current_model.get(project) is None:
            raise ValueError('Model is not loaded')

        current_model_version = self.get_model_version(project)
        requested_model_version = request_data.get('model_version')
        if current_model_version != requested_model_version:
            raise ValueError(
                f'Current model version "{current_model_version}" '
                f'!= requested model version "{requested_model_version}" for project {project}'
            )
        # self.load_model(requested_model_version)
        model_item = self.get_model(project)
        results = model_item.model.predict(request_data['tasks'])

        return results, current_model_version

    def update(self, request_data):
        project = request_data.pop('project')
        curr_model = self.get_model(project)
        schema = curr_model.schema if curr_model else {}
        self.queue.put((request_data, project, schema))

    def train_loop(self, queue):
        logger.info(f'Train loop starts, PID={os.getpid()}')
        tasks = defaultdict(list)  # TODO: its not good idea to save tasks in memory
        for request_data, project, schema in iter(queue.get, None):
            tasks[project].append(request_data)
            try:
                train_tasks = tasks[project]
                if len(train_tasks) % self.retrain_after_num_examples == 0 \
                        and len(train_tasks) >= self.min_examples_for_train:
                    model_version = str(datetime.now())
                    model_item = self._create_new_model(model_version, schema)
                    model_item.reassign_schema()
                    logger.info(f'Start training model with {len(train_tasks)} tasks')
                    fitted = model_item.model.fit(train_tasks)
                    if fitted:
                        self.save_model(model_item.model, model_version, project)
                else:
                    logger.info(f'Reaching {len(train_tasks)} examples, not time to train...')
            except Exception as e:
                logger.error(f'Training failed. Reason: {str(e)}', exc_info=True)
                continue
        logger.info('Exit train loop')
