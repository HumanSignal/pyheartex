import os
import multiprocessing as mp
import logging
import json

from datetime import datetime
from collections import defaultdict


logger = logging.getLogger(__name__)


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
        self._current_model_version = {}

    @property
    def model_version(self):
        return self._current_model_version

    def create_new_model(self):
        model = self.create_model_func()
        version = str(datetime.now())
        return model, version

    def load_model(self, model_version, project):
        if model_version != self._current_model_version[project]:
            model_file = os.path.join(self.model_dir, str(project), model_version)
            self._current_model[project] = self.create_model_func()
            self._current_model[project].load(model_file)
            self._current_model_version[project] = model_version

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

    def setup(self, project):
        model_list_file = os.path.join(self.model_dir, str(project), self._MODEL_LIST_FILE)
        if not os.path.exists(model_list_file) or os.stat(model_list_file).st_size == 0:
            logger.error(f'{self.model_list_file} is doesn''t exist or empty')
        else:
            logger.info(f'Reading "{model_list_file}"')
            with open(model_list_file) as f:
                model_list = f.read().splitlines()
            requested_model_version = model_list[-1]
            logger.info(f'Loading model version {requested_model_version}')
            self.load_model(requested_model_version, project)

    def predict(self, request_data):
        project = request_data['tasks'][0]['project']
        if self._current_model.get(project) is None:
            raise ValueError('Model is not loaded')

        requested_model_version = request_data.get('model_version')
        if self.model_version != requested_model_version:
            raise ValueError(
                f'Current model version "{self.model_version}" '
                f'!= requested model version "{requested_model_version}"'
            )
        # self.load_model(requested_model_version)
        results = self._current_model[project].predict(request_data['tasks'])

        return results

    def update(self, request_data):
        self.queue.put((request_data,))

    def train_loop(self, queue):
        logger.info(f'Train loop starts, PID={os.getpid()}')
        tasks = defaultdict(list)
        for request_data, in iter(queue.get, None):
            try:
                project = request_data.pop('project')
                tasks[project].append(request_data)
            except Exception as e:
                logger.error(f'Unable to collect new data:\n{json.dumps(request_data, indent=2)}\nReason:{str(e)}')
                continue
            try:
                train_tasks = tasks[project]
                if len(train_tasks) % self.retrain_after_num_examples == 0 \
                        and len(train_tasks) >= self.min_examples_for_train:
                    model, model_version = self.create_new_model()
                    logger.info(f'Start training model with {len(train_tasks)} tasks')
                    model.fit(train_tasks)
                    self.save_model(model, model_version, project)
                else:
                    logger.info(f'Reaching {len(train_tasks)} examples, not time to train...')
            except Exception as e:
                logger.error(f'Training failed. Reason: {str(e)}', exc_info=True)
                continue
        logger.info('Exit train loop')
