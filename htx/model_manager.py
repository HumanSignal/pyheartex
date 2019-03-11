import os
import pickle
import multiprocessing as mp
import numpy as np
import logging
import json

from datetime import datetime


logger = logging.getLogger(__name__)


class ModelManager(object):

    _MODEL_LIST_FILE = 'model_list.txt'
    _DEFAULT_MODEL_VERSION = 'model'
    queue = mp.Queue()

    def __init__(self, create_model_func, model_dir, from_name, to_name, data_field,
                 min_examples_for_train=10, retrain_after_num_examples=10):
        self.model_dir = model_dir
        self.create_model_func = create_model_func
        self.min_examples_for_train = min_examples_for_train
        self.retrain_after_num_examples = retrain_after_num_examples
        self.model_list_file = os.path.join(self.model_dir, self._MODEL_LIST_FILE)
        self.from_name = from_name
        self.to_name = to_name
        self.data_field = data_field

        self._current_model = None
        self._current_model_version = None
        self._idx2label = {}
        self._label2idx = {}

    @property
    def model_version(self):
        return self._current_model_version

    def create_new_model(self):
        model = self.create_model_func()
        version = str(datetime.now())
        return model, version

    def load_model(self, model_version):
        if model_version != self._current_model_version:
            model_file = os.path.join(self.model_dir, model_version)
            with open(model_file, mode='rb') as f:
                self._current_model = pickle.load(f)
                self._current_model_version = model_version
            with open(model_file + '.labels.json') as f:
                self._label2idx = json.load(f)
                self._idx2label = {v: k for k, v in self._label2idx.items()}

    def save_model(self, model, model_version, label2idx):
        output_model_file = os.path.join(self.model_dir, model_version)
        output_labels_file = output_model_file + '.labels.json'
        with open(output_model_file, mode='wb') as fout:
            pickle.dump(model, fout)
        with open(self.model_list_file, mode='a') as fout:
            fout.write(model_version + '\n')
        with open(output_labels_file, mode='w') as fout:
            json.dump(label2idx, fout, indent=4)
        logger.info(f'Model successfully saved to {output_model_file}')

    def setup(self):
        if not os.path.exists(self.model_list_file) or os.stat(self.model_list_file).st_size == 0:
            logger.error(f'{self.model_list_file} is doesn''t exist or empty')
        else:
            logger.info(f'Reading "{self.model_list_file}"')
            with open(self.model_list_file) as f:
                model_list = f.read().splitlines()
            requested_model_version = model_list[-1]
            logger.info(f'Loading model version {requested_model_version}')
            self.load_model(requested_model_version)

    def predict(self, request_data):
        if self._current_model is None:
            raise ValueError('Model is not loaded')

        requested_model_version = request_data.get('model_version')
        if self.model_version != requested_model_version:
            raise ValueError(
                f'Current model version "{self.model_version}" '
                f'!= requested model version "{requested_model_version}"'
            )
        # self.load_model(requested_model_version)
        inputs = [task['data'][self.data_field] for task in request_data['tasks']]
        predict_proba = self._current_model.predict_proba(inputs)
        predict_idx = np.argmax(predict_proba, axis=1)
        predict_scores = predict_proba[np.arange(len(predict_idx)), predict_idx]
        predict_labels = [self._idx2label[c] for c in predict_idx]
        results = []
        for predict_label, predict_score in zip(predict_labels, predict_scores):
            results.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'value': {'labels': [predict_label]}
                }],
                'score': predict_score
            })

        return results

    def update(self, request_data):
        self.queue.put((request_data,))

    @classmethod
    def _create_label_indexers(cls, outputs):
        unique_labels = np.unique(outputs)
        idx2label, label2idx = {}, {}
        for i, label in enumerate(unique_labels):
            idx2label[i] = label
            label2idx[label] = i
        output_idx = [label2idx[l] for l in outputs]
        return idx2label, label2idx, output_idx

    def train_loop(self, queue):
        logger.info(f'Train loop starts, PID={os.getpid()}')
        inputs, outputs = [], []
        for new_data, in iter(queue.get, None):
            try:
                new_input = new_data['task']['data'][self.data_field]
                # TODO: infer more general schema to extract outputs
                new_output = new_data['result'][0]['value']['labels'][0]
                inputs.append(new_input)
                outputs.append(new_output)
            except Exception as e:
                logger.error(f'Unable to collect new data:\n{json.dumps(new_data, indent=2)}\nReason:{str(e)}')
                continue
            try:
                if len(inputs) % self.retrain_after_num_examples == 0 and len(inputs) >= self.min_examples_for_train:
                    model, model_version = self.create_new_model()
                    idx2label, label2idx, outputs_idx = self._create_label_indexers(outputs)
                    if len(idx2label) < 2:
                        logger.warning(f'Only one class is presented: {idx2label.keys()}.'
                                       f' Need to collect more data...')
                        continue
                    logger.info(f'Start training model with {len(inputs)} examples')
                    model.fit(inputs, outputs_idx)
                    self.save_model(model, model_version, label2idx)
                else:
                    logger.info(f'Reaching {len(inputs)} examples, not time to train...')
            except Exception as e:
                logger.error(f'Training failed. Reason: {str(e)}', exc_info=True)
                continue
        logger.info('Exit train loop')
