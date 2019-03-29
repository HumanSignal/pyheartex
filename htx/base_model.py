import numpy as np
import logging
import pickle
import json

from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseModel(ABC):

    def __init__(self, data_field, from_name, to_name):
        self.data_field = data_field
        self.from_name = from_name
        self.to_name = to_name
        self._model = None

    @abstractmethod
    def fit(self, tasks):
        pass

    @abstractmethod
    def predict(self, tasks):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class ChoicesBaseModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idx2label = {}

    def get_inputs(self, tasks):
        inputs = []
        for task in tasks:
            inputs.append(task['data'][self.data_field])
        return inputs

    def get_outputs(self, tasks):
        outputs = []
        for task in tasks:
            outputs.append(task['result'][0]['value']['choices'][0])
        return outputs

    def make_results(self, labels, scores):
        results = []
        for label, score in zip(labels, scores):
            results.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'value': {'choices': [label]}
                }],
                'score': score
            })
        return results

    def _encode_labels(self, outputs):
        unique_labels = np.unique(outputs)
        label2idx = {}
        for i, label in enumerate(unique_labels):
            self._idx2label[i] = label
            label2idx[label] = i
        output_idx = [label2idx[l] for l in outputs]
        return output_idx

    @abstractmethod
    def create_model(self):
        pass

    def fit(self, tasks):

        outputs = self.get_outputs(tasks)
        outputs_idx = self._encode_labels(outputs)
        if len(self._idx2label) < 2:
            logger.warning(f'Only one class is presented: {self._idx2label.keys()}.'
                           f' Need to collect more data...')
            return
        inputs = self.get_inputs(tasks)

        self._model = self.create_model()

        self._model.fit(inputs, outputs_idx)

    def predict(self, tasks):
        inputs = self.get_inputs(tasks)
        predict_proba = self._model.predict_proba(inputs)
        predict_idx = np.argmax(predict_proba, axis=1)
        predict_scores = predict_proba[np.arange(len(predict_idx)), predict_idx]
        predict_labels = [self._idx2label[c] for c in predict_idx]
        return self.make_results(predict_labels, predict_scores)

    def save(self, filepath):
        labels_file = filepath + '.labels.json'
        with open(filepath, mode='wb') as fout:
            pickle.dump(self._model, fout)
        with open(labels_file, mode='w') as fout:
            json.dump(self._idx2label, fout, indent=4)

    def load(self, filepath):
        labels_file = filepath + '.labels.json'
        with open(filepath, mode='rb') as f:
            self._model = pickle.load(f)
        with open(labels_file) as f:
            self._idx2label = json.load(f)
