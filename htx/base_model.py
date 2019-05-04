import numpy as np
import logging
import pickle
import json

from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseModel(ABC):

    def __init__(self, tag_type, source_type, tag_name=None, source_name=None, source_value=None):
        self.tag_name = tag_name
        self.tag_type = tag_type
        self.source_name = source_name
        self.source_type = source_type
        self.source_value = source_value
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

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


class ChoicesBaseModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(tag_type='choices', **kwargs)
        self._idx2label = []

    def get_inputs(self, tasks):
        inputs = []
        for task in tasks:
            inputs.append(task['data'][self.source_value])
        return inputs

    def get_outputs(self, tasks):
        outputs = []
        for task in tasks:
            single_choice = None
            for r in task['result']:
                if r['from_name'] == self.tag_name and r['to_name'] == self.source_name:
                    single_choice = r['value'].get(self.tag_type)
                    if isinstance(single_choice, list):
                        single_choice = single_choice[0]
                        break

            if not single_choice:
                raise ValueError(f'Cannot parse {task} with '
                                 f'tag_name={self.tag_name}, source_name={self.source_name}, tag_type={self.tag_type}')
            outputs.append(single_choice)
        return outputs

    def make_results(self, labels, scores):
        results = []
        for label, score in zip(labels, scores):
            results.append({
                'result': [{
                    'from_name': self.tag_name,
                    'to_name': self.source_name,
                    'value': {self.tag_type: [label]}
                }],
                'score': score
            })
        return results

    def _encode_labels(self, outputs):
        unique_labels = np.unique(outputs)
        label2idx = {}
        self._idx2label = list(unique_labels)
        for i, label in enumerate(unique_labels):
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
            logger.warning(f'Only one class is presented: {self._idx2label}.'
                           f' Need to collect more data...')
            return False
        inputs = self.get_inputs(tasks)

        self._model = self.create_model()

        self._model.fit(inputs, outputs_idx)
        return True

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


class LabelsBaseModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(tag_type='labels', **kwargs)

    def get_inputs(self, tasks):
        inputs = []
        for task in tasks:
            inputs.append(task['data'][self.source_value])
        return inputs

    def get_outputs(self, tasks):
        outputs = []
        for task in tasks:
            spans = []
            for r in task['result']:
                if r['from_name'] == self.tag_name and r['to_name'] == self.source_name:
                    labels = r['value'].get(self.tag_type)
                    if not isinstance(labels, list) or len(labels) == 0:
                        logger.warning(f'Error while parsing {r}: list type expected for "labels"')
                        continue
                    label = labels[0]
                    start, end = r['value'].get('start'), r['value'].get('end')
                    if not start or not end:
                        logger.warning(f'Error while parsing {r}: '
                                       f'{self.tag_type} should contain "start" and "end" fields')
                    spans.append({
                        'label': label,
                        'start': start,
                        'end': end
                    })
            outputs.append(spans)
        return outputs

    def make_results(self, list_of_spans, scores):
        results = []
        for spans, score in zip(list_of_spans, scores):
            result = []
            for span in spans:
                result.append({
                    'from_name': self.tag_name,
                    'to_name': self.source_name,
                    'value': {
                        self.tag_type: [span['label']],
                        'start': span['start'],
                        'end': span['end'],
                        'text': span['substr']
                    }
                })
            results.append({'result': result, 'score': score})
        return results

    @abstractmethod
    def create_model(self):
        pass

    def fit(self, tasks):
        inputs = self.get_inputs(tasks)
        outputs = self.get_outputs(tasks)
        self._model = self.create_model()
        self._model.fit(inputs, outputs)
        return True

    def predict(self, tasks):
        inputs = self.get_inputs(tasks)
        list_of_spans, scores = self._model.predict(inputs)
        return self.make_results(list_of_spans, scores)

    def save(self, filepath):
        with open(filepath, mode='wb') as fout:
            pickle.dump(self._model, fout)

    def load(self, filepath):
        with open(filepath, mode='rb') as f:
            self._model = pickle.load(f)


class TextClassifier(ChoicesBaseModel):

    def __init__(self, **kwargs):
        super().__init__(source_type='text', **kwargs)


class TextTagger(LabelsBaseModel):

    def __init__(self, **kwargs):
        super().__init__(source_type='text', **kwargs)


class ImageClassifier(ChoicesBaseModel):

    def __init__(self, **kwargs):
        super().__init__(source_type='image', **kwargs)
