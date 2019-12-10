import logging
import json
import attr
import numpy as np
import xml.etree.ElementTree

from abc import ABC, abstractmethod
from collections import Iterable

logger = logging.getLogger(__name__)

# Input types
TEXT_TYPE = 'Text'
IMAGE_TYPE = 'Image'
AUDIO_TYPE = 'Audio'
AUDIO_PLUS_TYPE = 'AudioPlus'
HYPERTEXT_TYPE = 'HyperText'

# Output types
CHOICES_TYPE = 'Choices'
LABELS_TYPE = 'Labels'
BOUNDING_BOX_TYPE = 'RectangleLabels'
POLYGON_TYPE = 'PolygonLabels'
LIST_TYPE = 'Ranker'
TEXT_AREA_TYPE = 'TextArea'


@attr.s
class DataItem(object):
    input = attr.ib()
    input_types = attr.ib(default=None)
    input_names = attr.ib(default=None)
    id = attr.ib(default=None)
    output = attr.ib(default=None)
    meta = attr.ib(default=None)

    @property
    def empty_output(self):
        return self.output is None

    def serialize(self):
        return json.dumps(attr.asdict(self))

    @classmethod
    def deserialize_to_dict(cls, serialized_data_item):
        return json.loads(serialized_data_item)


class BaseModel(ABC):

    INPUT_TYPES = None
    OUTPUT_TYPES = None

    def __init__(self, input_names=None, output_names=None, input_values=None, input_types=None):
        self.input_names = input_names
        self.input_types = input_types
        self.output_names = output_names
        self.input_values = input_values

        self._model = None
        self._cluster = {}
        self._neighbors = {}

    def get_input(self, task):
        if 'data' not in task:
            return None
        inputs = []
        for input_value in self.input_values:
            maybe_meta = task.get('meta', {}).get(input_value)
            if maybe_meta:
                inputs.append(maybe_meta)
            else:
                inputs.append(task['data'].get(input_value))
        return inputs

    @abstractmethod
    def get_output(self, task):
        pass

    @classmethod
    def get_valid_schemas(cls, config_string):

        def _is_input_tag(tag):
            return tag.attrib.get('name') and tag.attrib.get('value', '').startswith('$')

        def _is_output_tag(tag):
            return tag.attrib.get('name') and tag.attrib.get('toName')

        xml_tree = xml.etree.ElementTree.fromstring(config_string)

        input_tags, output_tags = {}, {}
        for tag in xml_tree.iter():
            if _is_input_tag(tag):
                input_tags[tag.attrib['name']] = {'type': tag.tag, 'value': tag.attrib['value'].lstrip('$')}
            elif _is_output_tag(tag):
                output_tags[tag.attrib['name']] = {'type': tag.tag, 'to_name': tag.attrib['toName'].split(',')}

        schemas = []
        for output_name, output in output_tags.items():
            if output['type'] not in cls.OUTPUT_TYPES:
                continue
            input_names = output['to_name']
            input_types = [input_tags[name]['type'] for name in input_names]
            if cls.INPUT_TYPES is None or all(i in cls.INPUT_TYPES for i in input_types):
                input_values = [input_tags[name]['value'] for name in input_names]
                schema = {
                    'output_names': [output_name],
                    'input_names': input_names,
                    'input_values': input_values,
                    'input_types': input_types
                }
                schemas.append(schema)
                logger.debug(f'{cls.__class__.__name__} founds valid schema={schema} for config={config_string}')
        return schemas

    def get_data_item(self, task):
        meta = task.get('meta')
        id = task.get('id')
        try:
            task_input = self.get_input(task)
            task_input_types = self.input_types
            task_input_names = self.input_names
        except Exception as e:
            logger.error(f'Cannot parse task input from {task}. Reason: {e}')
            return DataItem(input=None, input_types=None, input_names=None, output=None, id=id, meta=meta)
        try:
            task_output = self.get_output(task)
        except Exception as e:
            logger.error(f'Cannot parse task output from {task}. Reason: {e}', exc_info=True)
            return DataItem(
                input=task_input, input_types=task_input_types, input_names=task_input_names, output=None, id=id,
                meta=meta
            )
        return DataItem(
            input=task_input, input_types=task_input_types, input_names=task_input_names, output=task_output, meta=meta,
            id=id
        )

    @abstractmethod
    def predict(self, tasks, **kwargs):
        pass

    @abstractmethod
    def load(self, serialized_train_output):
        pass

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


class SingleChoiceBaseModel(BaseModel):

    OUTPUT_TYPES = (CHOICES_TYPE,)

    def get_output(self, task):
        if not isinstance(task.get('result'), Iterable):
            return None
        single_choice = None
        input_name = self.input_names[0]
        output_name = self.output_names[0]

        for r in task['result']:
            if r['from_name'] == output_name and r['to_name'] == input_name:
                single_choice = r['value'].get('choices')
                if isinstance(single_choice, list):
                    break

        if not single_choice:
            raise ValueError(f'Cannot parse {task} with tag_type="choices", input_name={input_name}, '
                             f'output_name={output_name}')
        return single_choice

    def make_results(self, tasks, labels, scores):
        results = []
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for task, label, score in zip(tasks, labels, scores):
            if isinstance(label, str):
                choices = [label]
            elif isinstance(label, (list, tuple)):
                choices = label
            else:
                raise ValueError(f'Unexpected label type {type(label)}: list, tuple or str expected')
            results.append({
                'result': [{
                    'from_name': output_name,
                    'to_name': input_name,
                    'type': CHOICES_TYPE.lower(),
                    'value': {'choices': choices}
                }],
                'score': score,
                'cluster': self._cluster.get(str(task['id']))
            })
        return results


class SingleLabelsBaseModel(BaseModel):

    OUTPUT_TYPES = (LABELS_TYPE,)

    def get_output(self, task):
        if not isinstance(task.get('result'), Iterable):
            return None
        spans = []
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for r in task['result']:
            if r['from_name'] == output_name and r['to_name'] == input_name:
                labels = r['value'].get('labels')
                if not isinstance(labels, list) or len(labels) == 0:
                    logger.warning(f'Error while parsing {r}: list type expected for "labels"')
                    continue
                label = labels[0]
                start, end = r['value'].get('start'), r['value'].get('end')
                if start is None or end is None:
                    logger.warning(f'Error while parsing {r}: "labels" should contain "start" and "end" fields')
                spans.append({
                    'label': label,
                    'start': start,
                    'end': end
                })
        return spans

    def make_results(self, tasks, list_of_spans, scores):
        results = []
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for task, spans, score in zip(tasks, list_of_spans, scores):
            result = []
            for span in spans:
                result.append({
                    'from_name': output_name,
                    'to_name': input_name,
                    'type': LABELS_TYPE.lower(),
                    'value': {
                        'labels': [span['label']],
                        'start': span['start'],
                        'end': span['end'],
                        'text': span['substr']
                    }
                })
            results.append({
                'result': result,
                'score': score,
                'cluster': self._cluster.get(str(task['id']))
            })
        return results


class BoundingBoxBaseModel(BaseModel):
    OUTPUT_TYPES = (BOUNDING_BOX_TYPE,)

    def get_output(self, task):
        if not isinstance(task.get('result'), Iterable):
            return None
        input_name = self.input_names[0]
        labels_name = self.output_names[0]
        output = []
        for r in task['result']:
            if r['from_name'] != labels_name or r['to_name'] != input_name:
                continue
            value = r['value']
            output.append({
                'x': value['x'],
                'y': value['y'],
                'width': value['width'],
                'height': value['height'],
                'label': value['rectanglelabels'][0]
            })
        return output

    def make_result(self, tasks, list_of_bboxes, scores):
        results = []
        input_name = self.input_names[0] if self.input_names else None
        output_name = self.output_names[0] if self.output_names else None
        for task, bboxes, score in zip(tasks, list_of_bboxes, scores):
            result = []
            for bbox in bboxes:
                result.append({
                    'from_name': output_name,
                    'to_name': input_name,
                    'type': BOUNDING_BOX_TYPE.lower(),
                    'value': {
                        'rectanglelabels': [bbox['label']],
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'height': bbox['height'],
                        'width': bbox['width'],
                        'score': bbox.get('score')
                    }
                })
            results.append({
                'result': result,
                'score': score,
                'cluster': self._cluster.get(str(task['id']))
            })
        return results


class PolygonBaseModel(BaseModel):
    OUTPUT_TYPES = (POLYGON_TYPE,)

    def get_output(self, task):
        if not isinstance(task.get('result'), Iterable):
            return None
        input_name = self.input_names[0]
        labels_name = self.output_names[0]
        output = []
        for r in task['result']:
            if r['from_name'] != labels_name or r['to_name'] != input_name:
                continue
            value = r['value']
            output.append({
                'points': value['points'],
                'label': value['polygonlabels'][0]
            })
        return output

    def make_result(self, tasks, list_of_polygons, scores):
        results = []
        input_name = self.input_names[0] if self.input_names else None
        output_name = self.output_names[0] if self.output_names else None
        for task, polygons, score in zip(tasks, list_of_polygons, scores):
            result = []
            for polygon in polygons:
                result.append({
                    'from_name': output_name,
                    'to_name': input_name,
                    'type': POLYGON_TYPE.lower(),
                    'value': {
                        'polygonlabels': [polygon['label']],
                        'points': polygon['points'],
                        'score': polygon.get('score')
                    }
                })
            results.append({
                'result': result,
                'score': score,
                'neighbors': self._neighbors.get(str(task['id'])),
                'cluster': self._cluster.get(str(task['id']))
            })
        return results


class ListBaseModel(BaseModel):
    OUTPUT_TYPES = (LIST_TYPE,)

    def get_output(self, task):
        if not isinstance(task.get('result'), Iterable):
            return None
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for r in task['result']:
            if r['from_name'] != output_name or r['to_name'] != input_name:
                continue
            value = r['value']
            return {
                'selected': value['selected'],
                'weights': value.get('weights'),
                'items': value.get('items')
            }
        logger.warning(f'Can\'t get output for {self.__class__.__name__} from {task}')

    def make_result(self, tasks, list_scores, list_items):
        results = []
        input_name = self.input_names[0] if self.input_names else None
        output_name = self.output_names[0] if self.output_names else None
        for task, scores, items in zip(tasks, list_scores, list_items):
            results.append({
                'result': [{
                    'from_name': output_name,
                    'to_name': input_name,
                    'type': LIST_TYPE.lower(),
                    'value': {
                        'weights': scores,
                        'selected': [0] * len(scores),
                        'items': items
                    }
                }],
                'score': np.mean(scores),
                'cluster': self._cluster.get(str(task['id']))
            })
        return results


class SingleClassTextClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (TEXT_TYPE,)


class TextTagger(SingleLabelsBaseModel):
    INPUT_TYPES = (TEXT_TYPE,)


class AudioTagger(SingleLabelsBaseModel):
    INPUT_TYPES = (AUDIO_PLUS_TYPE,)


class SingleClassImageClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (IMAGE_TYPE,)


class SingleClassAudioClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (AUDIO_TYPE,)


class SingleClassImageAndTextClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (IMAGE_TYPE, TEXT_TYPE)


class ImageObjectDetection(BoundingBoxBaseModel):
    INPUT_TYPES = (IMAGE_TYPE,)


class ImageSegmentation(PolygonBaseModel):
    INPUT_TYPES = (IMAGE_TYPE,)
