import logging
import json
import attr
import xmljson
import numpy as np

from lxml import etree
from abc import ABC, abstractmethod
from itertools import product
from operator import itemgetter
from collections import Iterable

logger = logging.getLogger(__name__)

# Input types
TEXT_TYPE = 'Text'
IMAGE_TYPE = 'Image'

# Output types
CHOICES_TYPE = 'Choices'
LABELS_TYPE = 'Labels'
BOUNDING_BOX_TYPE = 'RectangleLabels'
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

    def get_input(self, task):
        if 'data' not in task:
            return None
        return [task['data'].get(input_value) for input_value in self.input_values]

    @abstractmethod
    def get_output(self, task):
        pass

    @classmethod
    def _parse_config_to_json(cls, config_string):
        parser = etree.XMLParser(recover=False)
        xml = etree.fromstring(config_string, parser)
        if xml is None:
            return None
        config = xmljson.badgerfish.data(xml)
        return config.get('View')

    @classmethod
    def get_valid_schemas(cls, config_string):
        config = cls._parse_config_to_json(config_string)

        def _iter_tagvalue(tagvalue):
            if isinstance(tagvalue, list):
                for t in tagvalue:
                    yield t
            else:
                yield tagvalue

        name_tag_map = {}
        for tag, value in config.items():
            for tagvalue in _iter_tagvalue(value):
                if '@name' not in tagvalue or '@value' not in tagvalue:
                    # non-data tags are ignored
                    continue
                name_tag_map[tagvalue['@name']] = {
                    'value': tagvalue['@value'],
                    'type': tag
                }

        def _get_inputs(output_tag):
            input_names, input_values, input_types = [], [], []
            for to_name in output_tag['@toName'].split(','):
                input_type = name_tag_map[to_name]['type']
                input_value = name_tag_map[to_name]['value'].lstrip('$')
                if cls.INPUT_TYPES is None or input_type in cls.INPUT_TYPES:
                    input_names.append(to_name)
                    input_values.append(input_value)
                    input_types.append(input_type)
            return input_names, input_values, input_types

        schemas = []
        for output_type in cls.OUTPUT_TYPES:
            if output_type not in config:
                continue
            for tagvalue in _iter_tagvalue(config[output_type]):
                input_names, input_values, input_types = _get_inputs(tagvalue)
                if len(input_names):
                    schema = {
                        'output_names': [tagvalue['@name']],
                        'input_names': input_names,
                        'input_values': input_values,
                        'input_types': input_types
                    }
                    schemas.append(schema)
                    logger.debug(f'{cls.__class__.__name__} founds valid schema={schema} for config={config}')
        return schemas

    # @classmethod
    # def get_valid_schemas(cls, config_string):
    #
    #     config = cls._parse_config_to_json(config_string)
    #     if not config:
    #         logger.warning(f'Cannot parse config string {config_string}.')
    #         return []
    #
    #     if not (all(i in config for i in cls.INPUT_TYPES) and all(o in config for o in cls.OUTPUT_TYPES)):
    #         # TODO: enhance log: currently is "INFO:htx.base_model:ABCMeta has no valid schemas..."
    #         logger.info(f'{cls.__class__.__name__} has no valid schemas for config {config}')
    #         return []
    #
    #     valid_inputs = []
    #     valid_input_names_found = set()
    #     for input_type in cls.INPUT_TYPES:
    #         valid_inputs_of_type = []
    #         tagvalue = config[input_type]
    #         if isinstance(tagvalue, list):
    #             for tagvalue_ in tagvalue:
    #                 valid_inputs_of_type.append(
    #                     {'type': input_type, 'name': tagvalue_['@name'], 'value': tagvalue_['@value'].lstrip('$')})
    #                 valid_input_names_found.add(tagvalue_['@name'])
    #         else:
    #             valid_inputs_of_type.append(
    #                 {'type': input_type, 'name': tagvalue['@name'], 'value': tagvalue['@value'].lstrip('$')})
    #             valid_input_names_found.add(tagvalue['@name'])
    #         valid_inputs.append(valid_inputs_of_type)
    #
    #     def _output_tag_is_applicable(tagvalue):
    #         # TODO: not sure this is correct way to check input->output bindings
    #         toNames = tagvalue['@toName'].split('+')
    #         if set(toNames).issubset(valid_input_names_found) and len(toNames) == len(cls.INPUT_TYPES):
    #             return True
    #         return False
    #
    #     valid_outputs = []
    #     for output_type in cls.OUTPUT_TYPES:
    #         valid_outputs_of_type = []
    #         tagvalue = config[output_type]
    #         if isinstance(tagvalue, list):
    #             for tagvalue_ in tagvalue:
    #                 if not _output_tag_is_applicable(tagvalue_):
    #                     continue
    #                 valid_outputs_of_type.append({'type': output_type, 'name': tagvalue_['@name']})
    #         else:
    #             if not _output_tag_is_applicable(tagvalue):
    #                 continue
    #             valid_outputs_of_type.append({'type': output_type, 'name': tagvalue['@name']})
    #         valid_outputs.append(valid_outputs_of_type)
    #
    #     schemas = []
    #     for valid_inputs_prod in product(*valid_inputs):
    #         for valid_outputs_prod in product(*valid_outputs):
    #             schema = {
    #                 'input_names': list(map(itemgetter('name'), valid_inputs_prod)),
    #                 'output_names': list(map(itemgetter('name'), valid_outputs_prod)),
    #                 'input_values': list(map(itemgetter('value'), valid_inputs_prod))
    #             }
    #             logger.debug(f'{cls.__class__.__name__} founds valid schema={schema} for config={config}')
    #             schemas.append(schema)
    #
    #     return schemas

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
    def predict(self, tasks):
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
                        'width': bbox['width']
                    }
                })
            results.append({
                'result': result,
                'score': score,
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


class SingleClassImageClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (IMAGE_TYPE,)


class SingleClassImageAndTextClassifier(SingleChoiceBaseModel):
    INPUT_TYPES = (IMAGE_TYPE, TEXT_TYPE)


class ImageObjectDetection(BoundingBoxBaseModel):
    INPUT_TYPES = (IMAGE_TYPE,)
