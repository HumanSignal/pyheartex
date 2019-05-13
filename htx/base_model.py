import logging
import pickle
import json
import attr
import xmljson

from lxml import etree
from abc import ABC, abstractmethod
from itertools import product
from operator import itemgetter


logger = logging.getLogger(__name__)

# Input types
TEXT_TYPE = 'Text'
IMAGE_TYPE = 'Image'

# Output types
CHOICES_TYPE = 'Choices'
LABELS_TYPE = 'Labels'
BOUNDING_BOX_TYPE = 'AddRectangleButton'


@attr.s
class DataItem(object):
    input = attr.ib()
    output = attr.ib(default=None)
    meta = attr.ib(default=None)


class BaseModel(ABC):

    INPUT_TYPES = None
    OUTPUT_TYPES = None

    def __init__(self, input_names=None, output_names=None, input_values=None):
        self.input_names = input_names
        self.output_names = output_names
        self.input_values = input_values

        self._model = None

    def get_input(self, task):
        if 'data' not in task:
            return None

        return [task['data'].get(input_value) for input_value in self.input_values]

    @abstractmethod
    def get_output(self, task):
        pass

    def _parse_config_to_json(cls, config_string):
        parser = etree.XMLParser(recover=False)
        xml = etree.fromstring(config_string, parser)
        if xml is None:
            return None
        config = xmljson.badgerfish.data(xml)
        return config.get('View')

    def set_current_schema(self, input_names, output_names, input_values):
        self.input_names = input_names
        self.output_names = output_names
        self.input_values = input_values

    def get_valid_schemas(self, config_string):

        config = self._parse_config_to_json(config_string)
        if not config:
            logger.warning(f'Cannot parse config string {config_string}.')
            return []

        if not (all(i in config for i in self.INPUT_TYPES) and all(o in config for o in self.OUTPUT_TYPES)):
            logger.info(f'{self.__class__.__name__} has no valid schemas for config {config}')
            return []

        valid_inputs = []
        valid_input_names_found = set()
        for input_type in self.INPUT_TYPES:
            valid_inputs_of_type = []
            tagvalue = config[input_type]
            if isinstance(tagvalue, list):
                for tagvalue_ in tagvalue:
                    valid_inputs_of_type.append(
                        {'type': input_type, 'name': tagvalue_['@name'], 'value': tagvalue_['@value'].lstrip('$')})
                    valid_input_names_found.add(tagvalue_['@name'])
            else:
                valid_inputs_of_type.append(
                    {'type': input_type, 'name': tagvalue['@name'], 'value': tagvalue['@value'].lstrip('$')})
                valid_input_names_found.add(tagvalue['@name'])
            valid_inputs.append(valid_inputs_of_type)

        def _output_tag_is_applicable(tagvalue):
            # TODO: not sure this is correct way to check input->output bindings
            toNames = tagvalue['@toName'].split('+')
            if set(toNames).issubset(valid_input_names_found) and len(toNames) == len(self.INPUT_TYPES):
                return True
            return False

        valid_outputs = []
        for output_type in self.OUTPUT_TYPES:
            valid_outputs_of_type = []
            tagvalue = config[output_type]
            if isinstance(tagvalue, list):
                for tagvalue_ in tagvalue:
                    if not _output_tag_is_applicable(tagvalue_):
                        continue
                    valid_outputs_of_type.append({'type': output_type, 'name': tagvalue_['@name']})
            else:
                if not _output_tag_is_applicable(tagvalue):
                    continue
                valid_outputs_of_type.append({'type': output_type, 'name': tagvalue['@name']})
            valid_outputs.append(valid_outputs_of_type)

        schemas = []
        for valid_inputs_prod in product(*valid_inputs):
            for valid_outputs_prod in product(*valid_outputs):
                schema = {
                    'input_names': list(map(itemgetter('name'), valid_inputs_prod)),
                    'output_names': list(map(itemgetter('name'), valid_outputs_prod)),
                    'input_values': list(map(itemgetter('value'), valid_inputs_prod))
                }
                logger.debug(f'{self.__class__.__name__} founds valid schema={schema} for config={config}')
                schemas.append(schema)

        return schemas

    def get_data_item(self, task, for_train=True):
        try:
            task_input = self.get_input(task)
        except Exception as e:
            logger.error(f'Cannot parse task input from {task}. Reason: {e}')
            return DataItem(None)

        if for_train:
            try:
                task_output = self.get_output(task)
            except Exception as e:
                logger.error(f'Cannot parse task output from {task}. Reason: {e}')
                return DataItem(None)
        else:
            task_output = None
        return DataItem(input=task_input, output=task_output, meta=task.get('meta'))

    @abstractmethod
    def predict(self, tasks):
        pass

    @abstractmethod
    def load(self, root_model_dir, version):
        pass

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


class SingleChoiceBaseModel(BaseModel):

    OUTPUT_TYPES = (CHOICES_TYPE,)

    def get_output(self, task):
        single_choice = None
        input_name = self.input_names[0]
        output_name = self.output_names[0]

        for r in task['result']:
            if r['from_name'] == output_name and r['to_name'] == input_name:
                single_choice = r['value'].get('choices')
                if isinstance(single_choice, list):
                    single_choice = single_choice[0]
                    break

        if not single_choice:
            raise ValueError(f'Cannot parse {task} with tag_type="choices", input_name={input_name}, '
                             f'output_name={output_name}')
        return single_choice

    def make_results(self, labels, scores):
        results = []
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for label, score in zip(labels, scores):
            results.append({
                'result': [{
                    'from_name': output_name,
                    'to_name': input_name,
                    'value': {'choices': [label]}
                }],
                'score': score
            })
        return results


class SingleLabelsBaseModel(BaseModel):

    OUTPUT_TYPES = (LABELS_TYPE,)

    def get_output(self, task):
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
                if not start or not end:
                    logger.warning(f'Error while parsing {r}: "labels" should contain "start" and "end" fields')
                spans.append({
                    'label': label,
                    'start': start,
                    'end': end
                })
        return spans

    def make_results(self, list_of_spans, scores):
        results = []
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        for spans, score in zip(list_of_spans, scores):
            result = []
            for span in spans:
                result.append({
                    'from_name': output_name,
                    'to_name': input_name,
                    'value': {
                        'labels': [span['label']],
                        'start': span['start'],
                        'end': span['end'],
                        'text': span['substr']
                    }
                })
            results.append({'result': result, 'score': score})
        return results

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


class BoundingBoxBaseModel(BaseModel):
    OUTPUT_TYPES = (LABELS_TYPE, BOUNDING_BOX_TYPE)

    def get_output(self, task):
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
                'label': value['labels'][0]
            })
        return output


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
