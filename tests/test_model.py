import pytest

from htx.base_model import *


@pytest.mark.parametrize('model_class, config_string, result', [
    (
        SingleClassTextClassifier,
        '''<View>
            <Text name="input_name" value="$input_value"/>
            <Choices name="output_name" toName="input_name">
                <Choice value="a"/>
                <Choice value="b"/>
            </Choices>
        </View>''',
        [{
            'input_names': ['input_name'],
            'output_names': ['output_name'],
            'input_values': ['input_value']
        }]
    ),
    (
        SingleClassImageClassifier,
        '''<View>
            <Text name="text_name" value="$text_value"/>
            <Choices name="choices_name" toName="text_name">
                <Choice value="a"/>
                <Choice value="b"/>
            </Choices>
        </View>''',
        []
    ),
    (
        SingleClassImageClassifier,
        '''<View>
            <Image name="input_name" value="$input_value"/>
            <Choices name="output_name" toName="input_name">
                <Choice value="a"/>
                <Choice value="b"/>
            </Choices>
        </View>''',
        [{
            'input_names': ['input_name'],
            'output_names': ['output_name'],
            'input_values': ['input_value']
        }]
    ),
    (
        SingleClassImageAndTextClassifier,
        '''<View>
            <Image name="input_name_1" value="$input_value_1"/>
            <Text name="input_name_2" value="$input_value_2"/>
            <Choices name="output_name" toName="input_name_1+input_name_2">
                <Choice value="a"/>
                <Choice value="b"/>
            </Choices>
        </View>''',
        [{
            'input_names': ['input_name_1', 'input_name_2'],
            'output_names': ['output_name'],
            'input_values': ['input_value_1', 'input_value_2']
        }]
    ),
    (
        TextTagger,
        '''<View>
            <Text name="input_name" value="$input_value"/>
            <Labels name="output_name_1" toName="input_name">
                <Label value="a"/>
                <Label value="b"/>
            </Labels>
            <Labels name="output_name_2" toName="input_name">
                <Label value="c"/>
                <Label value="d"/>
            </Labels>
        </View>''',
        [{
            'input_names': ['input_name'],
            'output_names': ['output_name_1'],
            'input_values': ['input_value']
        }, {
            'input_names': ['input_name'],
            'output_names': ['output_name_2'],
            'input_values': ['input_value']
        }]
    ),
    (
        TextTagger,
        '''<View>
            <Text name="text_name" value="$text_value"/>
            <Image name="image_name" value="$image_value"/>
            <Labels name="output_name_1" toName="text_name">
                <Label value="a"/>
                <Label value="b"/>
            </Labels>
            <Labels name="output_name_2" toName="image_name">
                <Label value="c"/>
                <Label value="d"/>
            </Labels>
        </View>''',
        [{
            'input_names': ['text_name'],
            'output_names': ['output_name_1'],
            'input_values': ['text_value']
        }]
    ),
    (
        ImageObjectDetection,
        '''<View>
            <Labels name="tag" toName="rect-1">
              <Label value="Cat"></Label>
              <Label value="Dog" background="blue"></Label>                
            </Labels>
            <AddRectangleButton name="rect-1" toName="image" value="Add Rectangle"></AddRectangleButton>
            <Image name="image" value="$image_url"></Image>
        </View>''',
        [{
            'input_names': ['image'],
            'output_names': ['rect-1'],
            'input_values': ['image_url']
        }]
    ),
])
def test_valid_schemas(model_class, config_string, result):
    m = model_class()
    assert m.get_valid_schemas(config_string) == result
