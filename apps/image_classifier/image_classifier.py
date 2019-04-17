import click
import logging
import os
import torch
import numpy as np

from functools import partial
from htx import run_model_server
from htx.base_model import ChoicesBaseModel
from urllib.request import urlretrieve

from sklearn.linear_model import LogisticRegression
from torch import nn, no_grad
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


logger = logging.getLogger(__name__)

torch.set_num_threads(4)


image_size = 224
preprocessing = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
resnet.eval()


class ImageClassifierDataset(Dataset):

    def __init__(self, inputs, outputs, image_folder):
        self.image_folder = image_folder
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def prepare_image(cls, image_url, image_folder):
        filename = image_url.split('/')[-1]
        filepath = os.path.join(image_folder, filename)
        if not os.path.exists(filepath):
            logger.info(f'Downloading {image_url} to {filepath}')
            urlretrieve(image_url, filepath)
        image_data = Image.open(filepath).convert('RGB')
        preprocessed_image_data = preprocessing(image_data)
        return preprocessed_image_data

    def __getitem__(self, index):
        image = self.prepare_image(self.inputs[index], self.image_folder)
        image_class = self.outputs[index]
        return image, image_class

    def __len__(self):
        return len(self.inputs)


class ImageClassifierModel(object):

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self._model = None

    def fit(self, inputs, outputs):
        dataset = ImageClassifierDataset(inputs, outputs, self.image_folder)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

        X, y = [], []
        with no_grad():
            for batch_inputs, batch_outputs in dataloader:
                batch_X = resnet(batch_inputs)
                batch_X = torch.reshape(batch_X, (batch_X.size(0), batch_X.size(1)))
                X.append(batch_X.data.numpy())
                y.append(batch_outputs.data.numpy())

        X = np.vstack(X)
        y = np.hstack(y)

        self._model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self._model.fit(X, y)

    def predict_proba(self, inputs):
        preprocessed_images = []
        for image_url in inputs:
            i = ImageClassifierDataset.prepare_image(image_url, self.image_folder)
            preprocessed_images.append(i)
        preprocessed_images = torch.stack(preprocessed_images)
        with no_grad():
            tensor_X = resnet(preprocessed_images)
            tensor_X = torch.reshape(tensor_X, (tensor_X.size(0), tensor_X.size(1)))
            X = tensor_X.data.numpy()
        return self._model.predict_proba(X)


class ImageClassifier(ChoicesBaseModel):

    def __init__(self, image_folder, **kwargs):
        super().__init__(**kwargs)
        self.image_folder = image_folder

    def create_model(self):
        return ImageClassifierModel(self.image_folder)


@click.command()
@click.option('--image-folder', help='Image folder', type=click.Path(exists=True))
@click.option('--model-dir', help='model directory', type=click.Path(exists=True))
@click.option('--from-name', help='"from_name" key', required=True)
@click.option('--to-name', help='"to_name" key', required=True)
@click.option('--data-field', help='key to extract target data from task', required=True)
@click.option('--update-period', help='model update period in samples', type=int, default=1)
@click.option('--min-examples', help='min examples to start training', type=int, default=1)
@click.option('--port', help='server port', default='10001')
def main(image_folder, model_dir, from_name, to_name, data_field, update_period, min_examples, port):
    logging.basicConfig(level=logging.DEBUG)
    run_model_server(
        create_model_func=partial(
            ImageClassifier,
            image_folder=image_folder,
            from_name=from_name,
            to_name=to_name,
            data_field=data_field
        ),
        model_dir=model_dir,
        retrain_after_num_examples=update_period,
        min_examples_for_train=min_examples,
        port=port
    )


if __name__ == "__main__":
    main()
