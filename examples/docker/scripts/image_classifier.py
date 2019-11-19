import io
import requests
import logging

from PIL import Image
from pathlib import Path
from fastai.vision import ImageDataBunch, get_transforms, models, cnn_learner, accuracy, load_learner, open_image
from htx.base_model import SingleClassImageClassifier
from htx.utils import download


logger = logging.getLogger(__name__)


class FastaiImageClassifier(SingleClassImageClassifier):

    def load(self, serialized_train_output):
        self._model = load_learner(serialized_train_output['model_path'])
        self._image_dir = serialized_train_output['image_dir']

    @classmethod
    def _get_image_from_url(self, url):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with io.BytesIO(r.content) as f:
            return Image.open(f).convert('RGB')

    def predict(self, tasks, **kwargs):
        pred_labels, pred_scores = [], []
        for task in tasks:
            image_file = download(task['input'][0], self._image_dir)
            _, label_idx, probs = self._model.predict(open_image(image_file))
            label = self._model.data.classes[label_idx]
            score = probs[label_idx]
            pred_labels.append(label)
            pred_scores.append(score.item())
        return self.make_results(tasks, pred_labels, pred_scores)


def train_script(input_data, output_dir, image_dir, batch_size=4, num_iter=10, **kwargs):
    """
    This script provides FastAI-compatible training for the input labeled images
    :param image_dir: directory with images
    :param filenames: image filenames
    :param labels: image labels
    :param output_dir: output directory where results will be exported
    :return: fastai.basic_train.Learner object
    """

    filenames, labels = [], []
    for item in input_data:
        if item['output'] is None:
            continue
        image_url = item['input'][0]
        image_path = download(image_url, image_dir)
        filenames.append(image_path)
        labels.append(item['output'][0])

    tfms = get_transforms()
    data = ImageDataBunch.from_lists(
        Path(image_dir),
        filenames,
        labels=labels,
        ds_tfms=tfms,
        size=224,
        bs=batch_size
    )
    learn = cnn_learner(data, models.resnet18, metrics=accuracy, path=output_dir)
    learn.fit_one_cycle(num_iter)
    learn.export()
    return {'model_path': output_dir, 'image_dir': image_dir}
