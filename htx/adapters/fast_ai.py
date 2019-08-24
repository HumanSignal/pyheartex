import os
import io
import requests
from PIL import Image

from pathlib import Path
from fastai.vision import ImageDataBunch, get_transforms, models, cnn_learner, accuracy, load_learner, open_image
from htx.base_model import SingleClassImageClassifier
from htx.utils import download
from htx import init_model_server, app


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

    def predict(self, tasks):
        pred_labels, pred_scores = [], []
        for task in tasks:
            image_file = download(task['input'][0], self._image_dir)
            _, label_idx, probs = self._model.predict(open_image(image_file))
            label = self._model.data.classes[label_idx]
            score = probs[label_idx]
            pred_labels.append(label)
            pred_scores.append(score.item())
        return self.make_results(tasks, pred_labels, pred_scores)


def fit_fastai_image_classifier(input_data, output_dir, image_dir, **kwargs):
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
        Path(image_dir), filenames, labels=labels, ds_tfms=tfms, size=224, bs=4)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy, path=output_dir)
    learn.fit(10)
    learn.export()
    return {'model_path': output_dir, 'image_dir': image_dir}


def serve(port=16118, debug=True, **fit_kwargs):
    """
    :param model: Sklearn-compatible model, that is pickleable and has interface for .fit(X, y) and .predict_proba(X)
    :param port: specify localhost port where the model will be served
    :param debug: whether to start at debug mode
    :return:
    """
    from fast_ai import fit_fastai_image_classifier
    init_model_server(
        create_model_func=FastaiImageClassifier,
        train_script=fit_fastai_image_classifier,
        image_dir='/mnt/big/images',
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_HOST', 6379),
        **fit_kwargs
    )

    app.run(host='localhost', port=port, debug=debug)
