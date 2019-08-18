import numpy as np
import os
import joblib
import io
import json

from functools import partial

from htx import app, init_model_server
from htx.base_model import SingleClassTextClassifier
from htx.utils import encode_labels


class SklearnTextClassifier(SingleClassTextClassifier):
    
    def __init__(self, model, **kwargs):
        super(SklearnTextClassifier, self).__init__(**kwargs)
        self._model = model
        self._idx_choice_map = None

    def load(self, train_output):

        self._model = joblib.load(train_output['model'])
        with io.open(train_output['idx_choice_map']) as f:
            self._idx_choice_map = json.load(train_output['idx_choice_map'])

    def predict(self, tasks):
        texts = []
        for task in tasks:
            texts.append(task['input'][0])
        predict_proba = self._model.predict_proba(texts)
        predict_idx = np.argmax(predict_proba, axis=1)
        predict_scores = predict_proba[np.arange(len(predict_idx)), predict_idx]
        predict_choices = [self._idx_choice_map[c] for c in predict_idx]
        return self.make_results(tasks, predict_choices, predict_scores)


def fit_sklearn_classifier(input_data, output_dir, model, **kwargs):
    texts, choices = [], []
    for item in input_data:
        if item['output'] is not None:
            texts.append(item['input'][0])
            choices.append(item['output'][0])

    # create label indexers
    idx_choice_map, choices_idx = encode_labels(choices)
    model.fit(texts, choices_idx, **kwargs)

    model_file = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_file)

    idx_choice_map_file = os.path.join(output_dir, 'idx_choice_map.json')
    with io.open(idx_choice_map_file, mode='w') as fout:
        json.dump(idx_choice_map, fout, indent=2)

    return {
        'model': model_file,
        'idx_choice_map': idx_choice_map_file
    }


def serve(model, port=16118, debug=True, **fit_kwargs):
    """
    :param model: Sklearn-compatible model, that is pickleable and has interface for .fit(X, y) and .predict_proba(X)
    :param port: specify localhost port where the model will be served
    :param debug: whether to start at debug mode
    :return:
    """
    init_model_server(
        create_model_func=partial(SklearnTextClassifier, model=model),
        train_script=partial(fit_sklearn_classifier, model=model),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_HOST', 6379),
        **fit_kwargs
    )

    app.run(host='localhost', port=port, debug=debug)
