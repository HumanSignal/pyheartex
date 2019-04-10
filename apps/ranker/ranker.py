import click
import logging
import numpy as np
import json
import requests

from operator import itemgetter
from htx import run_model_server
from htx.base_model import BaseModel
import xgboost as xgb


logger = logging.getLogger(__name__)

#from text_classifier import load_embeddings_data, tokenize, WordEmbeddings


# prefix = load_embeddings_data('word_models', lang='ru', words_limit=100000)
# word_embeddings = WordEmbeddings(f'{prefix}.words.marisa', f'{prefix}.vectors.npy')
#
#
# def texts_to_embeddings(texts):
#     normalized_texts = tokenize(texts)
#     return word_embeddings.transform(normalized_texts)

def get_snm():
    return None
    # logger.info('Start loading SMN...')
    # x = np.load('data/embeddings_matrix_SMN_22_a.npz')['arr_0']
    # logger.info('SMN loaded!')
    # return x


smn = get_snm()


class Ranker(BaseModel):

    def _create_model(self):
        params = {
            'objective': 'rank:pairwise',
            'learning_rate': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 6,
            #'n_estimators': 500
        }
        return xgb.sklearn.XGBRanker(**params)

    def _get_smn_feature(self, task):

        context = list(map(itemgetter('text'), task['data']['context']))
        replies = list(map(itemgetter('text'), task['data']['replies']))

        while len(context) < 10:
            context.insert(0, '')
        request_dict = {
            'context': [context + replies]
        }
        r = requests.post(
            url='http://127.0.0.1:5000/ranker',
            data=json.dumps(request_dict, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        resp_dict = json.loads(r.text)[0]
        scores = resp_dict[0]
        embeddings = resp_dict[1]
        return scores, embeddings

    def _get_features(self, task):

        _, embeddings = self._get_smn_feature(task)
        return np.vstack(embeddings)

    def _get_inputs(self, tasks):
        if not len(tasks):
            raise ValueError('Empty inputs')
        out, groups = [], []
        for task in tasks:
            features = self._get_features(task)
            out.append(features)
            groups.append(features.shape[0])
        return np.vstack(out) if len(out) > 1 else out[0], groups

    def _get_outputs(self, tasks):
        out = []
        for task in tasks:
            selected = task['result'][0]['value']['selected']
            weights = task['result'][0]['value']['weights']
            out.append([s * w for s, w in zip(selected, weights)])
        return np.hstack(out)

    def _make_results(self, scores, groups):
        start_idx = 0
        results = []
        for group_size in groups:
            group_scores = scores[start_idx:start_idx + group_size]
            results.append({
                'result': [{
                    'from_name': 'ranker',
                    'to_name': 'ranker',
                    'value': {'weights': group_scores.tolist(), 'selected': [0] * len(group_scores)}
                }],
                'score': 1.0
            })
            start_idx += group_size
        return results

    def fit(self, tasks):
        self._model = self._create_model()
        x_train, groups = self._get_inputs(tasks)
        y_train = self._get_outputs(tasks)
        self._model.fit(x_train, y_train, groups)

    def predict(self, tasks):
        x_test, groups = self._get_inputs(tasks)
        scores = self._model.predict(x_test)
        results = self._make_results(scores, groups)
        return results

    def save(self, filepath):
        self._model.save_model(filepath)

    def load(self, filepath):
        self._model = self._create_model()
        self._model.load_model(filepath)


@click.command()
@click.option('--model-dir', help='model directory', type=click.Path(exists=True))
@click.option('--update-period', help='model update period in samples', type=int, default=1)
@click.option('--min-examples', help='min examples to start training', type=int, default=1)
@click.option('--port', help='server port', default='10001')
def main(model_dir, update_period, min_examples, port):
    logging.basicConfig(level=logging.INFO)
    run_model_server(
        create_model_func=lambda: Ranker(from_name=None, to_name=None, data_field=None),
        model_dir=model_dir,
        retrain_after_num_examples=update_period,
        min_examples_for_train=min_examples,
        port=port
    )


if __name__ == "__main__":
    main()
