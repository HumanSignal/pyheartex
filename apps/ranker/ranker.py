import click
import logging
import numpy as np

from operator import itemgetter
from htx import run_model_server
from htx.base_model import BaseModel
import xgboost as xgb

from text_classifier import load_embeddings_data, tokenize, WordEmbeddings


prefix = load_embeddings_data('word_models', lang='ru', words_limit=100000)
word_embeddings = WordEmbeddings(f'{prefix}.words.marisa', f'{prefix}.vectors.npy')


def texts_to_embeddings(texts):
    normalized_texts = tokenize(texts)
    return word_embeddings.transform(normalized_texts)


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

    def _extract_left(self, task):
        # TODO: make generic
        if 'features' in task['meta'].get('question', {}):
            return np.array(task['meta']['question']['features'])
        else:
            return np.mean(texts_to_embeddings(task['data']['questions']), axis=0)

    def _extract_rights(self, task):
        # TODO: make generic
        if 'features' in task['meta'].get('replies', {}):
            return np.array(list(map(itemgetter('features'), task['meta']['replies'])))
        else:
            #print(json.dumps(task, indent=2))
            return texts_to_embeddings(list(map(itemgetter('text'), task['data']['answers'])))

    def _extract_labels(self, task):
        # TODO: make generic
        return np.array(task['result'][0]['value']['weights'])

    def _get_inputs(self, tasks):
        if not len(tasks):
            raise ValueError('Empty inputs')
        out, groups = [], []
        for task in tasks:
            rvs = self._extract_rights(task)
            lv = self._extract_left(task)

            num_rvs = rvs.shape[0]
            final_features = np.hstack([
                np.dot(rvs, lv)[:, None],
                np.hstack((np.tile(lv, (num_rvs, 1)), rvs)),
                np.abs(rvs - lv),
                rvs * lv
            ])
            #final_features = np.dot(rvs, lv)[:, None]
            out.append(final_features)
            groups.append(rvs.shape[0])
        return np.vstack(out) if len(out) > 1 else out[0], groups

    def _get_outputs(self, tasks):
        return np.hstack(list(map(self._extract_labels, tasks)))

    def _make_results(self, scores, groups):
        start_idx = 0
        results = []
        for group_size in groups:
            group_scores = scores[start_idx:start_idx + group_size]
            results.append({
                'result': [{
                    'from_name': 'ranker',
                    'to_name': 'ranker',
                    'value': {'weights': group_scores.tolist()}
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
