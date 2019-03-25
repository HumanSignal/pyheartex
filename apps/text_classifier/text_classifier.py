import numpy as np
import logging
import click
import gzip
import os

from marisa_trie import RecordTrie
from functools import partial
from urllib.request import urlretrieve
from tqdm import tqdm
from mosestokenizer import MosesTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from htx import run_model_server

from htx.base_model import ChoicesBaseModel


logger = logging.getLogger(__name__)


def tokenize(list_of_strings, lang='en'):
    with MosesTokenizer(lang) as tokenizer:
        return list(map(tokenizer, list_of_strings))


def load_embeddings_data(model_dir, lang, words_limit):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    target_file_name = f'cc.{lang}.300.vec.gz'
    target_file = os.path.join(model_dir, target_file_name)
    output_file_prefix = os.path.splitext(target_file)[0] + '.vectors.npy'
    vectors_file = output_file_prefix + '.vectors.npy'
    words_file = output_file_prefix + '.words.marisa'
    if os.path.exists(vectors_file) and os.path.exists(words_file):
        return output_file_prefix
    if not os.path.exists(target_file):
        target_file_url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{target_file_name}'
        logger.info(f'Downloading file {target_file_url} to {target_file}')
        urlretrieve(target_file_url, target_file)
    logger.info(f'Start reading {target_file}')
    with gzip.open(target_file, mode='rt', encoding='utf-8', errors='ignore', newline='\n') as fin:
        n, d = map(int, fin.readline().split())
        if words_limit is not None:
            n = min(words_limit, n)
        words = []
        vectors = np.zeros((n, d), dtype=np.float32)
        for i, line in tqdm(enumerate(fin), total=n):
            if i >= n:
                break
            tokens = line.rstrip().split(' ')
            vectors[i, :] = np.fromiter(map(float, tokens[1:]), dtype=np.float32)
            words.append((tokens[0], (i,)))

        logger.info(f'Saving vectors to {vectors_file}')
        np.save(vectors_file, np.array(vectors, dtype=np.float32))
        logger.info(f'Saving words trie to {words_file}')
        RecordTrie('<i', words).save(words_file)
        return output_file_prefix


class WordEmbeddings(TransformerMixin):

    def __init__(self, words_file, vectors_file):
        self._words_file = words_file
        self._vectors_file = vectors_file

        self._vectors = np.load(self._vectors_file)
        self._words = RecordTrie('<i')
        self._words.load(self._words_file)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = np.zeros((len(X), self._vectors.shape[1]), dtype=np.float32)
        for i, list_of_words in enumerate(X):
            words_idx = []
            for word in list_of_words:
                maybe_index = self._words.get(word)
                if not maybe_index:
                    continue
                words_idx.append(maybe_index[0][0])

            if len(words_idx) > 0:
                output[i, :] = np.mean(self._vectors[words_idx], axis=0)
        return output


class TextClassifier(ChoicesBaseModel):

    def __init__(self, embedding_file_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_file_prefix = embedding_file_prefix

    def create_model(self):
        if not self.embedding_file_prefix:
            return make_pipeline(
                TfidfVectorizer(),
                LogisticRegression(multi_class='multinomial', solver='lbfgs')
            )
        return make_pipeline(
            FunctionTransformer(tokenize, validate=False, check_inverse=False),
            WordEmbeddings(f'{self.embedding_file_prefix}.words.marisa', f'{self.embedding_file_prefix}.vectors.npy'),
            LogisticRegression(multi_class='multinomial', solver='lbfgs')
        )


@click.command()
@click.option('--lang', help='language', required=True)
@click.option('--words-limit', help='limiting number of words loaded', default=None, type=int)
@click.option('--model-dir', help='model directory', type=click.Path(exists=True))
@click.option('--from-name', help='"from_name" key', required=True)
@click.option('--to-name', help='"to_name" key', required=True)
@click.option('--data-field', help='key to extract target data from task', required=True)
@click.option('--update-period', help='model update period in samples', type=int, default=1)
@click.option('--min-examples', help='min examples to start training', type=int, default=1)
@click.option('--port', help='server port', default='10001')
def main(lang, words_limit, model_dir, from_name, to_name, data_field, update_period, min_examples, port):
    logging.basicConfig(level=logging.WARNING)
    embedding_file_prefix = load_embeddings_data(model_dir, lang, words_limit)
    run_model_server(
        create_model_func=partial(
            TextClassifier,
            embedding_file_prefix=embedding_file_prefix,
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

