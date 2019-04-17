import os
import logging

from htx import app, init_model_server
from functools import partial
from text_classifier import TextClassifier, load_embeddings_data

logging.basicConfig(level=logging.INFO)

lang = os.environ['lang']
embedding_file_prefix = load_embeddings_data('embeddings', lang, words_limit=50000)

init_model_server(
    create_model_func=partial(
        TextClassifier,
        embedding_file_prefix=embedding_file_prefix,
        tag_name='text_class',
        tag_type='choices',
        source_name='text',
        source_type='text'
    ),
    model_dir='models',
    retrain_after_num_examples=10,
    min_examples_for_train=10
)

if __name__ == "__main__":
    app.run()
