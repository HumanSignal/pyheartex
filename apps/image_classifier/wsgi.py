import logging

from htx import app, init_model_server
from functools import partial
from image_classifier import ImageClassifier

init_model_server(
    create_model_func=partial(
        ImageClassifier,
        image_folder='images',
        tag_type='choices',
        tag_name='image_class',
        source_type='image',
        source_name='image_url'
    ),
    model_dir='models',
    retrain_after_num_examples=1,
    min_examples_for_train=1
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run()
