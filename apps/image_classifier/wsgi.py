from htx import app, init_model_server
from functools import partial
from image_classifier import ImageClassifier

init_model_server(
    create_model_func=partial(
        ImageClassifier,
        image_folder='images',
        from_name='image_class',
        to_name='image_class',
        data_field='image_url'
    ),
    model_dir='models',
    retrain_after_num_examples=10,
    min_examples_for_train=10
)

if __name__ == "__main__":
    app.run()
