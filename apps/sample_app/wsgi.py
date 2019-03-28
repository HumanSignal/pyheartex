from htx import app, init_model_server
from functools import partial
from apps.sample_app.sample_app import DummyClassifier

init_model_server(
    create_model_func=partial(
        DummyClassifier,
        from_name='from_name',
        to_name='to_name',
        data_field='data'
    ),
    model_dir='models',
    retrain_after_num_examples=10,
    min_examples_for_train=10
)

if __name__ == "__main__":
    app.run()
