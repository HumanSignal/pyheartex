import logging

logging.basicConfig(level=logging.INFO)

from htx import app, init_model_server
from functools import partial
from ranker import Ranker

init_model_server(
    create_model_func=partial(
        Ranker,
        tag_name='ranker',
        tag_type='list',
        source_name='ranked',
        source_type='ranker'
    ),
    model_dir='models',
    retrain_after_num_examples=1,
    min_examples_for_train=1
)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port='10002', debug=True)
