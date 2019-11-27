import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)

from htx import app, init_model_server
from image_classifier import FastaiImageClassifier, train_script


init_model_server(
    create_model_func=FastaiImageClassifier,
    train_script=train_script,
    num_iter=10,
    image_dir='~/.heartex/images',
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=os.environ.get('REDIS_PORT', 6379),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', default='9090')
    args = parser.parse_args()
    app.run(host='localhost', port=args.port, debug=True)
