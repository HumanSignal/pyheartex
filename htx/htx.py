import json
import multiprocessing as mp
import logging

from flask import Flask, request, jsonify

from htx.model_manager import ModelManager


_server = Flask('htx.server')
logger = logging.getLogger(__name__)
_model_manager = None


def init_model_server(train_scipt, queue_name='default', **kwargs):
    global _model_manager
    _model_manager = ModelManager(**kwargs)
    train_process = mp.Process(target=_model_manager.train_loop, args=(_model_manager.queue, train_scipt, queue_name))
    train_process.start()


@_server.route('/predict', methods=['POST'])
def _predict():
    data = json.loads(request.data)
    results = _model_manager.predict(data)
    return jsonify(results)


@_server.route('/update', methods=['POST'])
def _update():
    task = json.loads(request.data)
    project = task.pop('project')
    schema = task.pop('schema')
    _model_manager.update(task, project, schema)
    return jsonify({'status': 'ok'})


@_server.route('/setup', methods=['POST'])
def _setup():
    data = json.loads(request.data)
    project = data['project']
    schema = data['schema']
    model_version = _model_manager.setup(project, schema)
    return jsonify({'model_version': model_version})


@_server.route('/validate', methods=['POST'])
def _validate():
    data = json.loads(request.data)
    config = data['config']
    validated_schemas = _model_manager.validate(config)
    if validated_schemas:
        return jsonify(validated_schemas)
    else:
        return jsonify({'status': 'failed'}), 422
