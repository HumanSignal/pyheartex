import io
import os
import json
import logging

from flask import Flask, request, jsonify, send_file
from rq.exceptions import NoSuchJobError
from functools import wraps

from htx.model_manager import ModelManager


_server = Flask('htx.server')
logger = logging.getLogger(__name__)
_model_manager = None
LOG_DIR = '/tmp'
USERNAME = 'heartex'
PASSWORD = 'heartex'


def init_model_server(**kwargs):
    global _model_manager
    _model_manager = ModelManager(**kwargs)


@_server.route('/predict', methods=['POST'])
def _predict():
    data = json.loads(request.data)
    tasks = data['tasks']
    project = data['project']
    schema = data.get('schema')
    model_version = data.get('model_version')
    params = data.get('params', {})

    logger.info(f'Request: predict {len(tasks)} tasks for project {project}')
    results, model_version = _model_manager.predict(tasks, project, schema, model_version, **params)
    response = {
        'results': results,
        'model_version': model_version
    }
    return jsonify(response)


@_server.route('/update', methods=['POST'])
def _update():
    task = json.loads(request.data)
    project = task.pop('project')
    schema = task.pop('schema')
    retrain = task.pop('retrain', False)
    params = task.pop('params', {})
    logger.info(f'Update for project {project} with retrain={retrain}')
    maybe_job = _model_manager.update(task, project, schema, retrain, params)
    response = {}
    if maybe_job:
        response['job'] = maybe_job.id
        return jsonify(response), 201
    return jsonify(response)


@_server.route('/train', methods=['POST'])
def _train():
    data = json.loads(request.data)
    tasks = data['tasks']
    project = data['project']
    schema = data['schema']
    params = data.get('params', {})
    if len(tasks) == 0:
        return jsonify({'status': 'error', 'message': 'No tasks found.'}), 400
    logger.info(f'Request: train for project {project} with {len(tasks)} tasks')
    job = _model_manager.train(tasks, project, schema, params)
    response = {'job': job.id}
    return jsonify(response), 201


@_server.route('/setup', methods=['POST'])
def _setup():
    data = json.loads(request.data)
    project = data['project']
    schema = data['schema']
    logger.info(f'Request: setup for project {project}')
    model_version = _model_manager.setup(project, schema)
    response = {'model_version': model_version}
    return jsonify(response)


@_server.route('/validate', methods=['POST'])
def _validate():
    data = json.loads(request.data)
    config = data['config']
    logger.info(f'Request: validate {request.data}')
    validated_schemas = _model_manager.validate(config)
    if validated_schemas:
        return jsonify(validated_schemas)
    else:
        return jsonify({'status': 'failed'}), 422


@_server.route('/job_status', methods=['POST'])
def _job_status():
    data = json.loads(request.data)
    job = data['job']
    logger.info(f'Request: job status for {job}')
    response = _model_manager.job_status(job)
    return jsonify(response or {})


@_server.route('/delete', methods=['POST'])
def _delete():
    data = json.loads(request.data)
    project = data['project']
    logger.info(f'Request: delete project {project}')
    result = _model_manager.delete(project)
    return jsonify(result or {})


@_server.route('/duplicate_model', methods=['POST'])
def _duplicate_model():
    data = json.loads(request.data)
    project_src = data['project_src']
    project_dst = data['project_dst']
    result = _model_manager.duplicate_model(project_src, project_dst)
    return jsonify(result or {})


@_server.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'UP'})


@_server.rout('/metrics', metrics=['GET'])
def metrics():
    return jsonify({})


@_server.errorhandler(NoSuchJobError)
def no_such_job_error_handler(error):
    logger.warning(f'Got error: {str(error)}')
    return str(error), 410


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning(f'Got error: {str(error)}')
    return str(error), 404


def login_required(f):
    @wraps(f)
    def wrapped_view(**kwargs):
        auth = request.authorization
        if not (auth and auth.username == USERNAME and auth.password == PASSWORD):
            return ('Unauthorized', 401,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(**kwargs)
    return wrapped_view


@_server.route('/logs/<path:path>')
@login_required
def send_log(path):
    """ Log access via web """
    logfile = os.path.join(LOG_DIR, path)
    if not logfile.startswith(os.path.abspath(LOG_DIR) + os.sep):
        return 'Wrong path'

    file = io.open(logfile, mode='r', encoding='utf-8', errors='ignore').read()
    out = file[-1024 * 100:]  # read last 100 kB
    out = out.replace('File "', 'File "<b>').replace('", line', '</b>", line')
    out = out.replace('\n[', '\n\n[')

    return '<pre>' + out + '</pre>'
