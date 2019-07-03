import json
import logging

from flask import Flask, request, jsonify

from htx.model_manager import ModelManager


_server = Flask('htx.server')
logger = logging.getLogger(__name__)
_model_manager = None


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

    logger.info(f'Request: predict {len(tasks)} tasks for project {project}')
    results, model_version = _model_manager.predict(tasks, project, schema, model_version)
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
    logger.info(f'Update for project {project} with retrain={retrain}')
    maybe_job = _model_manager.update(task, project, schema, retrain)
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
    if len(tasks) == 0:
        return jsonify({'status': 'error', 'message': 'No tasks found.'}), 400
    logger.info(f'Request: train for project {project} with {len(tasks)} tasks')
    job = _model_manager.train(tasks, project, schema)
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
    job_status, error, ended_at = _model_manager.job_status(job)
    response = {
        'job_status': job_status,
        'error': error,
        'ended_at': ended_at
    }
    return jsonify(response)
