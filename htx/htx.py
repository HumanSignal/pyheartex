import json
import multiprocessing as mp

from functools import wraps
from flask import Flask, request, jsonify

from htx.model_manager import ModelManager


_server = Flask('htx.server')


def predict(from_name, to_name):

    def _decorator(func):

        @wraps(func)
        @_server.route('/predict', methods=['POST'])
        def wrapper(*args, **kwargs):
            data = json.loads(request.data)
            tasks = data['tasks']
            model_version = data.get('model_version')
            predict_results = func(data=[task['data'] for task in tasks], model_version=model_version, *args, **kwargs)

            results = []
            for predict_result in predict_results:
                score = predict_result.pop('score', 1.0)
                results.append({
                    'result': [{
                        'from_name': from_name,
                        'to_name': to_name,
                        'value': predict_result
                    }],
                    'score': score
                })

            if len(results) != len(tasks):
                raise ValueError(
                    'Number of results "{}" != number of input tasks "{}"'.format(len(results), len(tasks)))

            response = {
                'results': results,
                'model_version': model_version
            }
            print(json.dumps(response, indent=2))
            return jsonify(response)

        return wrapper

    return _decorator


def run(**kwargs):
    host = kwargs.get('host', '127.0.0.1')
    port = kwargs.get('port', '8999')
    debug = kwargs.get('debug', True)
    _server.run(host=host, port=port, debug=debug)


_model_manager = None


def run_model_server(create_model_func, model_dir, from_name, to_name, data_field, min_examples_for_train=10,
                     retrain_after_num_examples=10, **kwargs):
    global _model_manager
    _model_manager = ModelManager(
        create_model_func=create_model_func,
        model_dir=model_dir,
        from_name=from_name,
        to_name=to_name,
        data_field=data_field,
        min_examples_for_train=min_examples_for_train,
        retrain_after_num_examples=retrain_after_num_examples
    )
    run(**kwargs)


@_server.before_first_request
def _start_train_loop():
    train_process = mp.Process(target=_model_manager.train_loop, args=(_model_manager.queue, ))
    train_process.start()


@_server.route('/predict', methods=['POST'])
def _predict():
    data = json.loads(request.data)
    results = _model_manager.predict(data)
    response = {
        'results': results,
        'model_version': _model_manager.model_version
    }
    return jsonify(response)


@_server.route('/update', methods=['POST'])
def _update():
    data = json.loads(request.data)
    _model_manager.update(data)
    return jsonify({'status': 'ok'})


@_server.route('/version', methods=['POST'])
def _setup():
    _model_manager.setup()
    return jsonify({'model_version': _model_manager.model_version})
