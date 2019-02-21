import json

from functools import wraps
from flask import Flask, request, jsonify

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
