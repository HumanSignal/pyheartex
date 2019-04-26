import os
import argparse

from htx import app, init_model_server
from htx.base_model import TextClassifier

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MyTextClassifierOnHeartex(TextClassifier):
    """
    This exposes simple language-independent text classifier (aka MaxEnt classifier)
    You can return any "model" object in 'create_model()' function,
    with 2 methods implemented
        model.fit(X, y)
        model.predict_proba(X)
    (following scikit-learn convention https://scikit-learn.org/stable/)
    """

    def create_model(self):
        return make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )


init_model_server(
    create_model_func=MyTextClassifierOnHeartex,
    model_dir=os.path.expanduser('~/.heartex/models')
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example Heartex ML backend server with simple text classifier')
    parser.add_argument('--host', dest='host', help='Hostname', default='localhost')
    parser.add_argument('--port', dest='port', help='Port', type=int, default=8999)
    parser.add_argument('--debug', dest='debug', help='Start in debug mode', action='store_true')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
