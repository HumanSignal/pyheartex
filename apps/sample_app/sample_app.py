import click
import logging

from functools import partial
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from htx import run_model_server
from htx.base_model import ChoicesBaseModel


class DummyClassifier(ChoicesBaseModel):

    def create_model(self):
        return make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(multi_class='multinomial', solver='lbfgs')
        )


@click.command()
@click.option('--model-dir', help='model directory', type=click.Path(exists=True))
@click.option('--from-name', help='"from_name" key', required=True)
@click.option('--to-name', help='"to_name" key', required=True)
@click.option('--data-field', help='key to extract target data from task', required=True)
@click.option('--update-period', help='model update period in samples', type=int, default=1)
@click.option('--min-examples', help='min examples to start training', type=int, default=1)
@click.option('--port', help='server port', default='10001')
def main(model_dir, from_name, to_name, data_field, update_period, min_examples, port):
    logging.basicConfig(level=logging.DEBUG)
    run_model_server(
        create_model_func=partial(
            DummyClassifier,
            from_name=from_name,
            to_name=to_name,
            data_field=data_field
        ),
        model_dir=model_dir,
        retrain_after_num_examples=update_period,
        min_examples_for_train=min_examples,
        port=port
    )


if __name__ == "__main__":
    main()
