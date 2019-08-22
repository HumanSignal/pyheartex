"""
This scripts shows how to start serving simple text classifier
at http://localhost:16118, make it listening for Heartex events
"""
from htx.adapters.sklearn import serve

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


if __name__ == "__main__":

    # Creating sklearn-compatible model
    my_model = make_pipeline(TfidfVectorizer(), LogisticRegression())

    # Start serving this model
    serve(my_model)
