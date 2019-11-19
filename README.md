# pyheartex

Python interface for running ML backend server and using it for active learning & prelabeling & prediction within [Heartex platform](https://www.heartex.net)

# Installation

First make sure you have [Redis server](https://redis.io/topics/quickstart) running (otherwise you can use only prediction, not active learning).

Install Heartex SDK:
```bash
git clone https://github.com/heartexlabs/pyheartex.git
cd pyheartex/
pip install -r requirements.txt
pip install -e .
```

Last thing you should do is to start RQ workers in the background:
```bash
rq worker default
```

# Using Docker
Here is an example how to start serving image classifier:
```bash
cd examples/docker
docker-compose up
```
All you need to replace with your own model is to change loading, inference and training scripts from [this file](examples/docker/scripts/image_classifier.py).

# Quick start

Quick start guide provides the usage of the following popular machine learning frameworks within Heartex platform:
- [scikit-learn](#scikit-learn)
- [FastAI](#fastai)


## Scikit-learn
Let's serve [scikit-learn](https://scikit-learn.org/stable/) model for text classification.

You can simply launch
```bash
python examples/quickstart.py
```

This script looks like
```python
from htx.adapters.sklearn import serve

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


if __name__ == "__main__":

    # Creating sklearn-compatible model
    my_model = make_pipeline(TfidfVectorizer(), LogisticRegression())

    # Start serving this model
    serve(my_model)
``` 

It starts serving at http://localhost:16118 listening for Heartex event. 
To connect your model, go to Heartex -> Settings -> Machine learning page and choose "Add custom model".

Or you can use Heartex API to activate your model:
```bash
curl -X POST -H 'Content-Type: application/json' \
-H 'Authorization: Token <PUT-YOUR-TOKEN-HERE>' \
-d '[{"url": "$HOST:$PORT", "name": "my_model", "title": "My model", "description": "My new model deployed on Heartex"}]' \
http://go.heartex.net/api/projects/{project-id}/backends/
```
where `$HOST:$PORT` is your server URL that should be accessible from the outside.

## FastAI
You can integrate [FastAI](https://docs.fast.ai/) models similarly to scikit-learn.
Check [this example](examples/run_fastai_image_classifier.py) to learn how to plug in updateable image classifier.

# Advanced usage
When you want to go beyond using sklearn compatible API, you can build your own model, by making manually input/output interface conversion.
You have to subclass Heartex models as follows:
```python
from htx.base_model import BaseModel

# This class exposes methods needed to handle model in the runtime (loading into memory, running predictions)
class MyModel(BaseModel):

    def get_input(self, task):
        """Extract input from serialized task"""
        pass
    
    def get_output(self, task):
        """Extract output from serialized task"""
        pass
        
    def load(self, train_output):
        """Loads model into memory. `train_output` dict is actually the output the `train` method (see below)"""
        pass
        
    def predict(self, tasks):
        """Get list of tasks, already processed by `get_input` method, and returns completions in Heartex format"""
        pass
        
# This method handles model retraining
def train(input_tasks, output_model_dir, **kwargs):
    """
    :param input_tasks: list of tasks already processed by `get_input`
    :param output_model_dir: output directory where you can optionally store model resources
    :param kwargs: any additional kwargs taken from `train_kwargs`
    :return: `train_output` dict for consequent model loading
    """
    pass
```
