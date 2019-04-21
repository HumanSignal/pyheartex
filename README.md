# pyheartex

Python interface for running ML backend server and using it for active learning & prediction within [Heartex platform](https://www.heartex.net)

# Installation
```bash
git clone https://github.com/niklub/pyheartex.git
cd pyheartex/
pip install -r requirements.txt
pip install -e .
```

# Quick start
Assume you want to build prediction service that classifies short texts onto 2 classes (e.g. cats/dogs).

First thing you need to configure labeling project on [Heartex](www.heartex.net) (read [docs](http://go.heartex.net/static/docs/#/Business?id=create-new-project) for detailed explanation how to create projects on Heartex).

Use the following labeling config:
```xml
<View>
<Text name="my_text_1" value="$my_text"></Text>
<Choices name="cats_or_dogs">
  <Choice value="cats"></Choice>
  <Choice value="dogs"></Choice>
</Choices>
</View>
```
Then you upload JSON file with the data:
```json
[
  {"my_text": "сat says miaou"},
  {"my_text": "dog says woof"}
]
```
Heartex platform interacts with labelers and send data to the model server.
The following scripts starts model server at `http://localhost:8999` with simple MaxEnt classifier by using [scikit-learn](https://scikit-learn.org/stable/)

```python
from htx import app, init_model_server
from htx.base_model import ChoicesBaseModel

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MaxEntClassifier(ChoicesBaseModel):

    def create_model(self):
        return make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )

init_model_server(
    create_model_func=MaxEntClassifier,
    model_dir='path/to/models/dir'
)

if __name__ == "__main__":
    app.run(host='localhost', port=8999)
```

Now you can send prediction request by using `TOKEN` and `PROJECT-ID` acquired [via Heartex]():
```bash
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token <TOKEN>" \
-d '{"my_text": "is this cat or dog?"}' \
http://go.heartex.net/api/projects/<PROJECT-ID>/predict
```