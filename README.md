# pyheartex

Python interface for running ML backend server and using it for active learning & prediction within [Heartex platform](https://www.heartex.net)

# Installation
```bash
git clone https://github.com/heartexlabs/pyheartex.git
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
The following scripts starts model server with simple text classifier

```bash
cd examples/
virtualenv -p python3 env && source env/bin/activate
pip install -r examples-requirements.txt
python run.py --host localhost --port 8999 --debug
```

Now you can send prediction request by using `TOKEN` and `PROJECT-ID` acquired [via Heartex]():
```bash
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token <TOKEN>" \
-d '[{"my_text": "is this cat or dog?"}]' \
http://go.heartex.net/api/projects/<PROJECT-ID>/predict
```
