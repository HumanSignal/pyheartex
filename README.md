# pyheartex

Python interface for running ML backend server and using it for active learning & prelabeling & prediction within [Heartex platform](https://www.heartex.net)

# Installation

```bash
git clone https://github.com/heartexlabs/pyheartex.git
cd pyheartex/
pip install -r requirements.txt
pip install -e .
```

# Quick start

Assume you want to build a prediction service that classifies short texts onto two classes (e.g., positive/negative).

First thing you need to create a new project on [Heartex](go.heartex.net) (read [docs](http://go.heartex.net/static/docs/#/Business?id=create-new-project) for a detailed explanation of how to create a project).

### Step 1

Use the following config when create a project:
```html
<View>
  <Text name="txt-1" value="$my_text"></Text>
  <Choices name="pos-neg" toName="txt-1">
    <Choice value="positive"></Choice>
    <Choice value="negative"></Choice>
  </Choices>
</View>
```

### Step 2

Upload JSON file:
```json
[
  {"my_text": "It was great"},
  {"my_text": "Terrible, terible movie"}
]
```

### Step 3 

Start model server running a text classifier

```bash
cd examples/
virtualenv -p python3 env && source env/bin/activate
pip install -r examples-requirements.txt
python run.py --host localhost --port 8999 --debug
```

### Step 4

Configure your project settings to make use of the above model. Go into project settings, Machine Learning tag, click Add  Custom Model and input model name (whatever you like) and it's URL. If you've started it with the above script, it should be accessible through HTTP on port 8999 and your IP. For example, mine is running on http://12.248.117.34:8999

### Step 5

Label above examples through the Heartex interface

### Step 6

Now you can send prediction request by using `TOKEN` and `PROJECT-ID` acquired [via Heartex](https://go.heartex.net/):
```bash
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token <TOKEN>" \
-d '[{"my_text": "great, great!"}]' \
https://go.heartex.net/api/projects/<PROJECT-ID>/predict/
```
