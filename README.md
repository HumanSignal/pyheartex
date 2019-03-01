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
Assume your project is configured as follows:
```xml
<View>
  <TextEditor>
    <Text name="my_text_1" value="# my_text #"></Text>
    <Choices name="cats_or_dogs">
      <Choice value="cats"></Choice>
      <Choice value="dogs"></Choice>
    </Choices>
  </TextEditor>
</View>
```
and you are uploading the data:
```json
[
  {"data": {"my_text": "сat says miaou"}},
  {"data": {"my_text": "dog says woof"}}
]
```
The following script runs prediction server at `http://localhost:8999`

```python
import htx

@htx.predict(from_name='cats_or_dogs', to_name='my_text_1')
def predict(data, *args, **kwargs):
    results = []
    for item in data:
        results.append({
            'labels': 'cats' if 'cat' in item['my_text'] else 'dogs'
        })
    return results


if __name__ == "__main__":
    htx.run(host='localhost', port=8999)
```
