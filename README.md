# pyheartex

Python interface for running ML backend server and using it for active learning & prediction within [Heartex platform](https://www.heartex.net)

# Installation
```bash
git clone https://github.com/niklub/pyheartex.git
cd pyheartex/
pip install -r requirements.txt
pip install -e .
```

# Usage
The following script runs server at default url `http://localhost:8999`

```python
import htx

@htx.predict(from_name='cats_or_dogs', to_name='text')
def predict(data, *args, **kwargs):
    results = []
    for item in data:
        results.append({
            'labels': 'Cats' if 'cats' in item['text'] else 'Dogs'
        })
    return results


if __name__ == "__main__":
    htx.run()
```