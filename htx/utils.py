import numpy as np
import io
import logging
import json
import os
import hashlib
import requests

from glob import glob
from datetime import datetime

logger = logging.getLogger(__name__)


def encode_labels(labels):
    unique_labels = np.unique(labels)
    label2idx = {}
    idx2label = list(unique_labels)
    for i, label in enumerate(unique_labels):
        label2idx[label] = i
    output_idx = [label2idx[l] for l in labels]
    return idx2label, output_idx


def generate_version():
    return str(int(datetime.now().timestamp()))


def download(url, output_dir, filename=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if filename is None:
        filename = hashlib.md5(url.encode()).hexdigest()
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        return filepath
    r = requests.get(url)
    r.raise_for_status()
    with io.open(filepath, mode='wb') as fout:
        fout.write(r.content)
    return filepath
