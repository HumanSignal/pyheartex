import numpy as np
import io
import logging
import json
import os

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


def iter_input_data_dir(data_dir):
    for filepath in glob(os.path.join(data_dir, '*.jsonl')):
        with io.open(filepath) as f:
            for line in f:  # TODO: what happens if another thread will write to this file?
                try:
                    yield json.loads(line.strip())
                except:
                    logger.error(f'Cannot parse {line} from file {filepath}')


def generate_version():
    return str(int(datetime.now().timestamp()))
