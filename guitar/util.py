import numpy as np
import json


def load_tab(tab_file):
    data = json.load(open(tab_file))
    return np.array(data['intervals']), data['labels']
