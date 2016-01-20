"""JAMS utilities, requires version 0.1

https://github.com/marl/jams/releases/tag/v0.1
"""

import json
import pyjams


def load_jamset(filepath):
    """Load a collection of keyed JAMS (a JAMSet) into memory.

    Parameters
    ----------
    filepath : str
        Path to a JAMSet on disk.

    Returns
    -------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    jamset = dict()
    with open(filepath) as fp:
        for k, v in json.load(fp).iteritems():
            jamset[k] = pyjams.JAMS(**v)

    return jamset


def save_jamset(jamset, filepath):
    """Save a collection of keyed JAMS (a JAMSet) to disk.

    Parameters
    ----------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    output_data = dict()
    with pyjams.JSONSupport():
        for k, jam in jamset.iteritems():
            output_data[k] = jam.__json__

    with open(filepath, 'w') as fp:
        json.dump(output_data, fp)
