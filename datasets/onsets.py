"""Utilities for parsing onset data.
"""

import numpy as np


def load_timepoint_file(timepoint_file):
    """Return a timepoint file as a numpy array.
    """
    return np.array([float(l.strip("\n")) for l in open(timepoint_file)])


def align_onset_labels(cqt_array, onsets, framerate):
    """Translate onset times into a boolean onset vector.
    """
    y_true = np.zeros(len(cqt_array), dtype=bool)
    y_true[np.round(onsets*framerate).astype(int)] = True
    return y_true
