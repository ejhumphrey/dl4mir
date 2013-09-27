"""
"""

import numpy as np
from collections import OrderedDict

NO_CHORD = "N"

def load_labfile(lab_file):
    """Load a lab file into a time array and a list of corresponding labels.

    Parameters
    ----------
    lab_file : string
        Path to an HTK chord label file.

    Returns
    -------
    boundaries : np.ndarray
        Chord boundaries, in seconds. Monotonically increasing.
    labels : list
        Chords labels corresponding to the time between boundaries.

    Note that len(time_points) = len(labels) + 1.
    """
    boundaries = []
    labels = []
    for i, line in enumerate(open(lab_file)):
        line = line.strip("\n").strip("\r")
        if not line:
            # Assume we're done?
            break
        if "\t" in line:
            line_parts = line.split("\t")
        elif " " in line:
            line_parts = []
            for p in line.split(" "):
                if len(p):
                    line_parts.append(p)
        if len(line_parts) != 3:
            raise ValueError(
                "Error parsing %s on line %d: %s" % (lab_file, i, repr(line)))
        start_time, end_time = float(line_parts[0]), float(line_parts[1])
        boundaries.append(start_time)
        labels.append(line_parts[-1])
    boundaries = np.array(boundaries + [end_time])
    assert np.diff(boundaries).min() > 0, \
        "Boundaries are not monotonically increasing."
    return boundaries, labels


def assign_labels_to_time_points(time_points, boundaries, labels):
    """Assign chord labels to a set of points in time.

    Parameters
    ----------
    time_points : array_like
        Points in time to assign chord labels.
    boundaries : np.ndarray
        Time boundaries between labels.
    labels : array_like
        Chord labels corresponding to the interval between adjacent boundaries.

    Returns
    -------
    output_labels : list
        Chord labels corresponding to the input time points.
    """
    output_labels = []
    for t in time_points:
        if t < boundaries.min() or t > boundaries.max():
            output_labels.append(NO_CHORD)
        else:
            index = np.argmax(boundaries > t) - 1
            output_labels.append(labels[index])
    return output_labels


def collect_unique_labels(lab_files):
    unique_labels = set()
    for lab_file in lab_files:
        unique_labels.update(load_labfile(lab_file)[1])
    return unique_labels


def count_chords(lab_files):
    all_chords = dict()
    for lab_file in lab_files:
        boundaries, labels = load_labfile(lab_file)
        for n, chord_label in enumerate(labels):
            if not chord_label in all_chords:
                all_chords[chord_label] = 0.0
            all_chords[chord_label] += (boundaries[n + 1] - boundaries[n])
    keys, values = [], []
    [(keys.append(k), values.append(v)) for k, v in all_chords.iteritems()]
    sorted_index = np.array(values).argsort()[::-1]
    return OrderedDict([(keys[i], values[i]) for i in sorted_index])


    return unique_labels
