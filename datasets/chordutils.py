"""
"""

import numpy as np
from collections import OrderedDict
from marl.hewey.core import DataPoint

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

def circshift_data(x, y, n, bins_per_octave):
    """
    x : np.ndarray
    y : int
    n : int
        Pitch shift, not bins.
    bins_per_octave : int
    """
    r = n * bins_per_octave / 12
    x = circshift(x, 0, r)
    ys = ((y + n) % 12) + (int(y) / 12) * 12
    return x, ys

def circshift(x, dim0=0, dim1=0):
    """Circular shift a matrix in two dimensions.

    For example...

          dim0
         aaa|bb      dd|ccc
    dim1 ------  ->  ------
         ccc|dd      bb|aaa

    Default behavior is a pass-through.

    Parameters
    ----------
    x : np.ndarray
        Input 2d matrix.
    dim0 : int
        Rotation along the first axis.
    dim1 : int
        Rotation along the second axis.

    Returns
    -------
    y : np.ndarray
        The circularly shifted matrix.
    """
    # Sanity check
    assert x.ndim == 2

    d0, d1 = x.shape
    z = np.zeros([d0, d1])

    # Make sure the rotation is bounded on [0,d0) & [0,d1)
    dim0, dim1 = dim0 % d0, dim1 % d1
    if not dim0 and dim1:
        z[:, :dim1] = x[:, -dim1:]  # A
        z[:, dim1:] = x[:, :-dim1]  # C
    elif not dim1 and dim0:
        z[:dim0, :] = x[-dim0:, :]  # A
        z[dim0:, :] = x[:-dim0, :]  # B
    elif dim0 and dim1:
        z[:dim0, :dim1] = x[-dim0:, -dim1:]  # A
        z[dim0:, :dim1] = x[:-dim0, -dim1:]  # B
        z[:dim0, dim1:] = x[-dim0:, :-dim1]  # C
        z[dim0:, dim1:] = x[:-dim0, :-dim1]  # D
    else:
        z = x
    return z


def align_array_and_labels(cqt_file, lab_file, framerate):
    cqt = np.load(cqt_file)
    boundaries, labels = load_labfile(lab_file)
    time_points = np.arange(len(cqt), dtype=float) / framerate
    timed_labels = assign_labels_to_time_points(time_points,
                                                           boundaries,
                                                           labels)
    return cqt, timed_labels
