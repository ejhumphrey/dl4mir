"""
"""

import numpy as np
from collections import OrderedDict
from marl.hewey.core import DataPoint
import json

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


def count_chords(lab_files, label_map):
    all_chords = dict()
    for lab_file in lab_files:
        boundaries, labels = load_labfile(lab_file)
        for n, chord_label in enumerate(labels):
            chord_label = label_map.get(chord_label, chord_label)
            if not chord_label in all_chords:
                all_chords[chord_label] = 0.0
            all_chords[chord_label] += (boundaries[n + 1] - boundaries[n])
    keys, values = [], []
    [(keys.append(k), values.append(v)) for k, v in all_chords.iteritems()]
    sorted_index = np.array(values).argsort()[::-1]
    return OrderedDict([(keys[i], values[i]) for i in sorted_index])

def circshift_chord(x, y, n, bins_per_octave, no_chord_index):
    """
    x : np.ndarray
    y : int
    n : int
        Pitch shift, not bins.
    bins_per_octave : int
    no_chord_index : int
    """
    r = n * bins_per_octave / 12
    x = circshift(x, 0, r)
    if y == no_chord_index:
        ys = y
    else:
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


def align_lab_file_to_array(array, lab_file, framerate):
    boundaries, labels = load_labfile(lab_file)
    time_points = np.arange(len(array), dtype=float) / framerate
    timed_labels = assign_labels_to_time_points(time_points,
                                                boundaries,
                                                labels)
    return timed_labels

def load_label_map(filepath):
    """JSON refuses to store integer zeros, so they are written as strings and
    interpreted as integers on load.
    """
    return OrderedDict([(k, int(v)) for k, v in json.load(open(filepath)).iteritems()])

# cid is 32 bit integer 1-4 bits for root, 5-8 bits for bass, 9-20 bits for
# quality, 21-32 bits for extensions(tension)
def chord_int_to_id(chord_int):
    """
    Parameters
    ----------
    chord_int : uint32

    Returns
    -------
    chord_id : str
    """
    BIT_DEPTH = 32
    chord_id = bin(chord_int)[2:]
    num_bits = len(chord_id)
    if num_bits != BIT_DEPTH:
        # Need to fill in leading zeros
        chord_id = "0"*(BIT_DEPTH - num_bits) + chord_id
    return chord_id

qualities = ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6',
             'dim', 'aug', 'sus4', 'sus2', 'hdim7', 'dim7']

quality_map = {'maj':  '100010010000',
               'min':  '100100010000',
               'maj7': '100010010001',
               'min7': '100100010010',
               '7':    '100010010010',
               'maj6': '100010010100',
               'min6': '100100010100',
               'dim':  '100100100000',
               'aug':  '100010001000',
               'sus4': '100001010000',
               'sus2': '101000010000',
               'hdim7':'100100100010',
               'dim7': '100100100100', }

def chord_id_to_index(chord_id, valid_qualities):
    """
    Parameters
    ----------
    chord_id : str
        32-bit vector representation of a chord.
    valid_qualities : list
        Qualities to label; if not found, the chord is labeled as -1.

    Returns
    chord_index : int
        Position in a dense vector for the chord.
    """
    qual_list = [quality_map.get(q) for q in valid_qualities]
    root = int(chord_id[:4], 2)
    qual = chord_id[8:20]
    if qual in qual_list:
        return root + 12 * qual_list.index(qual)
    elif int(chord_id) == 0:
        return len(qual_list) * 12
    else:
        return -1

def bigram_histogram(lab_files, label_map):
    """Compute the histogram of chord transitions from a set of label files.

    Parameters
    ----------
    lab_files : list
        List of lab files.
    label_map : dict
        Mapping from chord names to integers.

    Returns
    histogram : np.ndarray
    """
    num_files = len(lab_files)
    num_classes = len(set(label_map.values()))

    hist = np.zeros([num_classes, num_classes, num_files], dtype=np.int32)
    for n, lab_file in enumerate(lab_files):
        chord_indexes = [label_map.get(y) for y in load_labfile(lab_file)[1]]
        for i in range(len(chord_indexes) - 1):
            hist[chord_indexes[i], chord_indexes[i + 1], n] += 1.0

    return hist
