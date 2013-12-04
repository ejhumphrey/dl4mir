"""
"""

from collections import OrderedDict
import numpy as np
import csv
import os

from matplotlib.pyplot import figure, show
from matplotlib import cm as ColorMaps
from scipy.signal.windows import gaussian

NO_CHORD = "N"
pitch_classes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

qualities = ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6',
             'dim', 'aug', 'sus4', 'sus2', 'hdim7', 'dim7']

mirex_map = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

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
        ys = rotate_label_pitch_class(y, n)
    return x, ys


def rotate_label_pitch_class(y, n):
    """Rotate the pitch class of a chord label.

    Parameters
    ----------
    y : int
        Chord label index.
    n : int
        Number of semitones to rotate the chord.

    Returns
    -------
    ys : int
        The rotated chord label.
    """
    return ((y + n) % 12) + (y / 12) * 12



def align_chord_qualities(y_true, y_pred, nochord_index):
    """
    For np.ndarrays
    """
    pitch_shifts = y_true % 12
    y_trues = rotate_label_pitch_class(y_true, -pitch_shifts)
    y_trues[y_true == nochord_index] = nochord_index
    y_trues[y_true < 0] = -1
    y_preds = rotate_label_pitch_class(y_pred, -pitch_shifts)
    y_preds[y_pred == nochord_index] = nochord_index
    y_preds[y_pred < 0] = -1
    return y_trues, y_preds


def rotate_a_relatve_to_b(a, b, nochord_index):
    """
    """
    # No-chord passthrough
    if b == nochord_index or a == nochord_index:
        return a, b
    if b < 0 or a < 0:
        return a, b
    pitch_shift = b % 12
    b = rotate_label_pitch_class(b, -pitch_shift)
    a = rotate_label_pitch_class(a, -pitch_shift)
    return a, b


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



def load_guitar_chords(csv_file, qualities):
    fret_map = {"x":0, "0":1, "1":2, "2":3, "3":4, "4":5, "5":6, "6":7, "7":8}


    chord_shapes = [line for line in csv.reader(open(csv_file))][1:-1]
    num_chords = len(qualities) * 12 + 1
    string_idx = np.arange(6)[::-1]
    templates = np.zeros([num_chords, len(fret_map), 6])

    # No-chord is always last
    templates[-1, fret_map.get("x"), string_idx] = 1.0

    for chord in chord_shapes:
        # Index 0 is the root, 1 is the quality, 2:8 are frets
        assert chord[0] in pitch_classes, "Pitch class '%s' unsupported?" % chord[0]
        if not chord[1] in qualities:
#            print "Skipping '%s:%s'." % (chord[0], chord[1])
            continue
        chord_idx = pitch_classes.index(chord[0]) + qualities.index(chord[1]) * 12
        fret_idx = np.array([fret_map.get(f) for f in chord[2:8]])
        templates[chord_idx, fret_idx, string_idx] = 1.0

    return templates


def pair_cqts_and_labs(split_file, cqt_dir, lab_dir):
    cqts, labs = [], []
    for l in open(split_file):
        fbase = l.strip("\n")
        cqt_file = os.path.join(cqt_dir, "%s.npy" % fbase)
        assert os.path.exists(cqt_file)
        lab_file = os.path.join(lab_dir, "%s.lab" % fbase)
        assert os.path.exists(lab_file)
        cqts.append(cqt_file)
        labs.append(lab_file)

    return cqts, labs


def write_paired_cqt_labs(split_file, cqt_dir, lab_dir, output_file):
    cqts, labs = pair_cqts_and_labs(split_file, cqt_dir, lab_dir)
    fout = open(output_file, 'w')
    for cqt_file, lab_file in zip(cqts, labs):
        fout.write("%s\t%s\n" % (cqt_file, lab_file))
    fout.close()


def draw_guitar_chords(templates):
    """
    templates : np.ndarray

    """
    rows = 12
    columns = 5
    frets = 9
    strings = 6
    num_chords = rows * columns
    fig = figure()
    chord_idx = np.arange(num_chords).reshape(columns, rows).T
    ax = fig.gca()
    x_starts = np.arange(columns) * 9.5
    y_starts = np.arange(rows)[::-1] * 2.4
    width = 0.9
    height = 0.3
    for n in range(rows):
        for m in range(columns):
            chord = np.flipud(templates[:, chord_idx[n, m]].reshape(frets, strings).T)
            draw_fretboard(chord, x_starts[m], y_starts[n], width, height, ax)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim(x_starts[0], x_starts[-1] + 8.9)
    ax.set_ylim(y_starts[-1], y_starts[0] + 1.8)


def draw_fretboard(C, x, y, width, height, ax):
    """
    C : np.array
        Bit-map template, strings by frets
    x: scalar
        Lower-left coordinate.
    y: scalar
        Lower-left coordinate.
    width: scalar
        Width of each fret.
    height: scalar
        Height of each fret.
    ax: Matplotlib ax (instantiated).
        Axes to draw on.
    """
    fretboard_color = np.array([34., 29., 28.]) / 256.0
    fret_color = np.array([150., 150., 175.]) / 256.0
    strings, frets = C.shape
    notes = C.argmax(axis=1)
    for m in range(strings):
        xranges = [(width * n + x, width) for n in range(frets)]
        ax.broken_barh(xranges,
                       (y + m * height, height),
                       facecolors=fretboard_color,
                       edgecolors=fret_color)
        dot_x = (notes[m] + 0.5) * width + x
        dot_y = (m + 0.5) * height + y
        if notes[m] == 0:
            ax.scatter(dot_x, dot_y, marker='x', s=40, facecolors='white')
        else:
            ax.scatter(dot_x, dot_y, marker='o', s=40, facecolors='white')

def count_relative_transitions(lab_file, label_map, framerate=None):
    boundaries, labels = load_labfile(lab_file)
    if not framerate is None:
        time_points = np.arange(0, int(framerate * boundaries[-1])) / framerate
        labels = assign_labels_to_time_points(time_points, boundaries, labels)

    y = np.array([label_map.get(l, -1) for l in labels])
    q = np.floor(y / 12.0).astype(int)
    dq = np.floor(np.diff(y) / 12.0).astype(int)
    dr = np.diff(y) % 12

    num_classes = label_map["N"] + 1
    num_qualities = int(label_map["N"] / 12) + 1
    counts = np.zeros([num_qualities, num_classes], dtype=np.float)
    for n in range(len(y) - 1):
        # If either are out of the lexicon, skip
        if y[n] < 0 or y[n + 1] < 0:
            continue
        # If both "N", increment
        elif y[n] == y[n + 1] == label_map["N"]:
            counts[-1, label_map["N"]] += 1.0
        # If the
        elif y[n] == label_map["N"] != y[n + 1]:
            counts[-1, :-1] += 1.0 / float(label_map["N"])
        elif y[n + 1] == label_map["N"]:
            counts[q[n], -1] += 1.0
        else:
            # Transition indexes in the root relative to C
            t_idx = 12 * (q[n] + dq[n]) + dr[n]
            counts[q[n], t_idx] += 1.0

    return counts

def cumulative_transition_matrix(lab_files, label_map, framerate=None):
    qual_counts = np.array([count_relative_transitions(
        l, label_map, framerate) for l in lab_files]).sum(axis=0)

    scale = qual_counts.sum(axis=1)[:, np.newaxis]
    return qual_counts / scale

def cumulative_prior(lab_files, label_map):
    num_classes = label_map["N"] + 1
    prior = np.zeros(num_classes)
    for lab_file in lab_files:
        boundaries, labels = load_labfile(lab_file)
        y = label_map.get(labels[0], -1)
        if y >= 0:
            prior[y] += 1

    return prior / prior.sum()


def rotate_quality_transitions(qual_counts):
    num_chords = qual_counts.shape[1]
    num_qualities = qual_counts.shape[0]
    chord_counts = np.zeros([num_chords, num_chords], dtype=np.float)
    root = (np.arange(12)[:, np.newaxis] + np.arange(num_chords)[np.newaxis, :]) % 12
    # For no-chord
    root[:, -1] = 0
    root += 12 * np.floor(np.arange(num_chords) / 12)[np.newaxis, :]
    for q in range(num_qualities - 1):
        for idx in root:
#            print idx.shape, qual_counts[q]
            chord_counts[idx[0] + 12 * q, idx] = qual_counts[q]
    chord_counts[-1, :] = qual_counts[-1]
    return chord_counts

def viterbi_alg(posterior, trans_mat, prior=None, rho=0):
    """
    Run a tempo-minded variant of the Viterbi algorithm over the given
    matrix X.

    """
    def log(x):
        return np.log(x + np.power(2.0, -10.0))
    num_obs, num_states = posterior.shape
    probs = posterior / posterior.sum(axis=1)[:, np.newaxis]

    if prior is None:
        prior = np.ones(num_states) / float(num_states)
    elif prior == "histogram":
        prior = probs.sum(axis=0)
        prior /= prior.sum()

    V = [log(probs[0]) + log(prior)]
    path = []

    penalty = (np.eye(num_states, dtype=np.float) - 1) * rho
    for n in range(1, num_obs):
        Vn = log(probs[n]).reshape(1, num_states)
        Vm = V[-1].reshape(num_states, 1)
        V += [(Vm + (log(trans_mat) - penalty) + Vn).max(axis=0)]
        path += [(Vm + log(trans_mat) + Vn).argmax(axis=0)]

    path_idx = np.argmax(V[-1])
    path += [path[-1]]
    return np.asarray(path)[:, path_idx]


def fwrr(y_true, y_pred):
    return np.mean(y_true == y_pred)

def aica(qual_true, qual_pred, num_quals):
    quals = 12 * np.arange(num_quals, dtype=int)
    return np.mean([(qual_true[qual_true == qidx] == qual_pred[qual_true == qidx]).mean() for qidx in quals])

def draw_posterior(posterior, y_true, y_pred=None):
    if y_pred is None:
        y_pred = posterior.argmax(axis=1)

    fig = figure()
    ax = fig.gca()
    ax.imshow(posterior.T, interpolation='nearest', aspect='auto')
    ax.plot(y_true, 'w', alpha=0.6, linewidth=3)
    ax.plot(y_pred, 'y', alpha=0.6, linewidth=3)
    ax.set_xlim(0, len(y_true))
    ax.set_ylim(0, posterior.shape[1])

def chord_viterbi(posterior, trans_mat,
                  chord_self=0, nochord_self=0, chord_to_nochord=0, nochord_to_chord=0):
    num_states = len(trans_mat)
    T_c2n = np.zeros(trans_mat.shape)
    T_c2n[:-1, -1] = 1.0
    ST_c = np.zeros(trans_mat.shape)
    ST_c[np.arange(num_states - 1), np.arange(num_states - 1)] = 1.0
    ST_n = np.zeros(trans_mat.shape)
    ST_n[-1, -1] = 1.0
    T_n2c = np.zeros(trans_mat.shape)
    T_n2c[-1, :-1] = 1.0

    log_trans_mat = np.log(trans_mat + 0.00001)
    log_trans_mat -= chord_self * ST_c.T
    log_trans_mat -= nochord_self * ST_n.T
    log_trans_mat -= chord_to_nochord * T_c2n.T
    log_trans_mat -= nochord_to_chord * ST_c.T
    return viterbi_alg(posterior, np.exp(log_trans_mat))

def chord_viterbi2(posterior, trans_mat, rho, nochord_thresh=0.1):
    num_states = posterior.shape[1]
    y_pred = viterbi_alg(posterior[:, :-1], trans_mat, rho=rho)
    y_pred[posterior[:, -1] > nochord_thresh] = num_states - 1
    return y_pred


def wrap_chord_idx_to_MIREX(cidx, nochord_idx):
    if cidx == nochord_idx:
        return 24
    if cidx < 0:
        return -1
    q = int(np.floor(cidx / 12))
    r = int(cidx % 12)
    return r + mirex_map[q] * 12

def collect_posterior_splits(posterior_dir, train_split, test_split, lab_dir):
    ftrain = open(os.path.join(posterior_dir, "train.txt"), 'w')
    ftest = open(os.path.join(posterior_dir, "test.txt"), 'w')
    for l in open(train_split):
        l = l.strip("\n")
        ftrain.write("%s\t%s\n" % (os.path.join(posterior_dir, "%s.npy" % l),
                                    os.path.join(lab_dir, "%s.lab" % l)))
    for l in open(test_split):
        l = l.strip("\n")
        ftest.write("%s\t%s\n" % (os.path.join(posterior_dir, "%s.npy" % l),
                                   os.path.join(lab_dir, "%s.lab" % l)))
    ftrain.close()
    ftest.close()


def generate_cqt_mask(num_points, input_shape, noise_shape, alpha):
    Ii, Ij = input_shape
    Ni, Nj = noise_shape

    mask = np.ones([Ii + 2 * Ni, Ij + 2 * Nj])
    gaus_mask = make_gaussian_mask(noise_shape, alpha)

    for x, y in np.random.uniform(size=(num_points, 2)):
        i, j = int(x * (Ii + Ni)), int(y * (Ij + Nj))
        mask[i:i + Ni, j:j + Nj] *= gaus_mask

    return mask[Ni:Ni + Ii, Nj:Nj + Ij]


def make_gaussian_mask(gdim, alpha):
    """I'm an empty docstring"""
    G = [gaussian(gd, gd * alpha, True) for gd in gdim]
    G = G[0][:, np.newaxis] * G[1][np.newaxis, :]
    return 1.0 - G


def mean_pool(x_in, bins):
    """I'm an empty docstring"""
    num_frames = len(bins) - 1
    z_out = np.zeros([num_frames, x_in.shape[1]])
    for idx in range(num_frames):
        z_out[idx] = np.mean(x_in[bins[idx]:bins[idx+1]], axis=0)
    return z_out


def median_pool(x_in, bins):
    """I'm an empty docstring"""
    num_frames = len(bins) - 1
    z_out = np.zeros([num_frames, x_in.shape[1]])
    for idx in range(num_frames):
        z_out[idx] = np.median(x_in[bins[idx]:bins[idx+1]], axis=0)
    return z_out


def median_boundary_filter(x_in, bins):
    """I'm an empty docstring"""
    num_frames = len(bins) - 1
    z_out = np.zeros(x_in.shape)
    for idx in range(num_frames):
        z_out[bins[idx]:bins[idx+1]] = np.median(x_in[bins[idx]:bins[idx+1]], axis=0)
    return z_out


def majority(values, fail_on_tie=False):
    """I'm an empty docstring!
    """
    value_map = dict([(v, enum) for enum, v in enumerate(set(values))])
    inverse_map = dict([(v, k) for k, v in value_map.iteritems()])
    enum_values = [value_map[v] for v in values]
    value_counts = np.bincount(enum_values)
    idx = value_counts.argsort()[::-1]
    if fail_on_tie and len(value_map) > 1:
        top_1 = inverse_map[idx[0]]
        top_2 = inverse_map[idx[1]]
        assert value_counts[idx[0]] > value_counts[idx[1]], \
            "First-place tie between '%s' and '%s'." % (top_1, top_2)
    return inverse_map[idx[0]]


def majority_pool(y_in, bins, fail_on_tie=True):
    """I'm an empty docstring"""
    num_frames = len(bins) - 1
    return [majority(y_in[bins[idx]:bins[idx+1]], fail_on_tie) for idx in range(num_frames)]

