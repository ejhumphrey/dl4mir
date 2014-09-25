import mir_eval
import json
import os
import numpy as np
import dl4mir.common.util as util


ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = {
    25: ['maj', 'min', ''],
    61: ['maj', 'min', 'maj7', 'min7', '7', ''],
    157: ['maj', 'min', 'maj7', 'min7', '7',
          'maj6', 'min6', 'dim', 'aug', 'sus4',
          'sus2', 'dim7', 'hdim7', ''],
}

_QINDEX = dict([(v, dict([(tuple(mir_eval.chord.QUALITIES[q]), i)
                          for i, q in enumerate(QUALITIES[v])]))
                for v in QUALITIES])

NO_CHORD = "N"
SKIP_CHORD = "X"

encode = mir_eval.chord.encode
split = mir_eval.chord.split
join = mir_eval.chord.join
pitch_class_to_semitone = mir_eval.chord.pitch_class_to_semitone


def semitone_matrix(vocab_dim):
    return np.array([mir_eval.chord.QUALITIES[q]
                     for q in QUALITIES[vocab_dim]])


def semitone_to_pitch_class(semitone):
    r'''Convert a semitone to its pitch class.

    :parameters:
     - semitone : int
        Semitone value of the pitch class.

    :returns:
    - pitch_class : str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'

    :raises:
     - InvalidChordException
    '''
    return ROOTS[semitone % 12]


def semitones_index(semitones, vocab_dim=157):
    """Return the index of the semitone bitvector, or None if undefined."""
    return _QINDEX[vocab_dim].get(tuple(semitones), None)


def chord_label_to_quality_index(label, vocab_dim=157):
    """Map a chord label to its quality index, or None if undefined."""
    singleton = False
    if isinstance(label, str):
        label = [label]
        singleton = True
    root, semitones, bass = mir_eval.chord.encode_many(label)
    quality_idx = [semitones_index(s, vocab_dim) for s in semitones]
    return quality_idx[0] if singleton else quality_idx


def get_quality_index(semitones, vocab_dim):
    return _QINDEX[vocab_dim].get(tuple(semitones), None)


def chord_label_to_chroma(label):
    flatten = False
    if isinstance(label, str):
        label = [label]

    root, semitones, bass = mir_eval.chord.encode_many(label)
    chroma = np.array([mir_eval.chord.rotate_bitmap_to_root(s, r)
                       for s, r in zip(semitones, root)])

    return chroma.flatten() if flatten else chroma


def rotate(class_vector, root):
    """Rotate a class vector to C (root invariance)"""
    return np.array([class_vector[(n + root) % 12 + 12*(n/12)]
                     for n in range(len(class_vector) - 1)]+[class_vector[-1]])


def subtract_mod12(reference, index):
    """Return the index relative to reference, wrapped inside an octave of 12.

    Note: If 'reference' or `index` is None, this will return `index`.

    Parameters
    ----------
    reference : int
        Reference value.
    index : int
        Value to subtract.
    """
    if None in [reference, index]:
        return index
    ref_idx = reference % 12
    idx = index % 12
    octave = int(index) / 12
    return 12 * octave + (idx - ref_idx) % 12


def _generate_tonnetz_matrix(radii):
    """Return a Tonnetz transform matrix.

    Parameters
    ----------
    radii: array_like, shape=(3,)
        Desired radii for each harmonic subspace (fifths, maj-thirds,
        min-thirds).

    Returns
    -------
    phi: np.ndarray, shape=(12,6)
        Bases for transforming a chroma matrix into tonnetz coordinates.
    """
    assert len(radii) == 3
    basis = []
    for l in range(12):
        basis.append([
            radii[0]*np.sin(l*7*np.pi/6), radii[0]*np.cos(l*7*np.pi/6),
            radii[1]*np.sin(l*3*np.pi/2), radii[1]*np.cos(l*3*np.pi/2),
            radii[2]*np.sin(l*2*np.pi/3), radii[2]*np.cos(l*2*np.pi/3)])
    return np.array(basis)


def chroma_to_tonnetz(chroma, radii=(1.0, 1.0, 0.5)):
    """Return a Tonnetz coordinates for a given chord label.

    Parameters
    ----------
    chroma: str
        Chord label to transform.
    radii: array_like, shape=(3,), default=(1.0, 1.0, 0.5)
        Desired radii for each harmonic subspace (fifths, maj-thirds,
        min-thirds). Default based on E. Chew's spiral model.

    Returns
    -------
    tonnnetz: np.ndarray, shape=(6,)
        Coordinates in tonnetz space for the given chord label.
    """
    phi = _generate_tonnetz_matrix(radii)
    tonnetz = np.dot(chroma, phi)
    scalar = 1 if np.sum(chroma) == 0 else np.sum(chroma)
    return tonnetz / scalar


def chord_label_to_tonnetz(chord_label, radii=(1.0, 1.0, 0.5)):
    chroma = chord_label_to_chroma(chord_label)
    return chroma_to_tonnetz(chroma, radii)


def _load_json_labeled_intervals(label_file):
    """Load labeled intervals from a JSON file.

    Returns
    -------
    intervals : np.ndarray, shape=(N, 2)
        Intervals in time, should be monotonically increasing.
    labels : list, len=N
        String labels corresponding to the given time intervals.
    """
    data = json.load(open(label_file, 'r'))
    chord_labels = [str(l) for l in data['labels']]
    return np.asarray(data['intervals']), chord_labels


LOADERS = {
    "lab": mir_eval.io.load_labeled_intervals,
    "txt": mir_eval.io.load_labeled_intervals,
    "json": _load_json_labeled_intervals
    }


def compress_labeled_intervals(intervals, labels):
    """Collapse repeated labels and the corresponding intervals.

    Parameters
    ----------
    intervals : np.ndarray, shape=(N, 2)
        Intervals in time, should be monotonically increasing.
    labels : list, len=N
        Labels corresponding to the given time intervals.
    """
    intervals = np.asarray(intervals)
    new_labels, new_intervals = list(), list()
    idx = 0
    for label, step in util.run_length_encode(labels):
        new_labels.append(label)
        new_intervals.append([intervals[idx, 0], intervals[idx + step - 1, 1]])
        idx += step
    return np.asarray(new_intervals), new_labels


def load_labeled_intervals(label_file, compress=True):
    ext = os.path.splitext(label_file)[-1].strip(".")
    assert ext in LOADERS, "Unsupported extension: %s" % ext
    intervals, labels = LOADERS[ext](label_file)
    if compress:
        intervals, labels = compress_labeled_intervals(intervals, labels)
    return intervals, labels


_AFFINITY_VECTORS = [
    [(0, 1.0)],
    [(12, 1.0)],
    [(0, 0.75), (15, 0.5), (24, 1.0)],
    [(3, 0.5), (12, 0.75), (36, 1.0), (63, 0.5)],
    [(0, 0.75), (48, 1.0), (88, 0.5)],
    [(0, 0.75), (21, 0.25), (45, 0.5), (60, 1.0)],
    [(12, 0.75), (72, 1.0), (93, 0.25), (153, 0.5)],
    [(56, 0.25), (75, 0.25), (84, 1.0), (132, 0.75), (135, 0.5), (138, 0.5), (141, 0.5), (144, 0.75)],
    [(96, 1.0), (100, 0.5), (104, 0.5)],
    [(108, 1.0), (125, 0.5)],
    [(115, 0.5), (120, 1.0)],
    [(84, 0.75), (87, 0.25), (90, 0.25), (93, 0.25), (132, 1.0), (135, 0.75), (138, 0.75), (141, 0.75)],
    [(15, 0.25), (75, 0.75), (84, 0.5), (144, 1.0)]]


def affinity_vectors(vocab_dim=157):
    """
    Returns
    -------
    targets : np.ndarray, shape=(vocab_dim, vocab_dim)
        Rows contains non-negative class affinity vectors.
    """
    assert vocab_dim == 157, "Being lazy, only 157 currently supported"
    vectors = np.zeros([vocab_dim]*2)
    chord_idx = 0
    for row in _AFFINITY_VECTORS:
        for root in range(12):
            for idx, value in row:
                this_idx = (idx + root) % 12
                this_idx += (int(idx) / 12) * 12
                vectors[chord_idx, this_idx] = value
            chord_idx += 1
    vectors[-1, -1] = 1.0
    return vectors


def count_labels(reference_set, vocab_dim=157):
    labels = dict()
    for labeled_intervals in reference_set.values():
        rootless = [join(*list([''] + list(split(l)[1:])))
                    for l in labeled_intervals['labels']]
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for y, w in zip(rootless, durations):
            if not y in labels:
                labels[y] = 0
            labels[y] += w

    qlabels = labels.keys()
    counts = [labels[y] for y in qlabels]
    idx = np.argsort(counts)[::-1]
    return [qlabels[i] for i in idx], [counts[i] for i in idx]


def count_states(reference_set, vocab_dim=157):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = chord_label_to_class_index(labeled_intervals['labels'],
                                               vocab_dim)
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for y, w in zip(chord_idx, durations):
            s = relative_chord_index(y, y, 157)
            if not s in states:
                states[s] = 0
            states[s] += w

    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def count_bigrams(reference_set, vocab_dim=157):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = chord_label_to_class_index(labeled_intervals['labels'],
                                               vocab_dim)
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for n in range(1, len(chord_idx)):
            s = tuple([relative_chord_index(chord_idx[n],
                                            chord_idx[n + i], 157)
                       for i in range(-1, 1)])
            if not s in states:
                states[s] = 0
            states[s] += durations[n]
    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def count_trigrams(reference_set, vocab_dim=157):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = chord_label_to_class_index_soft(
            labeled_intervals['labels'], vocab_dim)
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for n in range(1, len(chord_idx) - 1):
            s = tuple([relative_chord_index(chord_idx[n],
                                            chord_idx[n + i], 157)
                       for i in range(-1, 2)])
            if not s in states:
                states[s] = 0
            states[s] += durations[n]
    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def sequence_to_bigrams(seq, previous_state):
    bigrams = [(previous_state, seq[0])]
    for n in range(1, len(seq)):
        bigrams.append(tuple([seq[n + i] for i in range(-1, 1)]))
    return bigrams


def sequence_to_trigrams(seq, start_state, end_state):
    trigrams = [(start_state, seq[0], seq[1])]
    for n in range(1, len(seq) - 1):
        trigrams.append(tuple([seq[n + i] for i in range(-1, 2)]))
    trigrams.append((seq[-2], seq[-1], end_state))
    return trigrams
