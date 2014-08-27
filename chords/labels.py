import mir_eval
import json
import os
import numpy as np

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


def chord_label_to_class_index_soft(label, vocab_dim=157):
    """Map a chord label to its class index, or None if undefined.

    Parameters
    ----------
    label : str or array_like
        Chord label(s) to map.
    vocab_dim : int
        Number of chords in the vocabulary.
    """
    N_quality_idx = (vocab_dim - 1) / 12
    singleton = False
    if isinstance(label, str):
        label = [label]
        singleton = True
    root, semitones, bass = mir_eval.chord.encode_many(label)
    quality_idx = [semitones_index(s, vocab_dim) for s in semitones]
    class_idx = []
    for r, q in zip(root, quality_idx):
        if N_quality_idx == q:
            idx = vocab_dim - 1
        else:
            idx = None if None in [q, r] else r + q * 12

        class_idx.append(idx)
    # class_idx[N_quality_idx == quality_idx] = vocab_dim - 1
    return class_idx[0] if singleton else class_idx


def chord_label_to_class_index(labels, vocab_dim=157):
    """Map chord labels to class index, or None if undefined.

    Note that this is a strict label mapping;

    Parameters
    ----------
    labels : str or array_like
        Chord label(s) to map.
    vocab_dim : int
        Number of chords in the vocabulary.
    """
    valid_qualities = QUALITIES[vocab_dim]
    singleton = False
    if isinstance(labels, str):
        labels = [labels]
        singleton = True

    index_map = dict()
    for l in np.unique(labels):
        try:
            row = mir_eval.chord.split(l)
        except mir_eval.chord.InvalidChordException:
            row = ['X', '', set(), '']
        skip = [row[0] == 'X',
                not row[1] in valid_qualities,
                len(row[2]) > 0,
                not row[3] in ['', '1']]
        if any(skip):
            idx = None
        elif row[0] == 'N':
            idx = vocab_dim - 1
        else:
            idx = mir_eval.chord.pitch_class_to_semitone(row[0])
            idx += valid_qualities.index(row[1]) * 12
        index_map[l] = idx

    chord_idx = np.array([index_map[l] for l in labels])
    return chord_idx[0] if singleton else chord_idx


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


def index_to_chord_label(index, vocab_dim):
    if index == vocab_dim - 1:
        return "N"
    return "%s:%s" % (ROOTS[index % 12],
                      QUALITIES[vocab_dim][int(index) / 12])


def chord_label_to_chroma(label):
    flatten = False
    if isinstance(label, str):
        label = [label]

    root, semitones, bass = mir_eval.chord.encode_many(label)
    chroma = np.array([mir_eval.chord.rotate_bitmap_to_root(s, r)
                       for s, r in zip(semitones, root)])

    return chroma.flatten() if flatten else chroma


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


def decode(root, semitones, bass):
    root_name = semitone_to_pitch_class(root)


def _load_json_labeled_intervals(label_file):
    data = json.load(open(label_file , 'r'))
    chord_labels = [str(l) for l in data['labels']]
    return np.asarray(data['intervals']), chord_labels


LOADERS = {
    "lab": mir_eval.io.load_labeled_intervals,
    "txt": mir_eval.io.load_labeled_intervals,
    "json": _load_json_labeled_intervals
    }

def load_labeled_intervals(label_file):
    ext = os.path.splitext(label_file)[-1].strip(".")
    assert ext in LOADERS, "Unsupported extension: %s" % ext
    return LOADERS[ext](label_file)


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
    vectors[-1,-1] = 1.0
    return vectors
