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


def chord_label_to_class_index(label, vocab_dim=157):
    """Map a chord label to its class index, or None if undefined."""
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
            class_idx.append(vocab_dim - 1)
        else:
            class_idx.append(None if None in [q, r] else r + q * 12)
    # class_idx[N_quality_idx == quality_idx] = vocab_dim - 1
    return class_idx[0] if singleton else class_idx


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


def chord_label_to_tonnetz(label):
    raise NotImplementedError("Come back to this if / when it matters")


def decode(root, semitones, bass):
    root_name = semitone_to_pitch_class(root)


def _load_json_labeled_intervals(label_file):
    data = json.load(open(label_file , 'r'))
    return np.asarray(data['intervals']), data['labels']


LOADERS = {
    "lab": mir_eval.io.load_labeled_intervals,
    "txt": mir_eval.io.load_labeled_intervals,
    "json": _load_json_labeled_intervals
    }

def load_labeled_intervals(label_file):
    ext = os.path.splitext(label_file)[-1].strip(".")
    assert ext in LOADERS, "Unsupported extension: %s" % ext
    return LOADERS[ext](label_file)
