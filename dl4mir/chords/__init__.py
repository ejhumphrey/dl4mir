import mir_eval.chord as _chord

ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = {
    25: ['maj', 'min'],
    61: ['maj', 'min', 'maj7', 'min7', '7'],
    157: ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6', 'dim', 'aug',
          'sus4', 'sus2', 'dim7', 'hdim7'],
}

_QINDEX = dict([(v, dict([(tuple(_chord.QUALITIES[q]), i)
                          for i, q in enumerate(QUALITIES[v])]))
                for v in QUALITIES])


def get_quality_index(semitones, vocab):
    return _QINDEX[vocab].get(tuple(semitones), None)


def parts_to_index(root, semitones, vocab):
    if root < 0:
        return vocab - 1
    q_idx = get_quality_index(semitones, vocab)
    return None if q_idx is None else root + q_idx*12


def chord_label_to_index(label, vocab):
    if label == "N":
        return vocab - 1
    root, semitones, bass = _chord.encode(label)
    return parts_to_index(root, semitones, vocab)


def index_to_chord_label(index, vocab):
    if index == vocab - 1:
        return "N"
    return "%s:%s" % (ROOTS[index % 12], QUALITIES[157][int(index) / 12])

DRIVER_ARGS = dict(
    max_iter=100000,
    save_freq=250,
    print_freq=100)

SOURCE_ARGS = dict(
    batch_size=50,
    refresh_prob=0.1,
    cache_size=500)
