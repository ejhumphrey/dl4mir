import mir_eval.chord as _chord

QUALITIES = {
    61: ['maj', 'min', 'maj7', 'min7', '7'],
    157: ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6', 'dim', 'aug',
          'sus4', 'sus2', 'aug', 'dim7', 'hdim7']
}

_QINDEX = dict([(v, dict([(tuple(_chord.QUALITIES[q]), i)
                          for i, q in enumerate(QUALITIES[v])]))
                for v in QUALITIES])


def get_quality_index(semitomes, vocab):
    return _QINDEX[vocab].get(tuple(semitomes), None)


def parts_to_index(root, semitones, vocab):
    if root < 0:
        return vocab - 1
    q_idx = get_quality_index(semitones, vocab)
    if q_idx is None:
        return q_idx
    return root + q_idx*12
