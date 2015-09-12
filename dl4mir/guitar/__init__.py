
import json
import os

ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = {
    25: ['maj', 'min'],
    61: ['maj', 'min', 'maj7', 'min7', '7'],
    157: ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6', 'dim', 'aug',
          'sus4', 'sus2', 'aug', 'dim7', 'hdim7'],
}

DRIVER_ARGS = dict(
    max_iter=500000,
    save_freq=2500,
    print_freq=100)


def __load_voicings__():
    fpath = os.path.join(os.path.split(__file__)[0],
                         'chord_voicings.json')
    return json.load(open(fpath))


VOICINGS = __load_voicings__()
