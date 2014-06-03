
ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = {
    25: ['maj', 'min'],
    61: ['maj', 'min', 'maj7', 'min7', '7'],
    157: ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6', 'dim', 'aug',
          'sus4', 'sus2', 'aug', 'dim7', 'hdim7'],
}

DRIVER_ARGS = dict(
    max_iter=1000,
    save_freq=200,
    print_freq=50)

SOURCE_ARGS = dict(
    batch_size=50,
    refresh_prob=0.0,
    cache_size=500)
