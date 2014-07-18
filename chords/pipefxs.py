import numpy as np
from dl4mir.chords import labels

import biggie
from marl.utils.matrix import circshift
from marl.chords.utils import transpose_chord_index


def pitch_shift(max_pitch_shift=12, bins_per_pitch=3):
    """Apply a random circular shift to the CQT, and rotate the root."""
    def fx(entity):
        # assert "cqt" in entity
        # assert "chord_label" in entity

        values = entity.values()
        cqt, chord_label = values.pop('cqt'), values.pop('chord_label')

        # Determine the amount of pitch-shift.
        pitch_shift = np.random.randint(low=-max_pitch_shift,
                                        high=max_pitch_shift)

        # Change the chord label if it has a harmonic root.
        chord_label = str(entity.chord_label.value)
        if not chord_label in [labels.NO_CHORD, labels.SKIP_CHORD]:

            root, quality, exts, bass = labels.split(chord_label)
            root = (labels.pitch_class_to_semitone(root) + pitch_shift) % 12
            new_root = labels.semitone_to_pitch_class(root)
            new_label = labels.join(new_root, quality, exts, bass)
            # print "Input %12s // Shift: %3s // Output %12s" % \
            #     (chord_label, pitch_shift, new_label)
            chord_label = new_label

        # Always rotate the CQT.
        bin_shift = pitch_shift*bins_per_pitch
        cqt = circshift(cqt[0], 0, bin_shift)[np.newaxis, ...]
        return biggie.Entity(cqt=cqt, chord_label=chord_label, **values)
    return fx


def map_to_chord_index(vocab_dim):
    """
    vocabulary: int
    """
    def fx(entity):
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        chord_idx = labels.chord_label_to_class_index(chord_label, vocab_dim)
        if chord_idx is None:
            return None
        return biggie.Entity(cqt=cqt, chord_idx=chord_idx)
    return fx


def map_to_chord_quality_index(vocab_dim):
    """
    vocabulary: int
    """
    def fx(entity):
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        qual_idx = labels.chord_label_to_quality_index(chord_label, vocab_dim)
        if qual_idx is None:
            return None
        return biggie.Entity(cqt=cqt, quality_idx=qual_idx)
    return fx