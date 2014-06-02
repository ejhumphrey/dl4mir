import optimus
import numpy as np
import mir_eval.chord as C
from marl.utils.matrix import circshift
from marl.chords.utils import transpose_chord_index

from . import parts_to_index


def chord_sample(length):
    """Slices the CQT of an entity and encodes the chord label in its
    (root, semitones, bass) triple.
    """
    def fx(entity):
        start_idx = np.random.randint(entity.cqt.value.shape[1] - length)
        mid_idx = start_idx + length / 2
        chord_label = entity.chord_labels.value[mid_idx]
        # Fucking billboard...
        if chord_label == 'X':
            return None
        root, semitones, bass = C.encode(chord_label)
        return optimus.Entity(
            cqt=entity.cqt.value[:, start_idx:start_idx + length, :],
            root=root,
            semitones=semitones,
            bass=bass)
    return fx


def pitch_shift(max_pitch_shift=12, bins_per_pitch=3):
    """Apply a random circular shift to the CQT, and rotate the root."""
    def fx(entity):
        values = entity.values
        cqt, root = values.pop("cqt"), values.pop("root")
        max_bin_shift = max_pitch_shift * bins_per_pitch
        bin_shift = np.random.randint(low=-max_bin_shift,
                                      high=max_bin_shift+1)
        cqt = circshift(cqt[0], 0, bin_shift)[np.newaxis, ...]
        pitch_shift = int(bin_shift / bins_per_pitch)
        if root >= 0:
            root = transpose_chord_index(root, pitch_shift)
        return optimus.Entity(cqt=cqt, root=root, **values)
    return fx


def map_to_chroma(entity):
    chroma = C.rotate_bitmap_to_root(entity.semitones.value,
                                     entity.root.value)
    return optimus.Entity(cqt=entity.cqt.value, target_chroma=chroma)


def map_to_tonnetz(entity):

    fifths_theta = 2*np.pi*np.arange(12)/12.0
    minthirds_theta = 2*np.pi*np.arange(12)/4.0
    majthirds_theta = 2*np.pi*np.arange(12)/3.0

    fifths_idx = (np.arange(12)*7) % 12
    fifths_theta = fifths_theta[fifths_idx]

    chroma = C.rotate_bitmap_to_root(entity.semitones.value,
                                     entity.root.value)

    target = np.zeros(6)
    tvect = np.zeros(3, dtype=np.complex)
    for p_idx in chroma.nonzero()[0]:
        tvect[0] = np.exp(1j*fifths_theta[p_idx])
        tvect[1] = np.exp(1j*minthirds_theta[p_idx])
        tvect[2] = np.exp(1j*majthirds_theta[p_idx])

        # tvect /= float(len(pcoll))
        target[::2] = tvect.real
        target[1::2] = tvect.imag

    return optimus.Entity(cqt=entity.cqt.value, target_tonnetz=target)


def map_to_index(vocabulary):
    """
    vocabulary: int
    """
    def fx(entity):
        chord_idx = parts_to_index(
            entity.root.value, entity.semitones.value, vocabulary)
        if chord_idx is None:
            return None
        return optimus.Entity(cqt=entity.cqt.value, chord_idx=chord_idx)
    return fx
