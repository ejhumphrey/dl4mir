import optimus
import numpy as np
import mir_eval.chord as C
from marl.utils.matrix import circshift
from marl.chords.utils import transpose_chord_index


def sample(length):
    """Slices the CQT of an entity and encodes the chord label in its
    (root, semitones, bass) triple.
    """
    def fx(entity):
        i0 = (entity.cqt.value.shape[1] - length) / 2
        root, semitones, bass = C.encode(entity.chord_labels.value[i0])
        return optimus.Entity(cqt=entity.cqt.value[:, i0:i0+length],
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
    return optimus.Entity(cqt=entity.cqt.value, chroma=chroma)


def map_to_tonnetz(entity):
    raise NotImplementedError("Write me!")


def map_to_index(vocabulary):
    def fx(entity):
        pass
    pass
