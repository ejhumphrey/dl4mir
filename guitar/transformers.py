import optimus
import numpy as np
import pychords.guitar as G
from marl.utils.matrix import circshift


def cqt_sample(length):
    """Slices the CQT of an entity and decodes the fret label."""
    def fx(entity):
        start_idx = np.random.randint(entity.cqt.value.shape[1] - length)
        mid_idx = start_idx + length / 2
        fret_label = entity.fret_labels.value[mid_idx]
        # Drop erroneous labels
        if fret_label == 'X':
            return None
        return optimus.Entity(
            cqt=entity.cqt.value[:, start_idx:start_idx + length, :],
            fret_indexes=G.decode(fret_label))
    return fx


def pitch_shift(max_frets, bins_per_pitch=3):
    """Apply a random circular shift to the CQT, and rotate the root."""
    def fx(entity):
        values = entity.values
        cqt = values.pop("cqt")
        frets = entity.fret_indexes.value
        # print "Base frets: %s" % frets
        if not 0 in frets and not frets.mean() == -1:
            fret_min = frets[frets > 0].min()
            fret_max = max_frets - frets[frets > 0].max()
            # print "\tBounds: (-%d, %d)" % (fret_min, fret_max)
            offset = np.ones(6, dtype=int)*np.greater(frets, 0)
            bin_shift = np.random.randint(low=-fret_min*bins_per_pitch,
                                          high=fret_max*bins_per_pitch)
            # print "\tBin shift: %d" % bin_shift
            cqt = circshift(cqt[0], 0, bin_shift)[np.newaxis, ...]
            frets += int(bin_shift / bins_per_pitch) * offset
            # print "\tFret shift: %d" % int(bin_shift / bins_per_pitch)

        return optimus.Entity(cqt=cqt, fret_indexes=frets)
    return fx


def fret_indexes_to_bitmap(fret_dim):
    """
    vocabulary: int
    """
    def fx(entity):
        fret_bitmap = np.zeros([6, fret_dim])
        fret_bitmap[np.arange(6), entity.fret_indexes.value] = 1.0
        return optimus.Entity(cqt=entity.cqt.value, fret_bitmap=fret_bitmap)
    return fx
