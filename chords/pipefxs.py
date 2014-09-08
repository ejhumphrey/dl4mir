import numpy as np
from dl4mir.chords import labels

import biggie
from marl.utils.matrix import circshift
from marl.utils.matrix import translate
from marl.chords.utils import transpose_chord_index

import dl4mir.common.util as util

from scipy.spatial.distance import cdist


def _pitch_circshift(entity, pitch_shift, bins_per_pitch):
    values = entity.values()
    cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))

    # Change the chord label if it has a harmonic root.
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


def _pitch_shift(entity, pitch_shift, bins_per_pitch, fill_value=0.0):
    values = entity.values()
    cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))

    # Change the chord label if it has a harmonic root.
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
    cqt = translate(cqt[0], 0, bin_shift, fill_value)[np.newaxis, ...]
    return biggie.Entity(cqt=cqt, chord_label=chord_label, **values)


def pitch_shift(stream, max_pitch_shift=6, bins_per_pitch=3):
    """Apply a random circular shift to the CQT, and rotate the root."""
    for entity in stream:
        if entity is None:
            yield entity
            continue

        # Determine the amount of pitch-shift.
        shift = np.random.randint(low=-max_pitch_shift,
                                  high=max_pitch_shift)
        yield _pitch_shift(entity, shift, bins_per_pitch)


def map_to_chord_index(stream, vocab_dim):
    """
    vocab_dim: int
    """
    for entity in stream:
        if entity is None:
            yield entity
            continue
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        chord_idx = labels.chord_label_to_class_index(chord_label, vocab_dim)
        yield None if chord_idx is None else biggie.Entity(cqt=cqt,
                                                           chord_idx=chord_idx)


def map_to_chroma(stream):
    """
    vocab_dim: int
    """
    for entity in stream:
        if entity is None:
            yield entity
            continue
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        chroma = labels.chord_label_to_chroma(chord_label)
        yield biggie.Entity(cqt=cqt, target_chroma=chroma.squeeze())


def chord_index_to_tonnetz(stream, vocab_dim):
    chord_labels = [labels.index_to_chord_label(n, vocab_dim)
                    for n in range(vocab_dim)]
    T = np.array([labels.chord_label_to_tonnetz(l)
                  for l in chord_labels]).squeeze()
    for entity in stream:
        if entity is None:
            yield entity
            continue
        yield biggie.Entity(cqt=entity.cqt.value,
                            target=T[entity.chord_idx.value])


def map_to_chord_quality_index(stream, vocab_dim):
    """
    vocab_dim: int
    """
    for entity in stream:
        if entity is None:
            yield entity
            continue
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        qual_idx = labels.chord_label_to_quality_index(chord_label, vocab_dim)
        yield None if qual_idx is None else biggie.Entity(cqt=cqt,
                                                          quality_idx=qual_idx)


def chord_index_to_tonnetz_distance(stream, vocab_dim):
    chord_labels = [labels.index_to_chord_label(n, vocab_dim)
                    for n in range(vocab_dim)]
    X = np.array([labels.chord_label_to_tonnetz(l) for l in chord_labels])
    ssm = cdist(X.squeeze(), X.squeeze())
    sn_distance = 1 - ssm / ssm.max()
    for entity in stream:
        if entity is None:
            yield entity
            continue
        yield biggie.Entity(cqt=entity.cqt.value,
                            target=sn_distance[entity.chord_idx.value])


def chord_index_to_affinity_vectors(stream, vocab_dim):
    affinity_vectors = labels.affinity_vectors(vocab_dim)
    for entity in stream:
        if entity is None:
            yield entity
            continue
        yield biggie.Entity(cqt=entity.cqt.value,
                            target=affinity_vectors[entity.chord_idx.value])


def chord_index_to_onehot_vectors(stream, vocab_dim):
    one_hots = np.eye(vocab_dim)
    for entity in stream:
        if entity is None:
            yield entity
            continue
        yield biggie.Entity(cqt=entity.cqt.value,
                            target=one_hots[entity.chord_idx.value])


def map_to_joint_index(stream, vocab_dim):
    """
    vocab_dim: int
    """
    for entity in stream:
        if entity is None:
            yield entity
            continue
        values = entity.values()
        cqt, chord_label = values.pop('cqt'), str(values.pop('chord_label'))
        chord_idx = labels.chord_label_to_class_index(chord_label, vocab_dim)
        if chord_idx is None:
            yield None
            continue
        if chord_idx == vocab_dim - 1:
            root_idx = 13
        else:
            root_idx = chord_idx % 12
        quality_idx = int(chord_idx) / 12

        yield biggie.Entity(cqt=cqt,
                            root_idx=root_idx,
                            quality_idx=quality_idx)


def rotate_chroma_to_root(stream, target_root):
    """Apply a circular shift to the CQT, and rotate the root."""
    for entity in stream:
        if entity is None:
            yield entity
            continue
        chroma = entity.chroma.value.reshape(1, 12)
        chord_label = str(entity.chord_label.value)
        chord_idx = labels.chord_label_to_class_index(chord_label, 157)
        shift = target_root - chord_idx % 12
        # print chord_idx, shift, chord_label
        yield circshift(chroma, 0, shift).flatten()


def rotate_chord_to_root(stream, target_root):
    """Apply a circular shift to the CQT, and rotate the root."""
    for entity in stream:
        if entity is None:
            yield entity
            continue
        chord_label = str(entity.chord_label.value)
        chord_idx = labels.chord_label_to_class_index(chord_label, 157)
        shift = target_root - chord_idx % 12
        # print chord_idx, shift, chord_label
        yield _pitch_shift(entity, shift, 3)


def unpack_contrastive_pairs(stream, vocab_dim, rotate_prob=0.0):
    """
    vocab_dim: int
    """
    for pair in stream:
        if pair is None:
            yield pair
            continue
        pos_entity, neg_entity = pair
        pos_chord_label = str(pos_entity.chord_label.value)
        neg_chord_label = str(neg_entity.chord_label.value)
        pos_chord_idx = labels.chord_label_to_class_index(pos_chord_label,
                                                          vocab_dim)
        neg_chord_idx = labels.chord_label_to_class_index(neg_chord_label,
                                                          vocab_dim)
        if np.random.binomial(1, rotate_prob):
            shift = (pos_chord_idx - neg_chord_idx) % 12
            neg_entity = _pitch_shift(neg_entity, shift, 3)
        # print pos_entity.chord_label.value, neg_entity.chord_label.value
        yield biggie.Entity(cqt=pos_entity.cqt.value,
                            chord_idx=pos_chord_idx, target=np.array([1.0]))
        yield biggie.Entity(cqt=neg_entity.cqt.value,
                            chord_idx=pos_chord_idx, target=np.array([0.0]))


def binomial_mask(stream, max_dropout=0.25):
    for entity in stream:
        if entity is None:
            yield entity
            continue
        p = 1.0 - np.random.uniform(0, max_dropout)
        mask = np.random.binomial(1, p, entity.cqt.value.shape)
        entity.cqt.value = entity.cqt.value * mask
        yield entity


def awgn(stream, mu=0.0, sigma=0.1):
    for entity in stream:
        if entity is None:
            yield entity
            continue
        noise = np.random.normal(mu, sigma, entity.cqt.value.shape)
        entity.cqt.value = entity.cqt.value + noise * np.random.normal(0, 0.25)
        yield entity


def drop_frames(stream, max_dropout=0.1):
    for entity in stream:
        if entity is None:
            yield entity
            continue
        p = 1.0 - np.random.uniform(0, max_dropout)
        mask = np.random.binomial(1, p, entity.cqt.value.shape[1])
        mask[len(mask)/2] = 1.0
        entity.cqt.value = entity.cqt.value * mask[np.newaxis, :, np.newaxis]
        yield entity


def wrap_cqt(stream, length=40, stride=36):
    for entity in stream:
        if entity is None:
            yield entity
            continue
        assert entity.cqt.value.shape[0] == 1
        entity.cqt = util.fold_array(entity.cqt.value[0], length, stride)
        yield entity
