import numpy as np
import itertools

import biggie
import pescador

from dl4mir.chords import labels
import dl4mir.chords.pipefxs as FX
from dl4mir.common import util

import scipy.cluster.vq as VQ
from sklearn.decomposition import PCA
import time

import marl.fileutils as futil
import os


def slice_chord_entity(entity, length, idx=None):
    """Return a windowed slice of a chord Entity.

    Parameters
    ----------
    entity : Entity, with at least {cqt, chord_labels} fields
        Observation to window.
        Note that entity.cqt is shaped (num_channels, num_frames, num_bins).
    length : int
        Length of the sliced array.
    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Returns
    -------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    cqt_shape = entity.cqt.value.shape
    idx = np.random.randint(cqt_shape[1]) if idx is None else idx
    start_idx = idx - length / 2
    end_idx = start_idx + length

    cqt = np.zeros([cqt_shape[0], length, cqt_shape[2]])
    if start_idx < 0:
        start_idx = np.abs(start_idx)
        cqt[:, start_idx:, :] = entity.cqt.value[:, :end_idx, :]
    elif end_idx > cqt_shape[1]:
        end_idx = cqt_shape[1] - start_idx
        cqt[:, :end_idx, :] = entity.cqt.value[:, start_idx:, :]
    else:
        cqt[:, :, :] = entity.cqt.value[:, start_idx:end_idx, :]

    return biggie.Entity(cqt=cqt, chord_label=entity.chord_labels.value[idx])


def chord_sampler(key, stash, win_length=20, index=None, max_samples=None):
    """Generator for sampling windowed chord observations from an entity.

    Parameters
    ----------
    key : str
        Key for the entity of interest; must be consistent across both `stash`
        and `index`, when the latter is provided.
    stash : dict_like
        Dict or biggie.Stash of chord entities.
    win_length: int
        Length of centered observation window for the CQT.
    index: dict of index arrays, default=None
        Indexing object for constrained sampling of the chord entity.
        If provided, must have a np.ndarray of integers under `key`; otherwise,
        this method will fail.
    max_samples: int, or None
        Maximum number of samples to return from this Generator; if None, runs
        indefinitely.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    entity = stash.get(key)
    num_samples = len(entity.chord_labels.value)
    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    idx = np.inf
    max_samples = np.inf if max_samples is None else max_samples
    count = 0
    while count < max_samples and len(valid_samples):
        if idx >= len(valid_samples):
            np.random.shuffle(valid_samples)
            idx = 0
        yield slice_chord_entity(entity, win_length, valid_samples[idx])
        idx += 1
        count += 1


def chord_stepper(key, stash, win_length=20, index=None):
    """Generator for stepping windowed chord observations from an entity.

    Parameters
    ----------
    key : str
        Key for the entity of interest; must be consistent across both `stash`
        and `index`, when the latter is provided.
    stash : dict_like
        Dict or biggie.Stash of chord entities.
    win_length: int
        Length of centered observation window for the CQT.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    entity = stash.get(key)
    num_samples = len(entity.chord_labels.value)
    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    idx = 0
    count = 0
    while count < len(valid_samples):
        yield slice_chord_entity(entity, win_length, valid_samples[idx])
        idx += 1
        count += 1


def chord_map(entity, vocab_dim=157):
    """Map an entity's chord_labels to quality indexes; for partitioning."""
    chord_labels = entity.chord_labels.value
    unique_labels = np.unique(chord_labels)
    unique_idxs = labels.chord_label_to_class_index(unique_labels)
    labels_to_index = dict([(l, i) for l, i in zip(unique_labels,
                                                   unique_idxs)])
    return np.array([labels_to_index[l] for l in chord_labels], dtype=object)


def quality_map(entity, vocab_dim=157):
    """Map an entity's chord_labels to quality indexes; for partitioning."""
    chord_labels = entity.chord_labels.value
    unique_labels = np.unique(chord_labels)
    unique_idxs = labels.chord_label_to_quality_index(unique_labels)
    labels_to_index = dict([(l, i) for l, i in zip(unique_labels,
                                                   unique_idxs)])
    return np.array([labels_to_index[l] for l in chord_labels], dtype=object)


def create_chord_stream(stash, win_length, pool_size=50, vocab_dim=157,
                        pitch_shift=True):
    """Return an unconstrained stream of chord samples."""
    entity_pool = [pescador.Streamer(chord_sampler, key, stash, win_length)
                   for key in stash.keys()]
    stream = pescador.mux(entity_pool, None, pool_size, lam=25)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_chord_index(stream, vocab_dim)


def create_stash_stream(stash, win_length, pool_size=50, vocab_dim=157,
                        pitch_shift=False):
    """Stream the contents of a stash."""
    partition_labels = util.partition(stash, quality_map)
    quality_index = util.index_partition_arrays(partition_labels, range(14))

    entity_pool = [pescador.Streamer(chord_stepper, key,
                                     stash, win_length, quality_index)
                   for key in stash.keys()]
    stream = pescador.mux(entity_pool, None, pool_size)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_chord_index(stream, vocab_dim)


def create_uniform_quality_stream(stash, win_length, partition_labels=None,
                                  pool_size=50, vocab_dim=157,
                                  pitch_shift=True, valid_idx=None):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, quality_map)

    if valid_idx is None:
        valid_idx = range(14)

    quality_pool, weights = [], []
    for qual_idx in valid_idx:
        weights.append(1 if qual_idx == 13 else 12)
        quality_subindex = util.index_partition_arrays(
            partition_labels, [qual_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, quality_subindex)
                       for key in quality_subindex.keys()]
        stream = pescador.mux(entity_pool, n_samples=None, k=25, lam=20)
        quality_pool.append(pescador.Streamer(stream))

    weights = np.array(weights, dtype=float) / np.sum(weights)
    stream = pescador.mux(quality_pool, n_samples=None, k=pool_size, lam=None,
                          with_replacement=False, pool_weights=weights)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_chord_index(stream, vocab_dim)


def uniform_quality_chroma_stream(stash, win_length, partition_labels=None,
                                  pool_size=50, pitch_shift=True):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, quality_map)

    quality_pool = []
    for qual_idx in range(14):
        quality_subindex = util.index_partition_arrays(
            partition_labels, [qual_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, quality_subindex)
                       for key in quality_subindex.keys()]
        stream = pescador.mux(entity_pool, n_samples=None, k=25, lam=20)
        quality_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(quality_pool, n_samples=None, k=pool_size,
                          lam=None, with_replacement=False)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_chroma(stream)


def create_uniform_factored_stream(stash, win_length, partition_labels=None,
                                   pool_size=50, vocab_dim=157,
                                   pitch_shift=True):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, quality_map)

    quality_pool = []
    for qual_idx in range(13):
        quality_subindex = util.index_partition_arrays(
            partition_labels, [qual_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, quality_subindex)
                       for key in quality_subindex.keys()]
        stream = pescador.mux(entity_pool, n_samples=None, k=25, lam=20)
        quality_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(quality_pool, n_samples=None, k=pool_size,
                          lam=None, with_replacement=False)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_joint_index(stream, vocab_dim)


def create_contrastive_quality_stream(stash, win_length,
                                      partition_labels=None, pool_size=50,
                                      vocab_dim=157, pitch_shift=True):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, quality_map)

    quality_streams = []
    for qual_idx in range(14):
        quality_subindex = util.index_partition_arrays(
            partition_labels, [qual_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, quality_subindex)
                       for key in quality_subindex.keys()]
        qstream = pescador.mux(entity_pool, None, 25)
        if pitch_shift:
            qstream = FX.pitch_shift(qstream)
        quality_streams.append(qstream)

    quality_streams = np.array(quality_streams)
    binary_pool = []
    for qual_idx in range(14):
        neg_mask = np.ones(14, dtype=bool)
        neg_mask[qual_idx] = False
        quality_pool = [pescador.Streamer(x)
                        for x in quality_streams[neg_mask]]
        neg_stream = pescador.mux(quality_pool, n_samples=None,
                                  k=len(quality_pool), lam=None,
                                  with_replacement=False)
        pair_stream = itertools.izip(quality_streams[qual_idx], neg_stream)
        binary_pool.append(pescador.Streamer(pair_stream))

    cstream = pescador.mux(binary_pool, n_samples=None, k=len(binary_pool),
                           lam=None, with_replacement=False)
    return FX.unpack_contrastive_pairs(cstream, vocab_dim)


def chroma_stepper(key, stash, index=None):
    entity = stash.get(key)
    num_samples = len(entity.chord_labels.value)
    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    idx = 0
    count = 0
    while count < len(valid_samples):
        n = valid_samples[idx]
        if n >= entity.chroma.value.shape[0]:
            print "Out of range! %s" % key
            break
        yield biggie.Entity(chroma=entity.chroma.value[n],
                            chord_label=entity.chord_labels.value[n])
        idx += 1
        count += 1
