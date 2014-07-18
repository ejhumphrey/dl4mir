import numpy as np

import biggie
import pescador

from dl4mir.chords import labels
from dl4mir.common import util


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


def quality_map(entity):
    """Map an entity's chord_labels to quality indexes; for partitioning."""
    chord_labels = entity.chord_labels.value
    unique_labels = np.unique(chord_labels)
    unique_idxs = labels.chord_label_to_quality_index(unique_labels)
    labels_to_index = dict([(l, i) for l, i in zip(unique_labels,
                                                   unique_idxs)])
    return np.array([labels_to_index[l] for l in chord_labels])


def create_chord_stream(stash, win_length):
    """Return an unconstrained stream of chord samples."""
    entity_pool = [pescador.Streamer(chord_sampler, key, stash, win_length)
                   for key in stash.keys()]
    return pescador.mux(entity_pool, None, 50)


def create_uniform_quality_stream(stash, win_length, pool_size=50):
    """Return a stream of chord samples, with uniform quality presentation."""
    partition_labels = util.partition(stash, quality_map)
    quality_pool = []
    for qual_idx in range(14):
        quality_subindex = util.index_partition_arrays(
            partition_labels, [qual_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, quality_subindex)
                       for key in quality_subindex.keys()]
        stream = pescador.mux(entity_pool, None, 25)
        quality_pool.append(pescador.Streamer(stream))

    return pescador.mux(quality_pool, n_samples=None, k=pool_size,
                        lam=None, with_replacement=False)
