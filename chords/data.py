import itertools
import numpy as np

import biggie
import pescador
from dl4mir.chords import labels
import dl4mir.chords.pipefxs as FX
from dl4mir.common import util


def extract_tile(x_in, idx, length):
    """Extract a padded tile from a matrix, along the first dimension.

    Parameters
    ----------
    x_in : np.ndarray, ndim=2
        2D Matrix to slice.
    idx : int
        Centered index for the resulting tile.
    length : int
        Total length for the output tile.

    Returns
    -------
    z_out : np.ndarray, ndim=2
        The extracted tile.
    """
    start_idx = idx - length / 2
    end_idx = start_idx + length
    tile = np.zeros([length, x_in.shape[1]])

    if start_idx < 0:
        tile[np.abs(start_idx):, :] = x_in[:end_idx, :]
    elif end_idx > x_in.shape[0]:
        end_idx = x_in.shape[0] - start_idx
        tile[:end_idx, :] = x_in[start_idx:, :]
    else:
        tile[:, :] = x_in[start_idx:end_idx, :]
    return tile


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
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    cqt = np.array([extract_tile(x, idx, length) for x in entity.cqt])
    return biggie.Entity(cqt=cqt, chord_label=entity.chord_labels[idx])


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
    num_samples = len(entity.chord_labels)
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


def cqt_buffer(entity, win_length=20, valid_samples=None):
    """Generator for stepping windowed chord observations from an entity.

    Parameters
    ----------
    entity : biggie.Entity
        CQT entity to step through
    win_length: int
        Length of centered observation window for the CQT.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    num_samples = len(entity.chord_labels)
    if valid_samples is None:
        valid_samples = np.arange(num_samples)

    idx = 0
    count = 0
    while count < len(valid_samples):
        yield slice_chord_entity(entity, win_length, valid_samples[idx])
        idx += 1
        count += 1


def lazy_cqt_buffer(key, stash, win_length=20, index=None):
    """Generator for stepping windowed chord observations from an entity; note
    that the entity is not queried until the generator is invoked.

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
    num_samples = len(entity.chord_labels)
    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    for x in cqt_buffer(entity, win_length, valid_samples):
        yield x


def chord_map(entity, vocab_dim=157):
    """Map an entity's chord_labels to class indexes; for partitioning."""
    chord_labels = entity.chord_labels
    unique_labels = np.unique(chord_labels)
    unique_idxs = labels.chord_label_to_class_index(unique_labels)
    labels_to_index = dict([(l, i) for l, i in zip(unique_labels,
                                                   unique_idxs)])
    return np.array([labels_to_index[l] for l in chord_labels], dtype=object)


def quality_map(entity, vocab_dim=157):
    """Map an entity's chord_labels to quality indexes; for partitioning."""
    chord_labels = entity.chord_labels
    unique_labels = np.unique(chord_labels)
    unique_idxs = labels.chord_label_to_quality_index(unique_labels)
    labels_to_index = dict([(l, i) for l, i in zip(unique_labels,
                                                   unique_idxs)])
    return np.array([labels_to_index[l] for l in chord_labels], dtype=object)


def create_chord_stream(stash, win_length, pool_size=50, vocab_dim=157,
                        pitch_shift=0):
    """Return an unconstrained stream of chord samples."""
    partition_labels = util.partition(stash, chord_map)
    chord_index = util.index_partition_arrays(
        partition_labels, range(vocab_dim))

    entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                     win_length, chord_index)
                   for key in stash.keys()]

    stream = pescador.mux(entity_pool, None, pool_size, lam=25)
    if pitch_shift > 0:
        stream = FX.pitch_shift(stream, max_pitch_shift=pitch_shift)

    return FX.map_to_chord_index(stream, vocab_dim)


def create_stash_stream(stash, win_length, pool_size=50, vocab_dim=157,
                        pitch_shift=0):
    """Stream the contents of a stash."""
    partition_labels = util.partition(stash, quality_map)
    quality_index = util.index_partition_arrays(partition_labels, range(14))

    entity_pool = [pescador.Streamer(lazy_cqt_buffer, key,
                                     stash, win_length, quality_index)
                   for key in stash.keys()]
    stream = pescador.mux(entity_pool, None, pool_size)
    if pitch_shift:
        stream = FX.pitch_shift(stream)

    return FX.map_to_chord_index(stream, vocab_dim)


def create_uniform_quality_stream(stash, win_length, partition_labels=None,
                                  pool_size=50, vocab_dim=157,
                                  pitch_shift=0, valid_idx=None):
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


def create_uniform_chord_stream(stash, win_length, partition_labels=None,
                                vocab_dim=157, pitch_shift=0,
                                working_size=4, valid_idx=None):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, chord_map)

    if valid_idx is None:
        valid_idx = range(vocab_dim)

    chord_pool = []
    for chord_idx in valid_idx:
        subindex = util.index_partition_arrays(partition_labels, [chord_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, subindex)
                       for key in subindex.keys()]
        if len(entity_pool) == 0:
            continue
        stream = pescador.mux(
            entity_pool, n_samples=None, k=working_size, lam=20)
        chord_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(chord_pool, n_samples=None, k=vocab_dim, lam=None,
                          with_replacement=False)
    if pitch_shift:
        stream = FX.pitch_shift(stream, max_pitch_shift=pitch_shift)

    return FX.map_to_chord_index(stream, vocab_dim)


def muxed_uniform_chord_stream(stash, synth_stash, win_length, vocab_dim=157,
                               pitch_shift=0, working_size=4):
    """Return a stream of chord samples, merging two separate datasets."""
    partition_labels = util.partition(stash, chord_map)
    synth_partition_labels = util.partition(synth_stash, chord_map)

    valid_idx = range(vocab_dim)
    valid_idx_synth = range(60, vocab_dim - 1)

    chord_pool = []
    for chord_idx in valid_idx:
        subindex = util.index_partition_arrays(partition_labels, [chord_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, subindex)
                       for key in subindex.keys()]
        if chord_idx in valid_idx_synth:
            subindex = util.index_partition_arrays(
                synth_partition_labels, [chord_idx])
            synth_pool = [pescador.Streamer(chord_sampler, key, synth_stash,
                                            win_length, subindex)
                          for key in subindex.keys()]
            entity_pool.extend(synth_pool)
        if len(entity_pool) == 0:
            continue
        stream = pescador.mux(
            entity_pool, n_samples=None, k=working_size, lam=20)
        chord_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(chord_pool, n_samples=None, k=vocab_dim, lam=None,
                          with_replacement=False)
    if pitch_shift:
        stream = FX.pitch_shift(stream, max_pitch_shift=pitch_shift)

    return FX.map_to_chord_index(stream, vocab_dim)


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


def create_contrastive_chord_stream(stash, win_length, valid_idx=None,
                                    partition_labels=None, working_size=2,
                                    vocab_dim=157, pitch_shift=0,
                                    neg_probs=None):
    """Return a stream of chord samples, with equal positive and negative
    examples."""
    if partition_labels is None:
        partition_labels = util.partition(stash, chord_map)

    if valid_idx is None:
        valid_idx = range(vocab_dim)

    if neg_probs is None:
        neg_probs = np.ones([vocab_dim]*2)
        neg_probs[np.eye(vocab_dim, dtype=bool)] = 0.0
        neg_probs = util.normalize(neg_probs, axis=1)

    chord_streams = []
    has_data = np.ones(vocab_dim, dtype=bool)
    for chord_idx in valid_idx:
        subindex = util.index_partition_arrays(partition_labels, [chord_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, subindex)
                       for key in subindex.keys()]
        if len(entity_pool) == 0:
            has_data[chord_idx] = False
            stream = None
        else:
            stream = pescador.mux(
                entity_pool, n_samples=None, k=working_size, lam=20)
        chord_streams.append(stream)

    chord_streams = np.array(chord_streams)
    binary_pool = []
    for chord_idx in range(vocab_dim):
        if chord_streams[chord_idx] is None:
            continue

        # Skip contrast streams with (a) no data or (b) no probability.
        not_chord_probs = neg_probs[chord_idx]
        not_chord_probs[chord_idx] = 0.0
        not_chord_probs *= has_data
        nidx = not_chord_probs > 0.0
        assert not_chord_probs.sum() > 0.0
        chord_pool = [pescador.Streamer(x)
                      for x in chord_streams[nidx]]
        neg_stream = pescador.mux(chord_pool, n_samples=None,
                                  k=len(chord_pool), lam=None,
                                  with_replacement=False,
                                  pool_weights=not_chord_probs[nidx])
        pair_stream = itertools.izip(chord_streams[chord_idx], neg_stream)
        binary_pool.append(pescador.Streamer(pair_stream))

    cstream = pescador.mux(binary_pool, n_samples=None, k=len(binary_pool),
                           lam=None, with_replacement=False)
    return FX.unpack_contrastive_pairs(cstream, vocab_dim)


def chroma_stepper(key, stash, index=None):
    """writeme."""
    entity = stash.get(key)
    num_samples = len(entity.chord_labels)
    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    idx = 0
    count = 0
    while count < len(valid_samples):
        n = valid_samples[idx]
        if n >= entity.chroma.shape[0]:
            print "Out of range! %s" % key
            break
        yield biggie.Entity(chroma=entity.chroma[n],
                            chord_label=entity.chord_labels[n])
        idx += 1
        count += 1


def count_transitions(stash, vocab_dim=157):
    """writeme."""
    transitions = np.zeros([(vocab_dim / 12) + 1, vocab_dim])
    for k in stash.keys():
        chord_labels = stash.get(k).chord_labels
        chord_idx = labels.chord_label_to_class_index(chord_labels, vocab_dim)
        for n in range(len(chord_idx) - 1):
            if chord_idx[n] is None or chord_idx[n + 1] is None:
                continue
            c_idx = int(chord_idx[n]) / 12
            rel_idx = labels.relative_chord_index(
                chord_idx[n], chord_idx[n+1], vocab_dim)
            transitions[c_idx, rel_idx] += 1

    trans_mat = []
    for row in transitions[:-1]:
        for _ in range(12):
            trans_mat.append(labels.rotate(row, 0-_))

    trans_mat.append(trans_mat[-1])
    return np.array(trans_mat)
