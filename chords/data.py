import itertools
import numpy as np

import biggie
import pescador
import mir_eval
from dl4mir.chords import labels as L
import dl4mir.chords.pipefxs as FX
from dl4mir.common import util


def intervals_to_durations(intervals):
    return np.abs(np.diff(np.asarray(intervals), axis=1)).flatten()


def slice_cqt_entity(entity, length, idx=None):
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
    sample: biggie.Entity with fields {data, chord_label}
        The windowed chord observation.
    """
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    cqt = np.array([util.slice_tile(x, idx, length) for x in entity.cqt])
    return biggie.Entity(data=cqt, chord_label=entity.chord_labels[idx])


def slice_chroma_entity(entity, length, idx=None):
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
    sample: biggie.Entity with fields {data, chord_label}
        The windowed chord observation.
    """
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    chroma = util.slice_tile(entity.chroma, idx, length)
    # chroma = np.array([util.slice_tile(x, idx, length) for x in entity.chroma])
    return biggie.Entity(data=chroma, chord_label=entity.chord_labels[idx])


def chord_sampler(key, stash, win_length=20, index=None, max_samples=None,
                  sample_func=slice_cqt_entity):
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
        yield sample_func(entity, win_length, valid_samples[idx])
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
        yield slice_cqt_entity(entity, win_length, valid_samples[idx])
        idx += 1
        count += 1


def lazy_cqt_buffer(key, stash, win_length=20, index=None):
    """Generator for stepping windowed chord observations from an entity; note
    that the entity is not queried until the generator is called.

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


def map_chord_labels(entity, lexicon):
    if hasattr(entity, 'chord_label'):
        labels = entity.chord_label
    else:
        labels = entity.chord_labels
    return lexicon.label_to_index(labels)


def map_bigrams(entity, lexicon):
    return lexicon.label_to_index(entity.bigrams)


def create_chord_index_stream(stash, win_length, lexicon,
                              index_mapper=map_chord_labels,
                              sample_func=slice_cqt_entity,
                              pitch_shift_func=FX.pitch_shift_cqt,
                              max_pitch_shift=0, working_size=50,
                              partition_labels=None, valid_idx=None):
    """Return an unconstrained stream of chord samples with class indexes.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    lexicon : lexicon.Lexicon
        Instantiated chord lexicon for mapping labels to indices.
    working_size : int
        Number of open streams at a time.
    pitch_shift : int
        Maximum number of semitones (+/-) to rotate an observation.
    partition_labels : dict


    Returns
    -------
    stream : generator
        Data stream of windowed chord entities.
    """
    if partition_labels is None:
        partition_labels = util.partition(stash, index_mapper, lexicon)

    if valid_idx is None:
        valid_idx = range(lexicon.num_classes)

    chord_index = util.index_partition_arrays(partition_labels, valid_idx)
    entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                     win_length, chord_index,
                                     sample_func=sample_func)
                   for key in stash.keys()]

    stream = pescador.mux(entity_pool, None, working_size, lam=25)
    if max_pitch_shift > 0:
        stream = pitch_shift_func(stream, max_pitch_shift=max_pitch_shift)

    return FX.map_to_class_index(stream, index_mapper, lexicon)


def create_chroma_stream(stash, win_length, working_size=50, pitch_shift=0,
                         bins_per_pitch=1):
    """Return an unconstrained stream of chord samples with class indexes.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    lexicon : lexicon.Lexicon
        Instantiated chord lexicon for mapping labels to indices.
    working_size : int
        Number of open streams at a time.
    pitch_shift : int
        Maximum number of semitones (+/-) to rotate an observation.
    partition_labels : dict


    Returns
    -------
    stream : generator
        Data stream of windowed chord entities.
    """
    entity_pool = [pescador.Streamer(chord_sampler, key, stash, win_length)
                   for key in stash.keys()]

    stream = pescador.mux(entity_pool, None, working_size, lam=25)
    if pitch_shift > 0:
        stream = FX.pitch_shift_cqt(stream, max_pitch_shift=pitch_shift)

    return FX.map_to_chroma(stream, bins_per_pitch)


def create_uniform_chord_index_stream(stash, win_length, lexicon,
                                      index_mapper=map_chord_labels,
                                      sample_func=slice_cqt_entity,
                                      pitch_shift_func=FX.pitch_shift_cqt,
                                      max_pitch_shift=0, working_size=4,
                                      partition_labels=None, valid_idx=None):
    """Return a stream of chord samples, with uniform quality presentation.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    lexicon : lexicon.Lexicon
        Instantiated chord lexicon for mapping labels to indices.
    working_size : int
        Number of open streams at a time.
    pitch_shift : int
        Maximum number of semitones (+/-) to rotate an observation.
    partition_labels : dict


    Returns
    -------
    stream : generator
        Data stream of windowed chord entities.
    """
    if partition_labels is None:
        partition_labels = util.partition(stash, index_mapper, lexicon)

    if valid_idx is None:
        valid_idx = range(lexicon.num_classes)

    chord_pool = []
    for chord_idx in valid_idx:
        subindex = util.index_partition_arrays(partition_labels, [chord_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, subindex,
                                         sample_func=sample_func)
                       for key in subindex.keys()]
        if len(entity_pool) == 0:
            continue
        stream = pescador.mux(
            entity_pool, n_samples=None, k=working_size, lam=20)
        chord_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(chord_pool, n_samples=None, k=lexicon.vocab_dim,
                          lam=None, with_replacement=False)
    if max_pitch_shift > 0:
        stream = pitch_shift_func(stream, max_pitch_shift=max_pitch_shift)

    return FX.map_to_class_index(stream, index_mapper, lexicon)


def create_uniform_chroma_stream(stash, win_length, lexicon, working_size=5,
                                 bins_per_pitch=1, max_pitch_shift=0,
                                 partition_labels=None, valid_idx=None):
    """Return an unconstrained stream of chord samples with class indexes.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    lexicon : lexicon.Lexicon
        Instantiated chord lexicon for mapping labels to indices.
    working_size : int
        Number of open streams at a time.
    pitch_shift : int
        Maximum number of semitones (+/-) to rotate an observation.
    partition_labels : dict


    Returns
    -------
    stream : generator
        Data stream of windowed chord entities.
    """
    if partition_labels is None:
        partition_labels = util.partition(stash, map_chord_labels, lexicon)

    if valid_idx is None:
        valid_idx = range(lexicon.num_classes)

    chord_pool = []
    for chord_idx in valid_idx:
        subindex = util.index_partition_arrays(partition_labels, [chord_idx])
        entity_pool = [pescador.Streamer(chord_sampler, key, stash,
                                         win_length, subindex,
                                         sample_func=slice_cqt_entity)
                       for key in subindex.keys()]
        if len(entity_pool) == 0:
            continue
        stream = pescador.mux(
            entity_pool, n_samples=None, k=working_size, lam=20)
        chord_pool.append(pescador.Streamer(stream))

    stream = pescador.mux(chord_pool, n_samples=None, k=lexicon.vocab_dim,
                          lam=None, with_replacement=False)

    if max_pitch_shift > 0:
        stream = FX.pitch_shift_cqt(stream, max_pitch_shift=max_pitch_shift)

    return FX.map_to_chroma(stream, bins_per_pitch)


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
        stream = FX.pitch_shift_cqt(stream, max_pitch_shift=pitch_shift)

    return FX.map_to_chord_index(stream, vocab_dim)


def create_uniform_factored_stream(stash, win_length, partition_labels=None,
                                   working_size=50, vocab_dim=157,
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

    stream = pescador.mux(quality_pool, n_samples=None, k=working_size,
                          lam=None, with_replacement=False)
    if pitch_shift:
        stream = FX.pitch_shift_cqt(stream)

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
        chord_idx = L.chord_label_to_class_index(chord_labels, vocab_dim)
        for n in range(len(chord_idx) - 1):
            if chord_idx[n] is None or chord_idx[n + 1] is None:
                continue
            c_idx = int(chord_idx[n]) / 12
            rel_idx = L.relative_chord_index(
                chord_idx[n], chord_idx[n+1], vocab_dim)
            transitions[c_idx, rel_idx] += 1

    trans_mat = []
    for row in transitions[:-1]:
        for _ in range(12):
            trans_mat.append(L.rotate(row, 0-_))

    trans_mat.append(trans_mat[-1])
    return np.array(trans_mat)


def count_labels(reference_set, vocab_dim=157):
    labels = dict()
    for labeled_intervals in reference_set.values():
        rootless = [L.join(*list([''] + list(L.split(l)[1:])))
                    for l in labeled_intervals['labels']]
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for y, w in zip(rootless, durations):
            if not y in labels:
                labels[y] = 0
            labels[y] += w

    qlabels = labels.keys()
    counts = [labels[y] for y in qlabels]
    idx = np.argsort(counts)[::-1]
    return [qlabels[i] for i in idx], [counts[i] for i in idx]


def count_states(reference_set, lexicon):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = lexicon.label_to_index(labeled_intervals['labels'])
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for y, w in zip(chord_idx, durations):
            s = L.relative_chord_index(y, y, 157)
            if not s in states:
                states[s] = 0
            states[s] += w

    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def count_bigrams(reference_set, vocab_dim=157):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = L.chord_label_to_class_index(labeled_intervals['labels'],
                                                 vocab_dim)
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for n in range(1, len(chord_idx)):
            s = tuple([L.relative_chord_index(chord_idx[n],
                                              chord_idx[n + i], 157)
                       for i in range(-1, 1)])
            if not s in states:
                states[s] = 0
            states[s] += durations[n]
    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def count_trigrams(reference_set, vocab_dim=157):
    states = dict()
    for labeled_intervals in reference_set.values():
        chord_idx = L.chord_label_to_class_index_soft(
            labeled_intervals['labels'], vocab_dim)
        intervals = np.array(labeled_intervals['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for n in range(1, len(chord_idx) - 1):
            s = tuple([L.relative_chord_index(chord_idx[n],
                                              chord_idx[n + i], 157)
                       for i in range(-1, 2)])
            if not s in states:
                states[s] = 0
            states[s] += durations[n]
    labels = states.keys()
    counts = [states[y] for y in labels]
    idx = np.argsort(counts)[::-1]
    return [labels[i] for i in idx], [counts[i] for i in idx]


def chroma_trigrams(ref_set):
    states = dict()
    for v in ref_set.values():
        labels = v['labels']
        y = L.chord_label_to_class_index(labels, 157)
        intervals = np.array(v['intervals'])
        durations = np.abs(np.diff(intervals, axis=1)).flatten()
        for n in range(1, len(y) - 1):
            sidx = [L.relative_chord_index(y[n], y[n + i], 157)
                    for i in range(-1, 2)]
            if None in sidx:
                continue
            rot_labels = [L.index_to_chord_label(s, 157) for s in sidx]
            c = tuple(["".join(["%d" % _
                                for _ in L.chord_label_to_chroma(l).flatten()])
                       for l in rot_labels])
            if not c in states:
                states[c] = dict(labels=set(), duration=0.0)
            states[c]['duration'] += durations[n]
            states[c]['labels'].add(labels[n])
    return states


def ideal_chroma_fr(ref_set, stash, framerate=20.0):
    sample_size = 1./framerate
    for k, v in ref_set.iteritems():
        chroma = L.chord_label_to_chroma(v['labels'])
        time_points, chroma = mir_eval.util.intervals_to_samples(
            np.asarray(v['intervals']), chroma.tolist(),
            sample_size=sample_size, fill_value=[0]*12)
        time_points, labels = mir_eval.util.intervals_to_samples(
            np.asarray(v['intervals']), v['labels'],
            sample_size=sample_size, fill_value='N')
        stash.add(str(k), biggie.Entity(
            chroma=chroma, chord_labels=[str(l) for l in labels],
            time_points=time_points), overwrite=True)
        print k


def ideal_chroma_ss(ref_set, stash):
    for k, v in ref_set.iteritems():
        intervals, labels = L.compress_labeled_intervals(**v)
        chord_labels = [str(l) for l in labels]
        chroma = L.chord_label_to_chroma(chord_labels)
        durations = intervals_to_durations(intervals)
        stash.add(str(k), biggie.Entity(
            chroma=chroma, chord_labels=chord_labels,
            time_points=intervals[:, 0], durations=durations), overwrite=True)
        print k
