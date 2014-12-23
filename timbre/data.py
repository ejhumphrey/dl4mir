import itertools
import numpy as np

import biggie
import pescador
from dl4mir.common import util


def slice_cqt_entity(entity, length, idx=None):
    """Return a windowed slice of a cqt Entity.

    Parameters
    ----------
    entity : Entity, with at least {cqt, icode} fields
        Observation to window.
        Note that entity.cqt is shaped (num_channels, num_frames, num_bins).
    length : int
        Length of the sliced array.
    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Returns
    -------
    sample: biggie.Entity with fields {cqt, label}
        The windowed observation.
    """
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    cqt = np.array([util.slice_tile(x, idx, length) for x in entity.cqt])
    return biggie.Entity(cqt=cqt, label=entity.icode)


def cqt_sampler(key, stash, win_length=20, max_samples=None,
                threshold=0.05, sample_func=slice_cqt_entity):
    """Generator for sampling windowed observations from an entity.

    Parameters
    ----------
    key : str
        Key for the entity of interest; must be consistent across both `stash`
        and `index`, when the latter is provided.
    stash : dict_like
        Dict or biggie.Stash of entities.
    win_length: int
        Length of centered observation window for the CQT.
    threshold: scalar, default=None
        If given, only select from indices with a maximum frequency magnitude
        over the threshold (eliminate silence).
    max_samples: int, or None
        Maximum number of samples to return from this Generator; if None, runs
        indefinitely.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, label}
        The windowed observation.
    """
    entity = stash.get(key)
    num_samples = entity.cqt.shape[1]
    valid_samples = np.arange(num_samples)
    if not threshold is None:
        valid_idx = entity.cqt.mean(axis=0).max(axis=-1) > threshold
        valid_samples = valid_samples[valid_idx]

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
    """Generator for stepping windowed observations from an entity.

    Parameters
    ----------
    entity : biggie.Entity
        CQT entity to step through
    win_length: int
        Length of centered observation window for the CQT.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed observation.
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


def create_labeled_stream(stash, win_length, working_size=5000, threshold=None,
                          sample_func=slice_cqt_entity):
    """Return an unconstrained stream of samples with class labels.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    working_size : int
        Number of open streams at a time.

    Returns
    -------
    stream : generator
        Data stream of windowed chord entities.
    """
    args = dict(sample_func=sample_func)
    if not threshold is None:
        args.update(threshold=threshold)

    entity_pool = [pescador.Streamer(cqt_sampler, key, stash,
                                     win_length, **args)
                   for key in stash.keys()]

    return pescador.mux(entity_pool, None, working_size, lam=25)


def create_pairwise_stream(stash, win_length, threshold=None,
                           working_size=100, sample_func=slice_cqt_entity):
    """Return a stream of samples, with equal positive and negative
    examples."""
    args = dict(sample_func=sample_func)
    if not threshold is None:
        args.update(threshold=threshold)

    keys = stash.keys()
    partitions = dict()
    # Group keys by instrument code
    for k in keys:
        icode = k.split("_")[0]
        if not icode in partitions:
            partitions[icode] = list()
        partitions[icode].append(k)

    inst_streams = []
    for icode, key_set in partitions.items():
        entity_pool = [pescador.Streamer(cqt_sampler, key, stash, win_length,
                                         **args)
                       for key in key_set]
        stream = pescador.mux(entity_pool, n_samples=None,
                              k=working_size, lam=1)
        inst_streams.append(stream)

    inst_streams = np.array(inst_streams)
    triple_pool = []
    nrange = np.arange(len(inst_streams))
    for idx, stream in enumerate(inst_streams):
        neg_pool = [pescador.Streamer(x)
                    for x in inst_streams[np.not_equal(nrange, idx)]]
        neg_stream = pescador.mux(neg_pool, n_samples=None,
                                  k=len(neg_pool), lam=None,
                                  with_replacement=False)
        triples = itertools.izip(
            inst_streams[idx], inst_streams[idx], neg_stream)
        triple_pool.append(pescador.Streamer(triples))

    cstream = pescador.mux(triple_pool, n_samples=None, k=len(triple_pool),
                           lam=None, with_replacement=False)

    return unpack_triples(cstream)


def unpack_triples(stream):
    for triple in stream:
        if triple is None:
            yield triple
            continue
        x1, x2, z = triple
        yield biggie.Entity(cqt=x1.cqt,
                            cqt_2=x2.cqt,
                            score=float(x1.label == x2.label))
        yield biggie.Entity(cqt=x1.cqt,
                            cqt_2=z.cqt,
                            score=float(x1.label == z.label))


def pairwise_filter(stream, filt_func, filt_key='pw_cost', **kwargs):
    for entity in stream:
        if entity is None:
            yield entity
            continue
        res = filt_func(
            cqt=entity.cqt[np.newaxis, ...],
            cqt_2=entity.cqt_2[np.newaxis, ...],
            score=np.array([entity.score]),
            **kwargs)
        yield entity if res[filt_key][0] > 0 else None


def batch_filter(stream, filt_func, threshold=2.0**-16.0,
                 filt_key='pw_cost', **kwargs):
    for data in stream:
        fargs = data.copy()
        fargs.update(**kwargs)
        mask = filt_func(**fargs)[filt_key] > threshold
        for k in data:
            data[k] = data[k][mask]

        yield data
