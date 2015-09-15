"""Methods and routines to manipulate timbre data."""

import biggie
import itertools
import numpy as np
import pescador
import time

from dl4mir.common import util
from dl4mir.common import fileutil as futil


def create_entity(npz_file, dtype=np.float32):
    """Create an entity from the given file.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.
    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated entity, with the following fields:
            {cqt, time_points, icode, note_number, fcode}.
    """
    (icode, note_number,
        fcode) = [np.array(_) for _ in futil.filebase(npz_file).split('_')]
    entity = biggie.Entity(icode=icode, note_number=note_number,
                           fcode=fcode, **np.load(npz_file))
    entity.cqt = entity.cqt.astype(dtype)
    return entity


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


def slice_embedding_entity(entity, length, idx=None):
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
    idx = np.random.randint(entity.embedding.shape[0]) if idx is None else idx
    return biggie.Entity(
        embedding=entity.embedding[idx],
        time=entity.time_points[idx],
        fcode=entity.fcode,
        note_number=entity.note_number,
        icode=entity.icode)


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
    num_samples = len(entity.time_points)
    valid_samples = np.arange(num_samples)
    if threshold is not None:
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
    threshold : scalar, default=None
        Threshold under which to suppress entities.
    sample_func : callable
        Sampling function to apply to each entity.

    Returns
    -------
    stream : generator
        Data stream of windowed entities.
    """
    args = dict(sample_func=sample_func, threshold=threshold)
    entity_pool = [pescador.Streamer(cqt_sampler, key, stash,
                                     win_length, **args)
                   for key in stash.keys()]

    return pescador.mux(entity_pool, None, working_size, lam=25)


def create_pairwise_stream(stash, win_length, working_size=100, threshold=None,
                           sample_func=slice_cqt_entity):
    """Return a stream of samples, with equal positive and negative
    examples.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    working_size : int
        Number of open streams at a time.
    threshold : scalar, default=None
        Threshold under which to suppress entities.
    sample_func : callable
        Sampling function to apply to each entity.

    Returns
    -------
    stream : generator
        Data stream of windowed entities.
    """
    args = dict(sample_func=sample_func)
    if threshold is not None:
        args.update(threshold=threshold)

    keys = stash.keys()
    partitions = dict()
    # Group keys by instrument code
    for k in keys:
        icode = k.split("_")[0]
        if icode not in partitions:
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

    return _unpack_triples(cstream)


def _unpack_triples(stream):
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


def batch_filter(stream, filt_func, threshold=2.0**-16.0, min_batch=1,
                 max_consecutive_skips=5, filt_key='pw_cost', **kwargs):
    """Apply a filter inline to discard datapoints in a batch.

    Parameters
    ----------
    stream : generator
        Stream to filter; must yield dictionart / **kwargs-able objects.
    filt_func : function
        Function to apply to each object returned by the stream.
    threshold : scalar
        Value to threshold the output of the filter function.
    min_batch : int, default=1
        Minimum batch size to return at each iteration.
    max_consecutive_skips : int, default=5
        Maximum number of retries to make before raising the StopIteration.
    filt_key : hashable
        Dictionary key for the filter function's result.
    **kwargs : misc
        Other keyword arguments to pass onto the filter function.

    Yields
    ------
    same as `stream`
    """
    assert min_batch >= 1, "`min_batch` must be at least 1."

    for data in stream:
        fargs = data.copy()
        fargs.update(**kwargs)
        res = filt_func(**fargs)
        mask = res[filt_key] > threshold

        if mask.sum() == 0:
            print "No valid data? Expect a failure"
            fargs.update(**res)
            np.savez("error_dump_{0}.npz".format(time.time()), **fargs)

        for k in data:
            data[k] = data[k][mask]

        yield data


def sample_embedding_stash(stash, num_points):
    """Sample a collection of embedding points from a stash.

    Parameters
    ----------
    stash : biggie.Stash
        Collection from which to draw samples.
    num_points : int
        Number of datapoints to sample.

    Returns
    -------
    data : np.ndarray, shape=(num_points, 3)
        Observations.
    labels : list, len=num_points
        Instrument labels.
    keys : list
        Source keys.
    time_points : np.ndarray
        Points in time of the observations.
    """
    data = np.zeros([num_points, 3])
    labels = list()
    time_points = np.zeros(num_points)
    keys = list()
    stream = create_labeled_stream(
        stash, 1, working_size=100, threshold=None,
        sample_func=slice_embedding_entity)

    for n in range(num_points):
        a = stream.next()
        data[n, ...] = a.embedding
        labels.append(str(a.icode))
        time_points[n] = a.time
        keys.append(
            "_".join([str(_) for _ in a.icode, a.note_number, a.fcode]))

    return data, labels, keys, time_points
