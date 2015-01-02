import numpy as np
import sys
import time

import biggie
import pyjams

from dl4mir.common import util


if hasattr(sys, 'ps1'):
    # Interactive mode
    Pool = None
else:
    # Command line executable -- enable multiprocessing
    from multiprocessing import Pool

__interactive__ = Pool is None
NUM_CPUS = 1 if __interactive__ else None


def populate_annotation(intervals, labels, confidence, annot):
    """Fill in annotation data, in-place.

    Parameters
    ----------
    intervals : np.ndarray, shape=(N, 2)
        Start and end times, in seconds.
    labels : list, len=N
        String labels corresponding to the returned intervals.
    confidence : list, len=N
        Confidence values (ave. log-likelihoods) of the labels.
    annot : pyjams.RangeAnnotation
        Annotation to populate, in-place.

    Returns
    -------
    None
    """
    pyjams.util.fill_range_annotation_data(
        intervals[:, 0], intervals[:, 1], labels, annot)

    for obs, conf in zip(annot.data, confidence):
        obs.label.confidence = conf


def decode_posterior(entity, penalty, vocab, **viterbi_args):
    """Decode a posterior Entity to a RangeAnnotation.

    Parameters
    ----------
    entity : biggie.Entity
        Entity to decode; expects {posterior, time_points}.
    penalty : scalar
        Self-transition penalty to use for Viterbi decoding.
    vocab : lexicon.Vocabulary
        Vocabulary object; expects an `index_to_label` method.
    **viterbi_args : dict
        Other arguments to pass to the Viterbi algorithm.

    Returns
    -------
    annot : pyjams.RangeAnnotation
        Populated chord annotation.
    """
    y_idx = util.viterbi(entity.posterior, penalty=penalty, **viterbi_args)
    labels = vocab.index_to_label(y_idx)

    n_range = np.arange(len(y_idx))
    idx_intervals = util.compress_samples_to_intervals(y_idx, n_range)[0]
    boundaries = np.sort(np.unique(idx_intervals.flatten()))
    likelihoods = np.log(entity.posterior[n_range, y_idx])
    confidence = util.boundary_pool(likelihoods, boundaries, pool_func='mean')
    confidence[np.invert(np.isfinite(confidence))] = 0.0

    intervals, labels = util.compress_samples_to_intervals(
        labels, entity.time_points)

    annot = pyjams.RangeAnnotation()
    populate_annotation(intervals, labels, confidence.tolist(), annot=annot)
    annot.sandbox.penalty = penalty
    return annot


def decode_posterior_parallel(entity, penalties, vocab, num_cpus=NUM_CPUS,
                              **viterbi_args):
    """Apply Viterbi decoding in parallel.

    Parameters
    ----------
    entity : biggie.Entity
        Entity to estimate, requires at least {posterior}.
    penalties : list
        Set of self-transition penalties to apply.
    vocab : chord.lexicon.Vocabulary
        Map from posterior indices to labels.

    Returns
    -------
    annotation : pyjams.JAMS.RangeAnnotation
        Populated JAMS annotation.
    """
    assert not __interactive__
    pool = Pool(processes=num_cpus)
    threads = [pool.apply_async(decode_posterior,
                                (entity, p, vocab),)
               for p in penalties]
    pool.close()
    pool.join()
    return [t.get() for t in threads]


def decode_stash_parallel(stash, penalty, vocab, num_cpus=NUM_CPUS,
                          **viterbi_args):
    assert not __interactive__
    keys = stash.keys()
    pool = Pool(processes=num_cpus)
    threads = [pool.apply_async(decode_posterior,
                                (stash.get(k), penalty, vocab),)
               for k in keys]
    pool.close()
    pool.join()
    results = [t.get() for t in threads]
    return {k: r for k, r in zip(keys, results)}
