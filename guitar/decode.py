import numpy as np
import sys
import pyjams
from dl4mir.common import util


if hasattr(sys, 'ps1'):
    # Interactive mode
    Parallel = None
else:
    # Command line executable -- enable multiprocessing
    from sklearn.externals.joblib import Parallel, delayed

__interactive__ = Parallel is None
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


def decode_fretboard(entity, penalty, label_map, **viterbi_args):
    """Decode a fretboard Entity to a RangeAnnotation.

    Parameters
    ----------
    entity : biggie.Entity
        Entity to decode; expects {posterior, time_points}.
    penalty : scalar
        Self-transition penalty to use for Viterbi decoding.
    label_map : function
        Map from frets to string labels.
    **viterbi_args : dict
        Other arguments to pass to the Viterbi algorithm.

    Returns
    -------
    annot : pyjams.RangeAnnotation
        Populated chord annotation.
    """
    num_frets = entity.fretboard.shape[-1]
    frets_pred = np.array([util.viterbi(x, penalty=penalty, **viterbi_args)
                           for x in entity.fretboard.transpose(1, 0, 2)]).T
    frets_pred[np.equal(frets_pred, num_frets - 1)] = -1
    labels = [label_map(frets.tolist()) for frets in frets_pred]

    intervals, labels = util.compress_samples_to_intervals(
        labels, entity.time_points)

    annot = pyjams.RangeAnnotation()
    populate_annotation(intervals, labels, list(), annot=annot)
    annot.sandbox.penalty = penalty
    return annot


def decode_fretboard_parallel(entity, penalties, label_map, num_cpus=NUM_CPUS,
                              **viterbi_args):
    """Apply Viterbi decoding in parallel.

    Parameters
    ----------
    entity : biggie.Entity
        Entity to estimate, requires at least {fretboard}.
    penalties : list
        Set of self-transition penalties to apply.
    label_map : callable object
        Map from frets to string labels.

    Returns
    -------
    annotation : pyjams.JAMS.RangeAnnotation
        Populated JAMS annotation.
    """
    assert not __interactive__
    pool = Parallel(n_jobs=num_cpus)
    decode = delayed(decode_fretboard)
    return pool(decode(entity, p, label_map) for p in penalties)


def decode_stash_parallel(stash, penalty, label_map, num_cpus=NUM_CPUS,
                          **viterbi_args):
    """Apply Viterbi decoding over a stash in parallel.

    Parameters
    ----------
    stash : biggie.Stash
        Stash of fretboard posteriors.
    penalty : scalar
        Self-transition penalty.
    label_map : callable object
        Map from frets to string labels.
    num_cpus : int
        Number of CPUs to use in parallel.
    **viterbi_args, other args to pass to util.viterbi

    Returns
    -------
    annotset : dict of pyjams.RangeAnnotations
        Range annotations under the same keys as the input stash.
    """
    assert not __interactive__
    keys = stash.keys()
    pool = Parallel(n_jobs=num_cpus)
    decode = delayed(decode_fretboard)
    results = pool(decode(stash.get(k), penalty, label_map) for k in keys)
    return {k: r for k, r in zip(keys, results)}
