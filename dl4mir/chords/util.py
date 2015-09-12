import numpy as np
from multiprocessing import Pool

import biggie
import pyjams

from dl4mir.common.util import run_length_encode
from dl4mir.common.util import viterbi
from dl4mir.common.util import boundary_pool

from dl4mir.common.transform_stash import convolve


def compress_samples_to_intervals(labels, time_points):
    """Compress a set of time-aligned labels via run-length encoding.

    Parameters
    ----------
    labels : array_like
        Set of labels of a given type.
    time_points : array_like
        Points in time corresponding to the given labels.

    Returns
    -------
    intervals : np.ndarray, shape=(N, 2)
        Start and end times, in seconds.
    labels : list, len=N
        String labels corresponding to the returned intervals.
    """
    assert len(labels) == len(time_points)
    intervals, new_labels = [], []
    idx = 0
    for label, count in run_length_encode(labels):
        start = time_points[idx]
        end = time_points[min([idx + count, len(labels) - 1])]
        idx += count
        intervals += [(start, end)]
        new_labels += [label]
    return np.array(intervals), new_labels


def posterior_to_labeled_intervals(entity, penalty, vocab, **viterbi_args):
    """Decode a posterior Entity to labeled intervals.

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
    intervals : np.ndarray, shape=(N, 2)
        Start and end times, in seconds.
    labels : list, len=N
        String labels corresponding to the returned intervals.
    confidence : list, len=N
        Confidence values (ave. log-likelihoods) of the labels.
    """
    y_idx = viterbi(entity.posterior, penalty=penalty, **viterbi_args)
    labels = vocab.index_to_label(y_idx)
    n_range = np.arange(len(y_idx))
    likelihoods = np.log(entity.posterior[n_range, y_idx])
    idx_intervals = compress_samples_to_intervals(y_idx, n_range)[0]
    boundaries = np.sort(np.unique(idx_intervals.flatten()))
    confidence = boundary_pool(likelihoods, boundaries, pool_func='mean')
    confidence[np.invert(np.isfinite(confidence))] = 0.0
    intervals, labels = compress_samples_to_intervals(
        labels, entity.time_points)
    return intervals, labels, confidence.tolist()


def multi_predict(entity, transform, p_vals, vocab, num_cpus=None):
    """Transform a CQT entity and apply Viterbi decoding.

    Parameters
    ----------
    entity : biggie.Entity
        Entity to estimate.
    transform : optimus.Graph
        Consumes 'cqt' fields, returns 'posterior' fields.
    p_vals : list
        Set of self-transition penalties to apply.

    Returns
    -------
    est_jams : pyjams.JAMS
        Populated JAMS object.
    """
    z = convolve(entity, transform, 'cqt')
    pool = Pool(processes=num_cpus)
    threads = [pool.apply_async(posterior_to_labeled_intervals,
                                (biggie.Entity(posterior=z.posterior,
                                               chord_labels=z.chord_labels),
                                 p, vocab),)
               for p in p_vals]
    pool.close()
    pool.join()

    jam = pyjams.JAMS()
    for penalty, thread in zip(p_vals, threads):
        annot = jam.chord.create_annotation()
        populate_annotation(*thread.get(), annot=annot)
        annot.sandbox.penalty = penalty
    return jam


def populate_annotation(intervals, labels, confidence, annot):
    pyjams.util.fill_range_annotation_data(
        intervals[:, 0], intervals[:, 1], labels, annot)

    for obs, conf in zip(annot.data, confidence):
        obs.label.confidence = conf

    annot.annotation_metadata.data_source = 'machine estimation'
