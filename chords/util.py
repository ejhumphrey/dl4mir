import numpy as np
from dl4mir.common.util import run_length_encode
from dl4mir.common.util import viterbi


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
    """
    y_idx = viterbi(entity.posterior, penalty=penalty, **viterbi_args)
    labels = vocab.index_to_label(y_idx)
    return compress_samples_to_intervals(labels, entity.time_points)
