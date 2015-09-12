"""Evaluation module for chord estimation."""
import fnmatch
import mir_eval
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

import dl4mir.chords.labels as L
import dl4mir.chords.lexicon as lex

STRICT = lex.Strict(157)


def align_labeled_intervals(ref_intervals, ref_labels, est_intervals,
                            est_labels, ref_fill_value=L.NO_CHORD,
                            est_fill_value=L.NO_CHORD):
    """Align two sets of labeled intervals.

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n, 2)
        Reference start and end times.
    ref_labels : list, shape=(n,)
        Reference labels.
    est_intervals : np.ndarray, shape=(n, 2)
        Estimated start and end times.
    est_labels : list, shape=(n,)
        Estimated labels.

    Returns
    -------
    durations : np.ndarray, shape=(m, 2)
        Time durations (weights) of each aligned interval.
    ref_labels : list, shape=(m,)
        Reference labels.
    est_labels : list, shape=(m,)
        Estimated labels.
    """
    t_min = ref_intervals.min()
    t_max = ref_intervals.max()
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals, ref_labels, t_min, t_max,
        ref_fill_value, ref_fill_value)

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min, t_max,
        est_fill_value, est_fill_value)

    # Merge the time-intervals
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    return durations, ref_labels, est_labels


def align_chord_annotations(ref_annot, est_annot, transpose=False):
    """Align two JAMS chord range annotations.

    Parameters
    ----------
    ref_annot : pyjams.range_annotation
        Range Annotation to use as a chord reference.
    est_annot : pyjams.range_annotation
        Range Annotation to use as a chord estimation.
    transpose : bool, default=False
        Transpose all chord pairs to the equivalent relationship in C.

    Returns
    -------
    durations : np.ndarray, shape=(m, 2)
        Time durations (weights) of each aligned interval.
    ref_labels : list, shape=(m,)
        Reference labels.
    est_labels : list, shape=(m,)
        Estimated labels.
    """
    durations, ref_labels, est_labels = align_labeled_intervals(
        ref_intervals=np.asarray(ref_annot.intervals),
        ref_labels=ref_annot.labels.value,
        est_intervals=np.asarray(est_annot.intervals),
        est_labels=est_annot.labels.value)

    if transpose:
        ref_labels, est_labels = L.relative_transpose(ref_labels, est_labels)

    return durations, ref_labels, est_labels


def v157_strict(reference_labels, estimated_labels):
    '''Compare chords along lexicon157 rules. Chords with qualities
    outside the following are ignored:
        [maj, min, maj7, min7, 7, maj6, min6,
        dim, aug, sus4, sus2, dim7, hdim7, N]

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> durations = mir_eval.util.intervals_to_durations(intervals)
        >>> comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
        >>> score = mir_eval.chord.weighted_accuracy(comparisons, durations)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    mir_eval.chord.validate(reference_labels, estimated_labels)

    ref_idx = STRICT.label_to_index(reference_labels)
    est_idx = STRICT.label_to_index(estimated_labels)
    is_invalid = np.equal(ref_idx, None)

    comparison_scores = np.equal(ref_idx, est_idx).astype(float)

    # Drop if invalid
    comparison_scores[is_invalid] = -1.0
    return comparison_scores


COMPARISONS = dict(
    thirds=mir_eval.chord.thirds,
    triads=mir_eval.chord.triads,
    tetrads=mir_eval.chord.tetrads,
    root=mir_eval.chord.root,
    mirex=mir_eval.chord.mirex,
    majmin=mir_eval.chord.majmin,
    sevenths=mir_eval.chord.sevenths,
    v157_strict=v157_strict)


def pairwise_score_labels(ref_labels, est_labels, weights, compare_func):
    """Tabulate the score and weight for a pair of annotation labels.

    Parameters
    ----------
    ref_annot : pyjams.RangeAnnotation
        Chord annotation to use as a reference.
    est_annot : pyjams.RangeAnnotation
        Chord annotation to use as a estimation.
    compare_func : method
        Function to use for comparing a pair of chord labels.

    Returns
    -------
    score : float
        Average score, in [0, 1].
    weight : float
        Relative weight of the comparison, >= 0.
    """
    scores = compare_func(ref_labels, est_labels)
    valid_idx = scores >= 0
    total_weight = weights[valid_idx].sum()
    correct_weight = np.dot(scores[valid_idx], weights[valid_idx])
    norm = total_weight if total_weight > 0 else 1.0
    return correct_weight / norm, total_weight


def pairwise_reduce_labels(ref_labels, est_labels, weights, compare_func,
                           label_counts=None):
    """Accumulate estimated timed of a collection label pairs.

    Parameters
    ----------
    ref_annot : pyjams.RangeAnnotation
        Chord annotation to use as a reference.
    est_annot : pyjams.RangeAnnotation
        Chord annotation to use as a estimation.
    compare_func : method
        Function to use for comparing a pair of chord labels.

    Returns
    -------
    label_counts : dict
        Map of reference labels to estimated label counts and support.
    """
    scores = compare_func(ref_labels, est_labels)

    if label_counts is None:
        label_counts = dict()

    for ref, est, s, w in zip(ref_labels, est_labels, scores, weights):
        if s < 0:
            continue
        if not ref in label_counts:
            label_counts[ref] = dict()
        if not est in label_counts[ref]:
            label_counts[ref][est] = dict(count=0.0, support=0.0)
        label_counts[ref][est]['count'] += s*w
        label_counts[ref][est]['support'] += w

    return label_counts


def pair_annotations(ref_jams, est_jams, ref_pattern='*', est_pattern='*'):
    """Align annotations given a collection of jams and regex patterns.

    Note: Uses glob-style filepath matching. See fnmatch.fnmatch for more info.

    Parameters
    ----------
    ref_jams : list
        A set of reference jams.
    est_jams : list
        A set of estimated jams.
    ref_pattern : str, default='*'
        Pattern to use for filtering reference annotation keys.
    est_pattern : str, default='*'
        Pattern to use for filtering estimated annotation keys.

    Returns
    -------
    ref_annots, est_annots : lists, len=n
        Equal length lists of corresponding annotations.
    """
    ref_annots, est_annots = [], []
    for ref, est in zip(ref_jams, est_jams):
        # All reference annotations vs all estimated annotations.
        for ref_annot in ref.chord:
            # Match against the given reference key pattern.
            if not fnmatch.fnmatch(ref_annot.sandbox.key, ref_pattern):
                continue
            for est_annot in est.chord:
                # Match against the given estimation key pattern.
                if not fnmatch.fnmatch(est_annot.sandbox.key, est_pattern):
                    continue
                ref_annots.append(ref_annot)
                est_annots.append(est_annot)

    return ref_annots, est_annots


def score_annotations(ref_annots, est_annots, metrics):
    """Tabulate overall scores for two sets of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    scores : np.ndarray, shape=(n, k)
        Resulting annotation-wise scores.
    weights : np.ndarray
        Relative weight of each score.
    """
    scores, support = np.zeros([2, len(ref_annots), len(metrics)])
    for n, (ref_annot, est_annot) in enumerate(zip(ref_annots, est_annots)):
        (weights, ref_labels,
            est_labels) = align_chord_annotations(ref_annot, est_annot)
        for k, metric in enumerate(metrics):
            scores[n, k], support[n, k] = pairwise_score_labels(
                ref_labels, est_labels, weights, COMPARISONS[metric])

    return scores, support


def score_annotations_parallel(ref_annots, est_annots, metrics, num_cpus=8):
    """Tabulate overall scores for two sets of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    scores : np.ndarray, shape=(n, k)
        Resulting annotation-wise scores.
    weights : np.ndarray
        Relative weight of each score.
    """
    def score_one(n, k, ref_annot, est_annot, metric):
        (weights, ref_labels,
            est_labels) = align_chord_annotations(ref_annot, est_annot)
        return (n, k, pairwise_score_labels(
            ref_labels, est_labels, weights, COMPARISONS[metric]))

    def gen(ref_annots, est_annots, metrics):
        for n, (ref, est) in enumerate(zip(ref_annots, est_annots)):
            for k, metric in enumerate(metrics):
                yield (n, k, ref, est, metric)

    scores, support = np.zeros([2, len(ref_annots), len(metrics)])
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(score_one)
    results = pool(fx(*args) for args in gen(ref_annots, est_annots, metrics))
    for n, k, res in results:
        scores[n, k], support[n, k] = res

    return scores, support


def reduce_annotations(ref_annots, est_annots, metrics):
    """Collapse annotations to a sparse matrix of label estimation supports.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    all_label_counts : list of dicts
        Sparse matrix mapping {metric, ref, est, support} values.
    """
    label_counts = dict([(m, dict()) for m in metrics])
    for ref_annot, est_annot in zip(ref_annots, est_annots):
        weights, ref_labels, est_labels = align_chord_annotations(
            ref_annot, est_annot, transpose=True)
        for metric in metrics:
            pairwise_reduce_labels(ref_labels, est_labels, weights,
                                   COMPARISONS[metric], label_counts[metric])

    return label_counts


def macro_average(label_counts, sort=True, min_support=0):
    """Tally the support of each reference label in the map.

    Parameters
    ----------
    label_counts : dict
        Map of reference labels to estimations, containing a `support` count.
    sort : bool, default=True
        Sort the results in descending order.
    min_support : scalar
        Minimum support value for returned results.

    Returns
    -------
    labels : list, len=n
        Unique reference labels in the label_counts set.
    scores : np.ndarray, len=n
        Resulting label-wise scores.
    support : np.ndarray, len=n
        Support values corresponding to labels and scores.
    """
    N = len(label_counts)
    labels = [''] * N
    scores, supports = np.zeros([2, N], dtype=float)
    for idx, (ref_label, estimations) in enumerate(label_counts.items()):
        labels[idx] = ref_label
        supports[idx] = sum([_['support'] for _ in estimations.values()])
        scores[idx] = sum([_['count'] for _ in estimations.values()])
        scores[idx] /= supports[idx] if supports[idx] > 0 else 1.0

    labels = np.asarray(labels)
    if sort:
        sidx = np.argsort(supports)[::-1]
        labels, scores, supports = labels[sidx], scores[sidx], supports[sidx]

    # Boolean mask of results with adequate support.
    midx = supports >= min_support
    return labels[midx].tolist(), scores[midx], supports[midx]


def tally_scores(ref_annots, est_annots, min_support, metrics=None):
    """Produce cumulative statistics over a paired set of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    min_support : scalar
        Minimum support value for macro-quality measure.
    metrics : list, len=k, default=all
        Metric names to compute overall scores.

    Returns
    -------
    results : dict
        Score dictionary of {statistic, metric, value} results.
    """
    if metrics is None:
        metrics = COMPARISONS.keys()

    scores, supports = score_annotations(ref_annots, est_annots, metrics)
    scores_macro = scores.mean(axis=0)
    scores_micro = (supports * scores).sum(axis=0) / supports.sum(axis=0)

    results = dict(macro=dict(), micro=dict(), macro_quality=dict())
    for m, smac, smic in zip(metrics, scores_macro, scores_micro):
        results['macro'][m] = smac
        results['micro'][m] = smic

    label_counts = reduce_annotations(ref_annots, est_annots, metrics)
    for m in metrics:
        quality_scores = macro_average(
            label_counts[m], sort=True, min_support=min_support)[1]
        results['macro_quality'][m] = quality_scores.mean()
    return results
