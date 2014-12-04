"""Evaluation framework for chord estimation."""
import re
import mir_eval
import dl4mir.chords.labels as L
import numpy as np
import marl.fileutils as futil
import pyjams


def align_labeled_intervals(ref_intervals, ref_labels,
                            est_intervals, est_labels):
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
        ref_intervals, ref_labels, t_min, t_max, L.NO_CHORD, L.NO_CHORD)

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min, t_max, L.NO_CHORD, L.NO_CHORD)

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
        ref_intervals=ref_annot.intervals,
        ref_labels=ref_annot.labels.value,
        est_intervals=est_annot.intervals,
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
    valid_qualities = ['maj', 'min', 'maj7', 'min7', '7', 'maj6', 'min6',
                       'dim', 'aug', 'sus4', 'sus2', 'dim7', 'hdim7', '']
    valid_refs = np.array([L.split(_)[1] in valid_qualities
                           for _ in reference_labels])
    valid_semitones = np.array([mir_eval.chord.QUALITIES[name]
                                for name in valid_qualities])

    (ref_roots, ref_semitones,
        ref_basses) = L.encode_many(reference_labels, False)
    (est_roots, est_semitones,
        est_basses) = L.encode_many(estimated_labels, False)

    eq_root = ref_roots == est_roots
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = (eq_root * eq_semitones).astype(np.float)

    # Test for reference chord inclusion
    is_valid = np.array([np.all(np.equal(ref_semitones, semitones), axis=1)
                         for semitones in valid_semitones])
    # Drop if NOR
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1
    comparison_scores[np.invert(valid_refs)] = -1
    comparison_scores[np.not_equal(ref_basses, ref_roots)] = -1
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


class Evaluator(object):

    MATCH = 'match'
    ERROR = 'error'
    TOTAL = 'total'
    LABELS = 'labels'
    WEIGHTS = 'weights'

    def __init__(self, metric, transpose=True):
        self.correct_weight = 0.0
        self.total_weight = 0.0
        self.assignments = dict()
        self.metric = metric
        self.fx = COMPARISONS[metric]
        self.transpose = transpose
        self.reset()

    def reset(self):
        """Clear out statistic accumulators."""
        self.correct_weight *= 0.0
        self.total_weight *= 0.0
        self.assignments = dict()

    def tally(self, ref_annot, est_annot):
        """Tally a pair of chord annotations to the running scores.

        Parameters
        ----------
        ref_labels : array_like, shape=(n,)
            Reference chord labels.
        est_labels : array_like, shape=(n,)
            Estimated chord labels.
        weights : array_like, shape(n,); or None
            Relative contribution of each chord pair.
        """
        weights, ref_labels, est_labels = align_chord_annotations(
            ref_annot, est_annot, transpose=self.transpose)

        if weights is None:
            weights = np.ones(len(ref_labels), dtype=float)

        assert len(ref_labels) == len(est_labels) == len(weights)

        scores = self.fx(ref_labels, est_labels)
        self.total_weight += ((scores >= 0.0) * weights).sum()
        for ref, est, s, w in zip(ref_labels, est_labels, scores, weights):
            # Drop negatives
            if s < 0:
                continue

            # Sanitize names for consistency
            ref = L.join(*L.split(ref))
            est = L.join(*L.split(est))

            # Tally overall accuracy
            self.correct_weight += w * float(s)

            b = Evaluator.MATCH if s else Evaluator.ERROR
            if not ref in self.assignments:
                self.assignments[ref] = {Evaluator.MATCH: dict(),
                                         Evaluator.ERROR: dict()}
            if not est in self.assignments[ref][b]:
                self.assignments[ref][b][est] = 0.0
            self.assignments[ref][b][est] += w

    def scores(self):
        macro = self.correct_weight / float(self.total_weight)
        class_scores = []
        for assignments in self.confusions(10, True).values():
            class_scores.append(
                np.sum(assignments[Evaluator.MATCH][Evaluator.WEIGHTS]))
        return dict(
            macro=macro,
            class_micro=np.mean(class_scores))

    def confusions(self, top_k=5, normalize=True):
        confusions = dict()
        for ref_label, estimations in self.assignments.items():
            confs = dict()
            total = np.sum([np.sum(_.values()) for _ in estimations.values()])
            for k in Evaluator.MATCH, Evaluator.ERROR:
                confs[k] = {Evaluator.LABELS: list(),
                            Evaluator.WEIGHTS: list()}
                labels = estimations[k].keys()
                counts = np.array([estimations[k][l] for l in labels])
                sorted_idx = np.argsort(counts)[::-1]
                for idx in sorted_idx[:top_k]:
                    confs[k][Evaluator.LABELS].append(labels[idx])
                    confs[k][Evaluator.WEIGHTS].append(counts[idx])
                    if normalize:
                        confs[k][Evaluator.WEIGHTS][-1] /= total

            confusions[ref_label] = confs

        return confusions


def score_many(reference_files, estimated_files, metrics=None,
               top_k_confusions=5, ref_pattern='', est_pattern=''):
    """Tabulate overall scores for a collection of key-aligned annotations.

    Parameters
    ----------
    reference_files : list
        Filepaths to a set of reference annotations.
    estimated_files : list
        Filepaths to a set of estimated annotations.
    metrics : list, or default=None
        Metric names to compute overall scores; if None, use all.
    top_k_confusions : int, default=5
        Number of confusions to return, in descending order.
    ref_pattern : str, default=''
        Pattern to use for filtering reference annotation keys.
    est_pattern : str, default=''
        Pattern to use for filtering estimated annotation keys.

    Returns
    -------
    scores : dict
        Resulting overall scores, keyed by metric.
    """
    if metrics is None:
        metrics = COMPARISONS.keys()

    evaluators = dict([(metric, Evaluator(metric=metric))
                       for metric in metrics])
    for ref, est in zip(reference_files, estimated_files):
        ref_key = futil.filebase(ref)
        est_key = futil.filebase(est)
        if ref_key != est_key:
            raise ValueError(
                "File keys do not match: %s != %s" % (ref_key, est_key))
        ref = pyjams.load(ref)
        est = pyjams.load(est)

        # All reference annotations vs all estimated annotations.
        for ref_annot in ref.chord:
            # Match against the given reference key pattern.
            if re.match(ref_pattern, ref_annot.sandbox.key) is None:
                continue
            for est_annot in est.chord:
                # Match against the given estimation key pattern.
                if re.match(est_pattern, est_annot.sandbox.key) is None:
                    continue
                for e in evaluators.values():
                    e.tally(ref_annot, est_annot)

    return dict([(metric, e.scores()) for metric, e in evaluators.items()])


def score_many_trackwise(reference_files, estimated_files, metrics=None,
                         ref_pattern='', est_pattern=''):
    """Tabulate overall scores for a collection of key-aligned annotations.

    Parameters
    ----------
    reference_files : list
        Filepaths to a set of reference annotations.
    estimated_files : list
        Filepaths to a set of estimated annotations.
    metrics : list, or default=None
        Metric names to compute overall scores; if None, use all.
    ref_pattern : str, default=''
        Pattern to use for filtering reference annotation keys.
    est_pattern : str, default=''
        Pattern to use for filtering estimated annotation keys.

    Returns
    -------
    results : np.ndarray, shape=(n, i, j, k)
        Resulting table of trackwise scores.
    tracks : list of str, len=n
        Track ids corresponding to the rows in the table.
    ref_annotators : list of str, len=i
        Annotator names corresponding to axis=1 in the table.
    est_annotators : list of str, len=j
        Annotator names corresponding to axis=2 in the table.
    metrics : list of str, len=k
        Metric names corresponding to axis=3 in the table.
    """
    if metrics is None:
        metrics = COMPARISONS.keys()

    results = list()
    tracks = list()
    ref_annotators = set()
    est_annotators = set()

    for ref, est in zip(reference_files, estimated_files):
        ref_key = futil.filebase(ref)
        est_key = futil.filebase(est)
        if ref_key != est_key:
            raise ValueError(
                "File keys do not match: %s != %s" % (ref_key, est_key))
        tracks.append(ref_key)
        results.append(dict())
        ref = pyjams.load(ref)
        est = pyjams.load(est)

        # All reference annotations vs all estimated annotations.
        for ref_annot in ref.chord:
            # Match against the given reference key pattern.
            if re.match(ref_pattern, ref_annot.sandbox.key) is None:
                continue
            ra_name = ref_annot.sandbox.key
            ref_annotators.add(ra_name)
            if not ra_name in results[-1]:
                results[-1][ra_name] = dict()

            for est_annot in est.chord:
                # Match against the given estimation key pattern.
                if re.match(est_pattern, est_annot.sandbox.key) is None:
                    continue
                ea_name = est_annot.sandbox.key
                est_annotators.add(ea_name)
                if not ea_name in results[-1][ra_name]:
                    results[-1][ra_name][ea_name] = dict()

                for metric in metrics:
                    e = Evaluator(metric=metric)
                    e.tally(ref_annot, est_annot)
                    results[-1][ra_name][ea_name][metric] = e.scores()['macro']

    table = np.zeros([len(tracks), len(ref_annotators),
                      len(est_annotators), len(metrics)], dtype=np.float)
    ref_annotators, est_annotators = list(ref_annotators), list(est_annotators)
    for n, res in enumerate(results):
        for i, ra_name in enumerate(ref_annotators):
            for j, ea_name in enumerate(est_annotators):
                for k, metric in enumerate(metrics):
                    table[n, i, j, k] = res.get(
                        ra_name, {}).get(ea_name, {}).get(metric, np.nan)
    return table, tracks, ref_annotators, est_annotators, metrics
