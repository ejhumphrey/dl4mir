import pyjams
import mir_eval
import dl4mir.chords.labels as L
import numpy as np


def align_intervals(ref_intervals, ref_labels, est_intervals, est_labels,
                    transpose=False):
    """

    Parameters
    ----------
    est_intervals : np.ndarray, shape=(n, 2)
        Estimated start and end times.
    est_labels : list, shape=(n,)
        Estimated labels.
    ref_intervals : np.ndarray, shape=(n, 2)
        Reference start and end times.
    ref_labels : list, shape=(n,)
        Reference labels.
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

    ref_roots, ref_semitones = L.encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = L.encode_many(estimated_labels, False)[:2]

    eq_root = ref_roots == est_roots
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = (eq_root * eq_semitones).astype(np.float)

    # Test for reference chord inclusion
    is_valid = np.array([np.all(np.equal(ref_semitones, semitones), axis=1)
                         for semitones in valid_semitones])
    # Drop if NOR
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1
    comparison_scores[np.invert(valid_refs)] = -1
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


class Scores(object):

    CORRECT = 'correct'
    ERRORS = 'errors'
    TOTAL = 'total'
    LABELS = 'labels'
    COUNTS = 'counts'

    def __init__(self):
        self.results = dict()
        self.confusions = dict()
        self.reset()

    def reset(self):
        """Clear out statistic accumulators."""
        for metric in COMPARISONS:
            self.results[metric] = dict(correct=0.0, total=0.0)
            self.confusions[metric] = dict()

    def accumulate(self, ref_labels, est_labels, weights=None):
        """Add a set of results to the score accumulator.

        Parameters
        ----------
        ref_labels : array_like, shape=(n,)
            Reference chord labels.
        est_labels : array_like, shape=(n,)
            Estimated chord labels.
        weights : array_like, shape(n,); or None
            Relative contribution of each chord pair.
        """
        if weights is None:
            weights = np.ones(len(ref_labels), dtype=float)

        assert len(ref_labels) == len(est_labels) == len(weights)

        for metric, fx in COMPARISONS.items():
            scores = fx(ref_labels, est_labels)
            total_weight = ((scores >= 0.0) * weights).sum()
            self.results[metric][Scores.TOTAL] = total_weight
            for ref, est, v, w in zip(ref_labels, est_labels, scores, weights):
                # Drop negatives
                if v < 0:
                    continue

                # Sanitize names for consistency
                ref = L.join(*L.split(ref))
                est = L.join(*L.split(est))

                b = Scores.CORRECT if v else Scores.ERRORS
                if not ref in self.confusions[metric]:
                    self.confusions[metric][ref] = {Scores.CORRECT: dict(),
                                                    Scores.ERRORS: dict()}
                if not est in self.confusions[metric][ref][b]:
                    self.confusions[metric][ref][b][est] = 0.0
                self.confusions[metric][ref][b][est] += w

                self.results[metric][Scores.CORRECT] += w * float(v)
                self.results[metric][Scores.TOTAL] += w * float(v)

    def compute_score(self):
        metrics = self.results.keys()
        metrics.sort()
        totals = dict()
        for metric in metrics:
            totals[metric] = self.results[metric][Scores.CORRECT]
            totals[metric] /= float(self.results[metric][Scores.TOTAL])
        return totals

    def compute_confusions(self, top_k=5, normalize=True):
        metrics = self.confusions.keys()
        metrics.sort()
        confusions = dict()
        for metric in metrics:
            confusions[metric] = dict()
            for act_label, result in self.confusions[metric].items():
                confs = dict()
                total = np.sum([np.sum(_.values()) for _ in result.values()])
                for k in Scores.CORRECT, Scores.ERRORS:
                    confs[k] = {Scores.LABELS: list(), Scores.COUNTS: list()}
                    labels = result[k].keys()
                    counts = np.array([result[k][l] for l in labels])
                    sorted_idx = np.argsort(counts)[::-1]
                    for idx in sorted_idx[:top_k]:
                        confs[k][Scores.LABELS].append(labels[idx])
                        confs[k][Scores.COUNTS].append(counts[idx])
                        if normalize:
                            confs[k][Scores.COUNTS][-1] /= total

                confusions[metric][act_label] = confs

        return confusions
