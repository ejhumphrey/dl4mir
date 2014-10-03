"""Compute final evaluation metrics and confusions for chord estimations."""

import argparse
import json
import numpy as np
import os
import marl.fileutils as futil
import sklearn.metrics as metrics
import warnings

import dl4mir.chords.labels as L
import dl4mir.chords.lexicon as lex


def collapse_estimations(estimations):
    """Accumulate a set of track-wise estimations to a flattened set.

    Parameters
    ----------
    estimations: dict
        Dict of track keys, pointing to another dict of chord labels and
        class counts.

    Returns
    -------
    results: dict
        Dict of chord labels and corresponding class counts.
    """

    total = dict()
    for key in estimations:
        for label, counts in estimations[key].items():
            if not label in total:
                total[label] = np.zeros_like(np.array(counts))
            total[label] += np.array(counts)
    return total


def confusion_matrix(results, num_classes, lexicon):
    """Deprecated: Don't use.

    Note: Confusion matrix is actual x estimations.

    Parameters
    ----------
    ??
    """
    raise NotImplementedError("Get outta here!")
    classifications = np.zeros([num_classes, num_classes])
    for label, counts in results.items():
        idx = lexicon.label_to_index(label, num_classes)
        if idx is None:
            continue
        classifications[idx, :] += counts
    return classifications


def quality_confusion_matrix(results, lexicon):
    """Populate a quality-rotated confusion matrix.

    Parameters
    ----------
    results : dict
        A flattened dictionary of chord labels, pointing to count vectors.

    Returns
    -------
    confusions : np.ndarrays
        The quality-rotated confusion matrix.
    """
    num_classes = 157
    qual_conf = np.zeros([num_classes, num_classes])
    chord_idxs = lexicon.label_to_index(results.keys())
    for chord_idx, counts in zip(chord_idxs, results.values()):
        if chord_idx is None:
            continue
        quality_idx = int(chord_idx) / 12
        root = chord_idx % 12
        counts = L.rotate(counts, root) if quality_idx != 13 else counts
        qual_conf[quality_idx*12, :] += counts
    return qual_conf


def confusions_to_str(confusions, lexicon, top_k=5):
    """Render the confusion matrix to human-readable text.

    Parameters
    ----------
    confusions : np.ndarray, shape=(num_classes, num_classes)
        Quality-rotated confusions.
    top_k : int
        Number of top confusions to print.

    Returns
    -------
    text : str
        Pretty confusions.
    """
    if confusions.shape[0] == 157:
        confusions = confusions[::12, :]

    outputs = []
    for idx, row in enumerate(confusions):
        row /= float(row.sum())
        line = "%7s (%7.4f) ||" % (lexicon.index_to_label(idx*12),
                                   row[idx*12]*100)
        sidx = row.argsort()[::-1]
        k = 0
        count = 0
        while count < top_k:
            if sidx[k] != idx*12:
                line += " %7s (%7.4f) |" % \
                    (lexicon.index_to_label(sidx[k]),
                     row[sidx[k]]*100)
                count += 1
            k += 1
        # print line
        outputs.append(line)
    return "\n".join(outputs) + "\n"


def compute_scores(estimations, lexicon):
    """Compute scores over a dataset.

    Parameters
    ----------
    estimations : dict
        A set of chord tallies, where chord labels point to a vector of counts.

    Returns
    -------
    stats : dict
        Weighted and averaged (classwise) statistics over the estimations.
    confusions : np.ndarray
        Confusion matrix over the estimations.
    """
    results = collapse_estimations(estimations)
    confusions = quality_confusion_matrix(results, lexicon)
    quality_true, quality_est = confusions_to_comparisons(confusions)
    stats = dict()
    with warnings.catch_warnings():
        labels = range(157)
        warnings.simplefilter("ignore")

        stats['precision_weighted'] = metrics.precision_score(
            quality_true, quality_est, labels=labels, average='weighted')
        stats['precision_averaged'] = np.mean(metrics.precision_score(
            quality_true, quality_est, labels=labels, average=None)[::12])

        stats['recall_weighted'] = metrics.recall_score(
            quality_true, quality_est, labels=labels, average='weighted')
        stats['recall_averaged'] = np.mean(metrics.recall_score(
            quality_true, quality_est, labels=labels, average=None)[::12])

        stats['f1_weighted'] = metrics.f1_score(
            quality_true, quality_est, labels=labels, average='weighted')
        stats['f1_averaged'] = np.mean(metrics.f1_score(
            quality_true, quality_est, labels=labels, average=None)[::12])

    return stats, confusions


def stats_to_string(stats):
    """Render the stats dictionary to human readable text.

    Parameters
    ----------
    stats : dict
        Has the following keys:
            [precision_weighted, recall_weighted, f1_weighted,
             precision_averaged, recall_averaged, f1_averaged,]

    Returns
    -------
    text : str
        Pretty stats.
    """
    stat_line = "  Precision: %0.4f\t Recall: %0.4f\tf1: %0.4f"
    res1 = "Weighted: " + stat_line % (100*stats['precision_weighted'],
                                       100*stats['recall_weighted'],
                                       100*stats['f1_weighted'])

    res2 = "Averaged: " + stat_line % (100*stats['precision_averaged'],
                                       100*stats['recall_averaged'],
                                       100*stats['f1_averaged'])
    res3 = "-"*72
    outputs = [res3, res1, res2, res3]
    return "\n".join(outputs) + '\n'


def confusions_to_comparisons(confusions):
    """Convert a confusion matrix to sklearn-style aligned class predictions.

    Parameters
    ----------
    confusions : np.ndarray, shape=(num_classes, num_classes)
        Overall confusion matrix; rows are actual classes, columns are
        predicted classes.

    Returns
    -------
    y_true, y_pred : np.ndarrays, shape=(N,)
        The actual and predicted class labels of the observations.
    """
    y_true, y_pred = [], []
    confusions = np.round(confusions).astype(int)
    for i in range(confusions.shape[0]):
        for j in range(confusions.shape[1]):
            count = confusions[i, j]
            y_true.append(i + np.zeros(count, dtype=int))
            y_pred.append(j + np.zeros(count, dtype=int))
    return np.concatenate(y_true), np.concatenate(y_pred)


def main(args):
    if not os.path.exists(args.estimation_file):
        print "File does not exist: %s" % args.estimation_file
        return
    vocab = lex.Strict(157)
    stats, confusions = compute_scores(
        json.load(open(args.estimation_file)), lexicon=vocab)
    res_str = stats_to_string(stats) + confusions_to_str(confusions, vocab)
    futil.create_directory(os.path.split(args.stats_file)[0])
    with open(args.stats_file, 'w') as fp:
        fp.write(res_str)
    print "\n%s\n%s" % (args.estimation_file, res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("estimation_file",
                        metavar="estimation_file", type=str,
                        help="Path to a JSON file of estimations.")
    # Outputs
    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="Path for the resulting statistics as plaintext.")
    main(parser.parse_args())
