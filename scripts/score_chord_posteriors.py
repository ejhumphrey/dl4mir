#!/usr/bin/env python
"""Compute a variety of evaluation metrics over a collection of posteriors.

Sample Call:
BASEDIR=/Volumes/Audio/Chord_Recognition
ipython ejhumphrey/scripts/score_chord_posteriors.py \
$BASEDIR/chord_posterior_list.txt \
$BASEDIR/MIREX09_chord_map.txt \
$BASEDIR/cqt_params.txt \
$BASEDIR/test-stats.txt
"""


import argparse
import numpy as np

import json
from ejhumphrey.datasets import chordutils
from sklearn import metrics

def load_prediction(posterior_file, lab_file, cqt_params, label_map):
    posterior = np.load(posterior_file)
    labels = chordutils.align_lab_file_to_array(
        posterior, lab_file, cqt_params.get("framerate"))

    y_pred = posterior.argmax(axis=1)
    y_true = [label_map.get(l) for l in labels]
    return y_true, y_pred

def print_confusion_matrix(matrix, top_k_confusions=5):
    header = "Confusions"
    msg = "\n%s\n%s\n%s\n" % ("-" * len(header),
                              header,
                              "-" * len(header))
    true_positives = 0
    for i, row in enumerate(matrix.copy()):
        true_positives += row[i]
        total = max([row.sum(), 1])
        row *= 100.0 / float(total)
        msg += "%3d: %0.2f\t[" % (i, row[i])
        row[i] = -1
        idx = row.argsort()[::-1]
        msg += ", ".join(["%3d: %0.2f" % (j, row[j]) for j in idx[:top_k_confusions]])
        msg += "]\n"

    header = "Total Precision: %0.3f" % (true_positives / float(matrix.sum()))
    msg += "\n%s\n%s\n%s\n" % ("-" * len(header),
                              header,
                              "-" * len(header))
    return msg

def print_classification_report(name, y_true, y_pred):
    hdr = "%s\n%s\n%s\n" % ("-" * len(name), name, "-" * len(name))
    return hdr + "%s\n" % metrics.classification_report(y_true, y_pred)

def main(args):
    cqt_params = json.load(open(args.cqt_params))
    label_map = chordutils.load_label_map(args.label_map)
    num_classes = len(np.unique(label_map.values()))
    confusion_matrix = np.zeros([num_classes, num_classes], dtype=int)

    report = open(args.stats_file, "w")
    for line in open(args.filelist):
        posterior_file, lab_file = line.strip("\n").split("\t")
        y_true, y_pred = load_prediction(
            posterior_file, lab_file, cqt_params, label_map)
        confusion_matrix += metrics.confusion_matrix(
            y_true, y_pred, range(num_classes))
        report.write(print_classification_report(posterior_file, y_true, y_pred))

    report.write(print_confusion_matrix(confusion_matrix, 5))
    report.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON dictionary mapping chords to index.")

    parser.add_argument("cqt_params",
                        metavar="cqt_params", type=str,
                        help="A JSON text file with parameters for the CQT.")

    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="File to write cumulative stats.")

    main(parser.parse_args())
