#!/usr/bin/env python
"""Compute a variety of evaluation metrics over a collection of posteriors.

Sample Call:
BASEDIR=/media/attic/chords
ipython ejhumphrey/scripts/score_chord_posteriors.py \
$BASEDIR/chord_posterior_test0.txt \
$BASEDIR/MIREX09_chord_map.txt \
$BASEDIR/cqts/cqt_params_20130926.txt \
$BASEDIR/chord_posterior_test0-stats.txt
"""


import argparse
import numpy as np

import json
from ejhumphrey.datasets import chordutils
from sklearn import metrics
from ejhumphrey.datasets.utils import load_label_enum_map
from ejhumphrey.eval import classification as cls


def load_prediction(posterior_file, lab_file, cqt_params, label_map):
    posterior = np.load(posterior_file)
    labels = chordutils.align_lab_file_to_array(
        posterior, lab_file, cqt_params.get("framerate"))

    y_pred = posterior.argmax(axis=1)
    y_true = [label_map.get(l) for l in labels]
    return y_true, y_pred


def main(args):
    cqt_params = json.load(open(args.cqt_params))
    label_map = load_label_enum_map(args.label_map)
    num_classes = len(np.unique(label_map.values()))
    confusion_matrix = np.zeros([num_classes, num_classes], dtype=int)

    report = open(args.stats_file, "w")
    for line in open(args.filelist):
        posterior_file, lab_file = line.strip("\n").split("\t")
        y_true, y_pred = load_prediction(
            posterior_file, lab_file, cqt_params, label_map)
        confusion_matrix += metrics.confusion_matrix(
            y_true, y_pred, range(num_classes))
        report.write(
            cls.print_classification_report(posterior_file, y_true, y_pred))

    report.write(cls.print_confusion_matrix(confusion_matrix, 5))
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
