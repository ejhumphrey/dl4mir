#!/usr/bin/env python
"""Compute a variety of evaluation metrics over a collection of posteriors.

Sample Call:
BASEDIR=/Volumes/Audio/LMD
ipython ejhumphrey/scripts/score_lmd_posteriors.py \
$BASEDIR/test00_posteriors.txt \
$BASEDIR/label_map.txt \
$BASEDIR/test00-stats.txt
"""


import argparse
import numpy as np

from sklearn import metrics
from ejhumphrey.datasets.utils import load_label_enum_map
from ejhumphrey.eval import classification as cls
from ejhumphrey.datasets.lmd import filename_to_genre


def mean_prediction(posterior_file, label_map):
    posterior = np.load(posterior_file)
    y_true = label_map.get(filename_to_genre(posterior_file))
    y_pred = np.mean(posterior, axis=0).argmax()
    return y_true, y_pred


def main(args):
    label_map = load_label_enum_map(args.label_map)
    num_classes = len(np.unique(label_map.values()))

    report = open(args.stats_file, "w")

    y = np.array([mean_prediction(l.strip('\n'),
                                  label_map) for l in open(args.filelist)]).T


    report.write(cls.print_classification_report(args.filelist, y[0], y[1]))
    confusion_matrix = metrics.confusion_matrix(
            y[0], y[1], range(num_classes))
    report.write(cls.print_confusion_matrix(confusion_matrix, num_classes))
    report.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WRITEME")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON dictionary mapping chords to index.")

    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="File to write cumulative stats.")

    main(parser.parse_args())
