"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import json
import glob
import mir_eval
import os
import marl.fileutils as futil
from dl4mir.chords import labels
import time


def align_estimation_to_reference(est_file, ref_file, num_classes=157):
    """

    Parameters
    ----------
    posterior: np.ndarray
        Posteriorgram of chord classes.
    viterbi_penalty: scalar, in (0, inf)
        Self-transition penalty; higher values produce more "stable" paths.

    Returns
    -------
    predictions: dict
        Chord labels and dense count vectors.
    """
    predictions = dict()
    est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_file)
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_file)
    t_min = ref_intervals.min()
    t_max = ref_intervals.max()
    ref_intervals, ref_labels = mir_eval.util.filter_labeled_intervals(
        *mir_eval.util.adjust_intervals(
            ref_intervals, ref_labels, t_min, t_max, "N", "N"))

    est_intervals, est_labels = mir_eval.util.filter_labeled_intervals(
        *mir_eval.util.adjust_intervals(
            est_intervals, est_labels, t_min, t_max, "N", "N"))

    # Merge the time-intervals
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)

    indexes = [labels.chord_label_to_class_index(l, num_classes)
               for l in est_labels]
    for interval, label, idx in zip(intervals, ref_labels, indexes):
        if idx is None:
            raise ValueError(
                "Received an erroneous estimated label: \n"
                "\tfile: %s\ttime: %s" % (est_file, interval.tolist()))
        if not label in predictions:
            predictions[label] = np.zeros(num_classes, dtype=np.int).tolist()
        predictions[label][idx] += float(np.abs(np.diff(interval)))

    return predictions


def main(args):
    results_files = glob.glob(os.path.join(args.results_dir, "*.txt"))
    predictions = dict()
    for idx, result_file in enumerate(results_files):
        key = futil.filebase(result_file.replace(".txt", ''))
        lab_file = os.path.join(args.labs_dir, "%s.lab" % key)
        try:
            predictions[key] = align_estimation_to_reference(result_file, lab_file)
        except IndexError:
            print "Index error: %s" % result_file
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx,
                                        len(results_files), key)

    futil.create_directory(os.path.split(args.output_file)[0])
    with open(args.output_file, 'w') as fp:
        json.dump(predictions, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("results_dir",
                        metavar="results_dir", type=str,
                        help=".")
    parser.add_argument("labs_dir",
                        metavar="labs_file", type=str,
                        help=".")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
