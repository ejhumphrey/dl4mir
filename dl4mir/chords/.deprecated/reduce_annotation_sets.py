"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import json
import mir_eval
import os
import marl.fileutils as futil
import dl4mir.chords.labels as L


def accumulate_estimations(est_intervals, est_labels, ref_intervals,
                           ref_labels, results=None, transpose=False):
    """

    Parameters
    ----------
    est_intervals : np.ndarray, shape=(len(est_labels), 2)
        Estimated start and end times.
    est_labels : list
        Estimated labels.
    ref_intervals : np.ndarray, shape=(len(ref_labels), 2)
        Reference start and end times.
    ref_labels : list
        Reference labels.
    results : dict, or None
        Estimation map for accumulating information; if None, creates an
        empty dictionary.
    transpose : bool, default=False
        Transpose all chord pairs to the equivalent relationship in C.

    Returns
    -------
    results : dict
        Reference labels mapped to a sparse set of estimations and weights.
    """
    if results is None:
        results = dict()

    t_min = ref_intervals.min()
    t_max = ref_intervals.max()
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals, ref_labels, t_min, t_max, L.NO_CHORD, L.NO_CHORD)

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min, t_max, L.NO_CHORD, L.NO_CHORD)

    # Merge the time-intervals
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)

    for interval, ref, est in zip(intervals, ref_labels, est_labels):
        if transpose:
            ref, est = L.relative_transpose(ref, est)
        if not ref in results:
            results[ref] = dict()
        if not est in results:
            results[ref][est] = 0.0
        results[ref][est] += float(np.abs(np.diff(interval)))

    return results


def main(args):
    estimations = json.load(open(args.estimation_set))
    references = json.load(open(args.reference_set))
    results = dict()
    for idx, key in enumerate(estimations):
        est_data = estimations[key]
        ref_data = references.get(key, None)
        if not ref_data is None:
            accumulate_estimations(est_data['intervals'], est_data['labels'],
                                   ref_data['intervals'], ref_data['labels'],
                                   results, transpose=args.transpose)

    futil.create_directory(os.path.split(args.output_file)[0])
    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("estimation_set",
                        metavar="estimation_set", type=str,
                        help="Path to a file of estimated labeled intervals.")
    parser.add_argument("reference_set",
                        metavar="reference_set", type=str,
                        help="Path to a file of reference labeled intervals.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Estimations collapsed to a single object.")
    parser.add_argument("--transpose",
                        metavar="--transpose", type=bool, default=True,
                        help="Transpose all chord pairs to a C reference.")
    main(parser.parse_args())
