"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import json
import numpy as np
import mir_eval.chord as chord_eval
import mir_eval.util as util
import time

FIELDS = ['intervals', 'labels']
VOCABULARIES = ['root', 'majmin', 'sevenths']
IGNORED = 'ignored'
COMPARISONS = 'comparisons'
DURATION = 'duration'
CORRECT = 'correct'


def compare_annotations(ref_intervals, ref_labels, est_intervals, est_labels):
    ref_intervals = np.asarray(ref_intervals)
    est_intervals = np.asarray(est_intervals)

    t_min = ref_intervals.min()
    t_max = ref_intervals.max()
    ref_intervals, ref_labels = util.filter_labeled_intervals(
        *util.adjust_intervals(
            ref_intervals, ref_labels, t_min, t_max,
            chord_eval.NO_CHORD, chord_eval.NO_CHORD))

    est_intervals, est_labels = util.filter_labeled_intervals(
        *util.adjust_intervals(
            est_intervals, est_labels, t_min, t_max,
            chord_eval.NO_CHORD, chord_eval.NO_CHORD))

    merged_data = util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)

    results = dict([(v, {COMPARISONS: [], IGNORED: []}) for v in VOCABULARIES])

    for vocab in VOCABULARIES:
        total = 0.0
        correct = 0.0
        for interval, ref_label, est_label in zip(*merged_data):
            weight = float(np.abs(np.diff(interval)))
            row = [ref_label, est_label, weight]
            one_score = chord_eval.METRICS[vocab](
                [ref_label], [est_label], np.array([interval]))
            if one_score is None:
                results[vocab][IGNORED].append(row)
            else:
                results[vocab][COMPARISONS].append(row + [one_score])
                correct += weight*one_score
                total += weight
        results[vocab][DURATION] = total
        results[vocab][CORRECT] = correct

    return results


def collate_results(results):
    totals = dict([(v, [0.0, 0.0, 0.0]) for v in VOCABULARIES])
    for v in VOCABULARIES:
        for key in results:
            totals[v][0] += results[key][v][CORRECT]
            totals[v][1] += results[key][v][DURATION]
            totals[v][2] += (totals[v][0] / totals[v][1])
        totals[v] = (totals[v][0] / totals[v][1], totals[v][2] / len(results))
    return totals


def main(args):
    estimations = json.load(open(args.estimations_file))
    references = json.load(open(args.references_file))

    results = dict()
    total_count = len(estimations)
    for idx, key in enumerate(estimations):
        if not key in references:
            print "Warning: Key '%s' not in references." % key
            continue
        ref_intervals, ref_labels = [references[key][n] for n in FIELDS]
        est_intervals, est_labels = [estimations[key][n] for n in FIELDS]
        results[key] = compare_annotations(
            ref_intervals, ref_labels, est_intervals, est_labels)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    print collate_results(results)
    # with open(args.output_file, 'w') as fp:
    #     json.dump(results, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("estimations_file",
                        metavar="estimations_file", type=str,
                        help="Path to a JSON file of estimated annotations.")
    parser.add_argument("references_file",
                        metavar="references_file", type=str,
                        help="Path to a JSON file of reference annotations.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
