import argparse
import json
import os
import tabulate

import marl.fileutils as futil

from dl4mir.common import util
import dl4mir.chords.evaluate as EVAL


METRICS = EVAL.COMPARISONS.keys()
METRICS_ENUM = dict([(k, i) for i, k in enumerate(METRICS)])


def main(args):
    ref_jamset = util.load_jamset(args.ref_jamset)
    est_jamset = util.load_jamset(args.est_jamset)
    keys = est_jamset.keys()
    keys.sort()

    ref_annots = [ref_jamset[k].chord[0] for k in keys]
    est_annots = [est_jamset[k].chord[0] for k in keys]

    scores, supports = EVAL.score_annotations(ref_annots, est_annots, METRICS)
    results = dict(metrics=METRICS,
                   score_annotations=(scores.tolist(), supports.tolist()))
    scores_macro = scores.mean(axis=0)
    scalar = supports.sum(axis=0)
    scalar[scalar == 0] = 1.0
    scores_micro = (supports * scores).sum(axis=0) / scalar

    print tabulate.tabulate(
        [['macro'] + scores_macro.tolist(), ['micro'] + scores_micro.tolist()],
        headers=[''] + METRICS)

    label_counts = EVAL.reduce_annotations(ref_annots, est_annots, METRICS)

    mac_aves = []
    for m in METRICS:
        (labels, scores,
            support) = EVAL.macro_average(label_counts[m], True, 0.0)
        mac_aves.append([labels, scores.tolist(), support.tolist()])

    results.update(macro_average=mac_aves)

    output_dir = os.path.split(args.output_file)[0]
    futil.create_directory(output_dir)

    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("ref_jamset",
                        metavar="ref_jamset", type=str,
                        help="Path to a JAMSet to use as a reference.")
    parser.add_argument("est_jamset",
                        metavar="est_jamset", type=str,
                        help="Path to a JAMSet to use as an estimation.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for saving the results as JSON.")
    parser.add_argument("--min_support",
                        metavar="--min_support", type=float, default=0.0,
                        help="Minimum label duration for macro-quality stats.")
    main(parser.parse_args())
