import argparse
import json
import os
from sklearn.externals.joblib import Parallel, delayed
import time

import marl.fileutils as futil

from dl4mir.common import util
import dl4mir.chords.evaluate as EVAL


METRICS = EVAL.COMPARISONS.keys()
METRICS_ENUM = dict([(k, i) for i, k in enumerate(METRICS)])


def score_one(ref_jamset, jamset_file, min_support):
    est_jamset = util.load_jamset(jamset_file)
    keys = est_jamset.keys()
    keys.sort()

    ref_annots = [ref_jamset[k].chord[0] for k in keys]
    est_annots = [est_jamset[k].chord[0] for k in keys]
    print "[{0}] {1}".format(time.asctime(), jamset_file)
    return EVAL.tally_scores(ref_annots, est_annots, METRICS)


def main(args):
    ref_jamset = util.load_jamset(args.ref_jamset)
    jamset_files = futil.load_textlist(args.jamset_textlist)

    pool = Parallel(n_jobs=args.num_cpus)
    fx = delayed(score_one)
    results = pool(fx(ref_jamset, f, args.min_support) for f in jamset_files)

    results = {f: r for f, r in zip(jamset_files, results)}
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
    parser.add_argument("jamset_textlist",
                        metavar="jamset_textlist", type=str,
                        help="Path to a JAMSet to use as an estimation.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for saving the results as JSON.")
    parser.add_argument("--min_support",
                        metavar="--min_support", type=float, default=60.0,
                        help="Minimum label duration for macro-quality stats.")
    parser.add_argument("--num_cpus",
                        metavar="--num_cpus", type=int, default=8,
                        help="Number of CPUs to use.")
    main(parser.parse_args())
