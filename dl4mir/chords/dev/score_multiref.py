"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import json
import mir_eval
import os
import marl.fileutils as futil


def main(args):
    estimations = json.load(open(args.estimation_set))
    reference_set = [json.load(open(_)) for _ in args.reference_set]
    best_refs = dict()
    worst_refs = dict()

    for idx, key in enumerate(estimations):
        est_data = estimations[key]
        for ridx, ref in enumerate(references):
            ref_data = ref.get(key, None)
            if not ref_data is None:
                results[key] = mir_eval.chord.evaluate(
                    est_data['intervals'], est_data['labels'],
                    ref_data['intervals'], ref_data['labels'])

    futil.create_directory(os.path.split(args.output_file)[0])
    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("estimation_set",
                        metavar="estimation_set", type=str,
                        help="Path to a set of estimated labeled intervals.")
    parser.add_argument("reference_sets",
                        metavar="reference_sets", type=str, nargs='+',
                        help="Paths to sets of reference labeled intervals.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Estimations collapsed to a single object.")
    main(parser.parse_args())
