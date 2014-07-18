"""Apply a graph convolutionally to datapoints in an biggie file."""

import argparse
import numpy as np
import json
import biggie
import os
import marl.fileutils as futil
import time


def estimate_classes(entity):
    """

    Parameters
    ----------
    posterior: np.ndarray
        Posteriorgram of chord classes.
    viterbi_penalty: scalar, in (0, inf)
        Self-transition penalty; higher values produce more "stable" paths.

    Returns
    -------
    estimations: dict
        Chord labels and dense count vectors.
    """
    num_classes = entity.posterior.value.shape[1]
    estimations = dict()
    for label, idx in zip(entity.chord_labels.value,
                          entity.posterior.value.argmax(axis=1)):
        if not label in estimations:
            estimations[label] = np.zeros(num_classes, dtype=np.int).tolist()
        estimations[label][idx] += 1

    return estimations


def main(args):
    dset = biggie.Stash(args.posterior_file)
    estimations = dict()
    for idx, key in enumerate(dset.keys()):
        estimations[key] = estimate_classes(dset.get(key))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(dset), key)

    futil.create_directory(os.path.split(args.estimation_file)[0])
    with open(args.estimation_file, 'w') as fp:
        json.dump(estimations, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an optimus file of chord posteriors.")
    # Outputs
    parser.add_argument("estimation_file",
                        metavar="estimation_file", type=str,
                        help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
