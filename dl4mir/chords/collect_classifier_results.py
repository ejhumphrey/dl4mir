"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import json
import optimus
import os
import marl.fileutils as futil
import time


def predict_entity(entity):
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
    num_classes = entity.posterior.value.shape[1]
    predictions = dict()
    for label, idx in zip(entity.chord_labels.value,
                          entity.posterior.value.argmax(axis=1)):
        if not label in predictions:
            predictions[label] = np.zeros(num_classes, dtype=np.int).tolist()
        predictions[label][idx] += 1

    return predictions


def main(args):
    predictions = dict()
    split = futil.filebase(args.posterior_file)
    dset = optimus.File(args.posterior_file)
    for idx, key in enumerate(dset.keys()):
        predictions[split][key] = predict_entity(dset.get(key))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx,
                                        len(dset), key)

    futil.create_directory(os.path.split(args.output_file)[0])
    with open(args.output_file, 'w') as fp:
        json.dump(predictions, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an optimus file of chord posteriors.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
