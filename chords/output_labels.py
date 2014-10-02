"""Apply a graph convolutionally to datapoints in a biggie Stash."""

import argparse
import numpy as np
import json
from marl.chords.utils import viterbi_alg
from scipy import signal
import biggie
import os
import marl.fileutils as futils
import time

from dl4mir.chords.labels import index_to_chord_label
from dl4mir.common.util import run_length_encode


def compress_samples_to_intervals(labels, framerate):
    framerate = float(framerate)
    intervals, new_labels = [], []
    last_end = -1.0/framerate
    for l, duration in run_length_encode(labels):
        intervals += [[last_end, last_end + duration / framerate]]
        last_end = intervals[-1][-1]
        new_labels += [l]
    intervals[0][0] = 0.0
    return intervals, new_labels


def predict_posterior(posterior, viterbi_penalty=0, medfilt=0):
    """Transform a cqt-based entity with a given network.

    Parameters
    ----------
    posterior: np.ndarray
        Posteriorgram of chord classes.
    viterbi_penalty: scalar, in (0, inf)
        Self-transition penalty; higher values produce more "stable" paths.

    Returns
    -------
    labels: list
        List of chord label names (str).
    """

    if medfilt > 0:
        posterior = signal.medfilt(posterior, [medfilt, 1])
        posterior /= posterior.sum(axis=1)[:, np.newaxis]

    if viterbi_penalty > 0:
        indexes = viterbi_alg(posterior, rho=viterbi_penalty)
    else:
        indexes = posterior.argmax(axis=1)

    vocab = posterior.shape[1]
    return [index_to_chord_label(idx, vocab) for idx in indexes]


def posterior_to_labeled_intervals(posterior, framerate=20.0,
                                   viterbi_penalty=0.0, medfilt=0):
    labels = predict_posterior(posterior, viterbi_penalty, medfilt)
    return compress_samples_to_intervals(labels, framerate)


def main(args):
    dset = biggie.Stash(args.posterior_file)
    output_dir = futils.create_directory(args.output_directory)

    framerate = json.load(open(args.cqt_params))['framerate']
    total_count = len(dset.keys())
    for idx, key in enumerate(dset.keys()):
        intervals, labels = posterior_to_labeled_intervals(
            dset.get(key).posterior,
            framerate=framerate,
            viterbi_penalty=args.viterbi_penalty,
            medfilt=args.medfilt)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)
        output_file = os.path.join(output_dir, "%s.json" % key)
        with open(output_file, 'w') as fp:
            json.dump(dict(intervals=intervals, labels=labels), fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an biggie stash of chord posteriors.")
    parser.add_argument("viterbi_penalty",
                        metavar="viterbi_penalty", type=float,
                        help="Penalty for the viterbi algorithm (skip if 0).")
    parser.add_argument("medfilt",
                        metavar="medfilt", type=int,
                        help="Length of median filter; must be odd.")
    parser.add_argument("cqt_params",
                        metavar="cqt_params", type=str,
                        help="JSON file containing parameters of the cqt.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for labeled intervals as JSON.")
    main(parser.parse_args())
