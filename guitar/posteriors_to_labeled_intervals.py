"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
from itertools import groupby
import json
import marl.chords.utils as cutils
from scipy import signal
import optimus
import os
import marl.fileutils as futils
import time

from ejhumphrey.dl4mir.chords import index_to_chord_label


def encode(l):
    return [(len(list(group)), name) for name, group in groupby(l)]


def compress_samples_to_intervals(labels, framerate):
    framerate = float(framerate)
    intervals, new_labels = [], []
    last_end = -1.0/framerate
    for duration, l in encode(labels):
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

    if viterbi_penalty > 0:
        indexes = cutils.viterbi_alg(posterior, rho=viterbi_penalty)
    else:
        indexes = posterior.argmax(axis=1)

    vocab = posterior.shape[1]
    return [index_to_chord_label(idx, vocab) for idx in indexes]


def main(args):
    dset = optimus.File(args.posterior_file)
    futils.create_directory(os.path.split(args.output_file)[0])
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    framerate = json.load(open(args.cqt_params))['framerate']
    total_count = len(dset.keys())
    results = dict()
    for idx, key in enumerate(dset.keys()):
        labels = predict_posterior(
            dset.get(key).posterior.value,
            viterbi_penalty=args.viterbi_penalty,
            medfilt=args.medfilt)
        intervals, labels = compress_samples_to_intervals(labels, framerate)
        results[key] = dict(intervals=intervals, labels=labels)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an optimus file of chord posteriors.")
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
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
