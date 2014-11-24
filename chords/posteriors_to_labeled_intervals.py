"""Viterbi decode a stash of posteriors and output labeled intervals."""

import argparse
import json
import biggie
import os
import marl.fileutils as futils
import time

from dl4mir.chords.lexicon import Strict
from dl4mir.common.util import run_length_encode
from dl4mir.common.util import viterbi


def compress_samples_to_intervals(labels, time_points):
    assert len(labels) == len(time_points)
    intervals, new_labels = [], []
    idx = 0
    for label, count in run_length_encode(labels):
        start = time_points[idx]
        end = time_points[min([idx + count, len(labels) - 1])]
        idx += count
        intervals += [(start, end)]
        new_labels += [label]
    return intervals, new_labels


def posterior_to_labeled_intervals(entity, penalty, vocab):
    y_idx = viterbi(entity.posterior, penalty=penalty)
    labels = vocab.index_to_label(y_idx)
    return compress_samples_to_intervals(labels, entity.time_points)


def main(args):
    stash = biggie.Stash(args.posterior_file)
    output_dir = futils.create_directory(args.output_directory)
    stats = json.load(open(args.validation_file))
    penalty = float(stats['best_config']['penalty'])
    vocab = Strict(157)
    total_count = len(stash.keys())
    for idx, key in enumerate(stash.keys()):
        intervals, labels = posterior_to_labeled_intervals(
            stash.get(key), penalty=penalty, vocab=vocab)
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
    parser.add_argument("validation_file",
                        metavar="validation_file", type=str,
                        help="")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for labeled intervals as JSON.")
    main(parser.parse_args())
