"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import json
import numpy as np
import mir_eval.chord as chord_eval
from ejhumphrey.dl4mir import chords as C
import mir_eval.util as util
import time

# np.mean([acqa_n[n, n*12] for n in range(13)] + [acqa_n[-1,-1]])
# np.sum([acqa[n, n*12] for n in range(13)] + [acqa[-1,-1]]) / float(np.sum(acqa))
# np.sum([acqa[n, n*12] for n in range(6)] + [acqa[-1,-1]]) / float(np.sum(acqa[:6]))


def rotate_to_C(posterior, root):
    return np.array([posterior[(n + root) % 12 + 12*(n/12)]
                     for n in range(len(posterior) - 1)]+[posterior[-1]])


def aggregate_over_labels(predictions):
    total = dict()
    for key in predictions:
        for label, counts in predictions[key].items():
            if not label in total:
                total[label] = np.zeros_like(np.array(counts))
            total[label] += np.array(counts)
    return total


def chord_recall(split):
    acc = np.zeros([157, 157])
    for label, counts in split.items():
        idx = C.chord_label_to_index(label, 157)
        if idx is None:
            continue
        acc[idx, :] += counts
    return acc


def average_chord_quality_accuracy(predictions):
    vocab = 157
    acqa = np.zeros([14, vocab])
    ignored_total = 0
    for label, counts in predictions.items():
        root, semitones, bass = chord_eval.encode(label)
        qidx = 13 if label == 'N' else C.get_quality_index(semitones, vocab)
        if qidx is None:
            ignored_total += np.sum(counts)
            continue
        acqa[qidx, :] += rotate_to_C(counts, root) if qidx != 13 else counts
    acqa_norm = acqa / acqa.astype(float).sum(axis=1).reshape(14, 1)
    return acqa, acqa_norm, ignored_total


def main(args):
    predictions = json.load(open(args.prediction_file))
    acqa, acqa_norm = average_chord_quality_accuracy(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("prediction_file",
                        metavar="prediction_file", type=str,
                        help="Path to a JSON file of predictions.")
    # Outputs
    # parser.add_argument("output_file",
    #                     metavar="output_file", type=str,
    #                     help="Path for the lab-file style output as JSON.")
    main(parser.parse_args())
