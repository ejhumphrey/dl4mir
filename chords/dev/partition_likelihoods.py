import argparse
import biggie
import json
import numpy as np
import time

import dl4mir.common.util as util
import dl4mir.chords.labels as L
import dl4mir.chords.data as D


def quality_likelihood_histogram(stash, bins=100, val_range=(0, 1)):
    likelihoods = dict()
    for idx, key in enumerate(stash.keys()):
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(stash), key)
        entity = stash.get(key)
        posterior = np.asarray(entity.posterior)
        vocab_dim = posterior.shape[1]

        class_idx = L.chord_label_to_class_index(
            entity.chord_labels, vocab_dim)
        quality_idx = D.quality_map(entity, vocab_dim)
        for pdf, cidx, qidx in zip(posterior, class_idx, quality_idx):
            if None in [qidx, cidx]:
                continue
            if not qidx in likelihoods:
                likelihoods[qidx] = list()
            likelihoods[qidx].append(pdf[cidx])

    for idx in likelihoods:
        likelihoods[idx] = np.histogram(
            likelihoods[idx], bins=bins, range=val_range)
    bins = np.array([likelihoods[n][0] for n in range(14)])
    return bins, likelihoods[0][1]


def collision_histogram(stash, bins=100, val_range=(0, 1)):
    likelihoods = dict()
    for idx, key in enumerate(stash.keys()):
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(stash), key)
        entity = stash.get(key)
        posterior = np.asarray(entity.posterior)
        vocab_dim = posterior.shape[1]

        class_idx = L.chord_label_to_class_index(
            entity.chord_labels, vocab_dim)
        quality_idx = D.quality_map(entity, vocab_dim)
        for pdf, cidx, qidx in zip(posterior, class_idx, quality_idx):
            if None in [qidx, cidx]:
                continue
            if not qidx in likelihoods:
                likelihoods[qidx] = list()

            pdf[cidx] = -1
            likelihoods[qidx].append(pdf.max())

    for idx in likelihoods:
        likelihoods[idx] = np.histogram(
            likelihoods[idx], bins=bins, range=val_range)
    bins = np.array([likelihoods[n][0] for n in range(14)])
    return bins, likelihoods[0][1]


def likelihood_threshold(entity, thresholds, label_map=D.quality_map):
    quality_idx = label_map(entity)
    posterior = np.asarray(entity.posterior)
    vocab_dim = posterior.shape[1]
    class_idx = L.chord_label_to_class_index(
        entity.chord_labels, vocab_dim)
    for idx in range(len(quality_idx)):
        cidx, qidx = class_idx[idx], quality_idx[idx]
        if qidx is None:
            continue
        if thresholds[qidx] > posterior[idx, cidx]:
            quality_idx[idx] = None
    return quality_idx


def likelihood_collision_threshold(entity, thresholds):
    quality_idx = D.quality_map(entity)
    posterior = np.asarray(entity.posterior)
    vocab_dim = posterior.shape[1]
    class_idx = L.chord_label_to_class_index(
        entity.chord_labels, vocab_dim)
    for idx in range(len(quality_idx)):
        cidx, qidx = class_idx[idx], quality_idx[idx]
        if qidx is None:
            continue
        posterior[idx, cidx] = -1
        if thresholds[qidx] < posterior[idx, :].max():
            quality_idx[idx] = None
    return quality_idx


def main(args):
    stash = biggie.Stash(args.posterior_file)
    thresh = [0.02] * 5 + [0.0] * 9
    partition_labels = util.partition(stash, likelihood_threshold, thresh)
    partition_labels = dict([(k, v.tolist())
                             for k, v in partition_labels.items()])
    with open(args.partition_file, 'w') as fp:
        json.dump(partition_labels, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to a biggie stash of posteriors.")
    # Outputs
    parser.add_argument("partition_file",
                        metavar="partition_file", type=str,
                        help="Path to write the resulting partition labels.")
    main(parser.parse_args())
