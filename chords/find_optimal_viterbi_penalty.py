"""Apply a graph convolutionally to datapoints in an biggie file."""

import argparse
import numpy as np
import json
import biggie
import os
import marl.fileutils as futil
import time
import warnings
import sklearn.metrics as metrics

import dl4mir.common.util as util
import dl4mir.chords.labels as CL


def predict(entity, penalty, weighting=None):
    """

    Parameters
    ----------
    entity: biggie.Entity
        Posteriorgram of chord classes.
    penalty: scalar, in (0, inf)
        Self-transition penalty.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    posterior = np.array(entity.posterior)
    if not weighting is None:
        posterior *= weighting.reshape(1, posterior.shape[1])
    y_true = CL.chord_label_to_class_index(entity.chord_labels, 157)
    y_pred = util.viterbi(posterior, np.ones([157]*2), penalty=penalty)
    L = min([len(y_true), len(y_pred)])
    y_true = np.asarray(y_true[:L])
    y_pred = np.asarray(y_pred[:L])
    idx = np.not_equal(y_true, None)
    return y_true[idx].astype(int), y_pred[idx]


def f1_score(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics.f1_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    if not set(y_pred).intersection(set(y_true)):
        return 0.0
    kwargs = dict()
    if len(set(y_true)) == 1:
        kwargs.update(pos_label=np.unique(y_true)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return metrics.recall_score(y_true, y_pred, **kwargs)


def penalty_sweep(stash, penalties, scoring_fx=f1_score):
    all_scores = []
    for p in penalties:
        scores, weights = np.zeros([2, len(stash)])
        for idx, key in enumerate(stash.keys()):
            y_true, y_pred = predict(stash.get(key), p)
            scores[idx] = scoring_fx(y_true, y_pred)
            weights[idx] = len(y_true)
            print "[%s] %12d / %12d: %s // f1: %0.4f" % (time.asctime(), idx,
                                                         len(stash), key,
                                                         scores[idx])
        all_scores.append(((scores * weights) / weights.sum()).sum())
    return all_scores


def weighting_sweep(stash, prior, exponents, penalty=-6, scoring_fx=f1_score):
    all_scores = []
    for e in exponents:
        scores, weights = np.zeros([2, len(stash)])
        for idx, key in enumerate(stash.keys()):
            y_true, y_pred = predict(stash.get(key), penalty, prior**e)
            scores[idx] = scoring_fx(y_true, y_pred)
            weights[idx] = len(y_true)
            print "[%s] %12d / %12d: %s // f1: %0.4f" % (time.asctime(), idx,
                                                         len(stash), key,
                                                         scores[idx])
        all_scores.append(((scores * weights) / weights.sum()).sum())
    return all_scores


def main(args):
    print args.posterior_file
    if not os.path.exists(args.posterior_file):
        print "File does not exist: %s" % args.posterior_file
        return
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
