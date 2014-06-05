#!/usr/bin/env python
"""Compute a variety of evaluation metrics over a collection of posteriors.

Sample Call:
BASEDIR=/media/attic/chords
ipython ejhumphrey/scripts/score_chord_posteriors.py \
$BASEDIR/chord_posterior_test0.txt \
$BASEDIR/MIREX09_chord_map.txt \
$BASEDIR/cqts/cqt_params_20130926.txt \
$BASEDIR/chord_posterior_test0-stats.txt
"""


import argparse
import numpy as np
from scipy.signal.signaltools import medfilt
import json
from ejhumphrey.datasets import chordutils
from sklearn import metrics
from ejhumphrey.datasets.utils import load_label_enum_map
from ejhumphrey.datasets.utils import filebase
from ejhumphrey.eval import classification as cls
import time
from multiprocessing import Pool
import cPickle


global_params = {"results":[],
                 "y_true":[],
                 "y_pred":[], }
NUM_CPUS = None


def load_prediction(posterior_file, lab_file, framerate, label_map):
    posterior = np.load(posterior_file)
    labels = chordutils.align_lab_file_to_array(posterior, lab_file, framerate)
    y_true = np.array([label_map.get(l) for l in labels])
    return posterior, y_true


def score_posterior(data):
    fbase, posterior, y_true, param_set = data
    a, b, c, d = param_set
    trans_mat = global_params.get("trans_mat")
    label_map = global_params.get("label_map")

    y_pred = chordutils.chord_viterbi(posterior, trans_mat, a, b, c, d)
    y_true, y_pred = chordutils.align_chord_qualities(
        y_true, y_pred, label_map['N'])

    print "%s:\t%0.2f" % (fbase, 100 * chordutils.fwrr(y_true, y_pred))
    return (y_true, y_pred)


def data_generator(posterior_files, lab_files, param_set):
    label_map = global_params.get("label_map")
    framerate = global_params.get("framerate")
    for posterior_file, lab_file in zip(posterior_files, lab_files):
        posterior, y_true = load_prediction(
            posterior_file, lab_file, framerate, label_map)
        yield (filebase(lab_file), posterior, y_true, param_set)


def accumulate(res):
    global_params["results"].append(res)


def compute_transition_matrix(filelist):
    lab_files = [l.strip('\n') for l in open(filelist)]
    label_map = global_params['label_map']
    framerate = global_params['framerate']
    qual_trans_mat = chordutils.cumulative_transition_matrix(
        lab_files, framerate, label_map)
    global_params["trans_mat"] = chordutils.rotate_quality_transitions(qual_trans_mat).T

def create_params(pop_size, mu, sig):
    params = []
    for mu_i, sig_i in zip(mu, sig):
        params.append(np.random.normal(loc=mu_i, scale=sig_i, size=(pop_size,)))
    return np.array(params).T


def eval_posteriors(posterior_files, lab_files, param_set):
    global_params['results'] = []
    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=score_posterior,
                   iterable=data_generator(posterior_files, lab_files, param_set),
                   callback=accumulate)
    pool.close()
    pool.join()
    y_true, y_pred = [], []
    for track_res in global_params['results'][0]:
        y_true.append(track_res[0])
        y_pred.append(track_res[1])

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return chordutils.fwrr(y_true, y_pred), chordutils.aica(y_true, y_pred)


def parse_filelist(filelist):
    posterior_files, lab_files = [], []
    for line in filelist:
        file_pair = line.strip("\n").split("\t")
        if len(file_pair) != 2:
            break
        posterior_files.append(file_pair[0])
        lab_files.append(file_pair[1])

    return posterior_files, lab_files


def main(args):
    cqt_params = json.load(open(args.cqt_params))
    global_params['framerate'] = cqt_params['framerate']
    global_params['label_map'] = load_label_enum_map(args.label_map)
    if args.trans_mat_file:
        print "Loading transition matrix"
        global_params["trans_mat"] = np.load(args.trans_mat_file)
    else:
        pass

    posterior_files, lab_files = parse_filelist(open(args.filelist))
    pop_size = 10
    mu = np.zeros(4)
    sigma = np.ones(4)
    params = create_params(pop_size, mu, sigma)
    iter_count, max_iter = 0, 100
    best_score, best_param_set = -1, None
    print "Starting loop"
    while iter_count < max_iter:
        fitness = []
        for param_set in params:
            fwrr, aica = eval_posteriors(
                posterior_files[:50], lab_files[:50], param_set)
            msg = "%s\tFWRR: %0.3f\tAICA: %0.3f" % (param_set, 100 * fwrr, 100 * aica)
            print "-"*len(msg) + "\n" + msg + "\n" + "-"*len(msg)
            fitness.append(aica)
        w = np.array(fitness)
        score = w.max()
        if score > best_score:
            best_param_set = params[w.argmax(), :]
            best_score = score
            print "\nNew optima: %0.3f - %s\n" % (score * 100, best_param_set)
        else:
            pass
#            print "Best: %0.3f - %s" % (best_score * 100, best_param_set)
        iter_count += 1

        params = create_params(pop_size, best_param_set, sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON dictionary mapping chords to index.")

    parser.add_argument("cqt_params",
                        metavar="cqt_params", type=str,
                        help="A JSON text file with parameters for the CQT.")

    parser.add_argument("--trans_mat_file",
                        action='store', type=str,
                        default="", dest='trans_mat_file',
                        help="Transition matrix to load.")

    main(parser.parse_args())
