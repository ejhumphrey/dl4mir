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
from ejhumphrey.datasets.utils import load_label_enum_map
from ejhumphrey.datasets.utils import filebase
from ejhumphrey.eval import classification as cls
import time
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import cPickle
from marlib.signal import hwr
from sklearn.metrics.metrics import confusion_matrix


global_params = {"results":[],
                 "nochord_thresh":0.1,
                 "rho":1.2,
                 "templates":None}
NUM_CPUS = None


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1)[:, np.newaxis]


def load_prediction(posterior_file, lab_file, framerate, label_map):
    posterior = np.load(posterior_file)
    labels = chordutils.align_lab_file_to_array(posterior, lab_file, framerate)
    y_true = np.array([label_map.get(l) for l in labels])
    return posterior, y_true


def score_fretboard(data_pair):
    fbase, posterior, y_true = data_pair
    medfilt_len = global_params.get("medfilt_len", 0)

    label_map = global_params.get("label_map")
    dshape = posterior.shape
    if medfilt_len:
        posterior = medfilt(posterior, [medfilt_len] + [1] * len(dshape[1:]))

    templates = global_params["param_values"]['classifier/weights'].T
    D = cdist(posterior.reshape(dshape[0], np.prod(dshape[1:])), templates)
    posterior = softmax(-D)

    y_pred = posterior.argmax(axis=1)

    # MIREX Resolution happens here.
    nochord_idx = label_map['N']
    old_nochord_idx = posterior.shape[1] - 1
    # MIREX Resolution happens here.
    if nochord_idx == 24:
        y_pred = np.array([chordutils.wrap_chord_idx_to_MIREX(cidx, old_nochord_idx) for cidx in y_pred])

    y_true, y_pred = chordutils.align_chord_qualities(
        y_true, y_pred, nochord_idx)
    print "%s: %0.2f" % (fbase, 100 * chordutils.fwrr(y_true, y_pred))
    hard_chords = (y_true > 60) == (y_true < 156)
    if np.sum((y_true == y_pred)[hard_chords]):
        hard_idx = np.arange(len(y_true))[(y_true == y_pred) & hard_chords]
        print fbase, ", ".join(["%d:%d" % (i, y_true[i]) for i in hard_idx])
    return y_true, y_pred

def score_penultimate(data_pair):
    """
    TODO(ejhumphrey): This is a temporary hack to make use of the penultimate
    representations produced by accident just prior to the ICASSP submission
    deadline. Fix the hell out of this in the future.
    """
    fbase, posterior, y_true = data_pair
    medfilt_len = global_params.get("medfilt_len", 0)
    label_map = global_params.get("label_map")

    dshape = posterior.shape
    if medfilt_len:
        posterior = medfilt(posterior, [medfilt_len] + [1] * len(dshape[1:]))

    W = global_params["param_values"]['classifier/weights']
    b = global_params["param_values"]['classifier/bias']
    posterior = softmax(hwr(np.dot(posterior, W) + b.reshape(1, len(b))))

    y_pred = posterior.argmax(axis=1)
    nochord_idx = label_map['N']
    old_nochord_idx = posterior.shape[1] - 1
    # MIREX Resolution happens here.
    if nochord_idx == 24:
        y_pred = np.array([chordutils.wrap_chord_idx_to_MIREX(cidx, old_nochord_idx) for cidx in y_pred])

    y_true, y_pred = chordutils.align_chord_qualities(
        y_true, y_pred, nochord_idx)
    print "%s: %0.2f" % (fbase, 100 * chordutils.fwrr(y_true, y_pred))
    return y_true, y_pred


def data_generator(filelist):
    label_map = global_params.get("label_map")
    framerate = global_params.get("framerate")
    for line in filelist:
        posterior_file, lab_file = line.strip("\n").split("\t")
        posterior, y_true = load_prediction(
            posterior_file, lab_file, framerate, label_map)
        yield (filebase(lab_file), posterior, y_true)

# def score_filelist(filelist, label_map, cqt_params, medfilt_len=0, trans_mat=None, report=None):
#    qual_true, qual_pred = [], []
#    chord_true, chord_pred = [], []
#    for line in filelist:
#        posterior_file, lab_file = line.strip("\n").split("\t")
#        posterior, y_true = load_prediction(
#            posterior_file, lab_file, cqt_params, label_map)
#
#        if medfilt_len:
#            posterior = medfilt(posterior, [medfilt_len, 1])
#        if not trans_mat is None:
#            path = chordutils.viterbi_alg(posterior, trans_mat)
#            posterior = np.zeros(posterior.shape)
#            posterior[np.arange(len(posterior)), path] = 1.0
#
#        y_pred = posterior.argmax(axis=1)
#
#        y_trueC, y_predC = chordutils.align_chord_qualities(
#            y_true, y_pred, label_map['N'])
#        qual_true.append(y_trueC)
#        qual_pred.append(y_predC)
#        chord_true.append(y_true)
#        chord_pred.append(y_pred)
#        print "[%s] %s: \t%0.3f" % (time.asctime(),
#                                    filebase(lab_file),
#                                    np.mean(y_true == y_pred) * 100.0)
#        if report:
#            report.write(cls.print_classification_report(
#                posterior_file, y_true, y_pred))
#    chord_true, chord_pred = np.concatenate(chord_true), np.concatenate(chord_pred)
#    qual_true, qual_pred = np.concatenate(qual_true), np.concatenate(qual_pred)
#    return (chord_true, chord_pred), (qual_true, qual_pred)


def accumulate(res):
    global_params["results"].append(res)


def load_transition_matrix(split_file):
    lab_files = [l.strip('\n') for l in open(split_file)]
    label_map = global_params['label_map']
    framerate = global_params['framerate']
    qual_trans_mat = chordutils.cumulative_transition_matrix(
        lab_files, framerate, label_map)
    global_params["trans_mat"] = chordutils.rotate_quality_transitions(qual_trans_mat).T


def main(args):
    cqt_params = json.load(open(args.cqt_params))
    global_params['framerate'] = cqt_params['framerate']
    global_params['label_map'] = load_label_enum_map(args.label_map)
    global_params['medfilt_len'] = args.medfilt
#    if args.trans_mat_file:
#        print "Loading transition matrix"
#        global_params["trans_mat"] = np.load(args.trans_mat_file)

    if args.param_file:
        global_params["param_values"] = cPickle.load(open(args.param_file))

#    report = open(args.stats_file, "w")
#    if args.medfilt:
#        report.write("Using median filter: length = %d" % args.medfilt)

    if args.type == 'fretboard':
        score_fx = score_fretboard
    elif args.type == 'penultimate':
        score_fx = score_penultimate
    else:
#        score_fx = score_posterior
        raise NotImplementedError("Haven't gotten here yet.")

    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=score_fx,
                   iterable=data_generator(open(args.filelist)),
                   callback=accumulate)
    pool.close()
    pool.join()

    y_true, y_pred = [], []
    for track_res in global_params['results'][0]:
        y_true.append(track_res[0])
        y_pred.append(track_res[1])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    num_quals = global_params['label_map']['N'] / 12 + 1
    print "FWRR: %0.2f" % (100 * chordutils.fwrr(y_true, y_pred))
    print "AICA: %0.2f" % (100 * chordutils.aica(y_true, y_pred, num_quals))

    print cls.print_confusion_matrix(confusion_matrix(y_true, y_pred),
                                     top_k_confusions=10)

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

    parser.add_argument("--param_file",
                        action='store', type=str,
                        default="", dest='param_file',
                        help="Template file.")

#    parser.add_argument("stats_file",
#                        metavar="stats_file", type=str,
#                        help="File to write cumulative stats.")

    parser.add_argument("--medfilt",
                        action='store', type=int,
                        default=0, dest='medfilt',
                        help="Length of median filter to apply.")

    parser.add_argument("--type",
                        action='store', type=str,
                        default='posterior', dest='type',
                        help="Type of data to evaluate.")

    main(parser.parse_args())
