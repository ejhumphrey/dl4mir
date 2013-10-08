'''
Created on Oct 8, 2013

@author: ejhumphrey
'''

import argparse
import numpy as np


import glob
import numpy as np
import os
import time

from marl.hewey.core import context_slice

from ejhumphrey.dnn.core.graphs import Network
from ejhumphrey.dnn import utils


import json
from ejhumphrey.datasets import chordutils
from sklearn import metrics
from ejhumphrey.datasets.utils import load_label_enum_map, filebase, \
    expand_filebase
from ejhumphrey.eval import classification as cls
import cPickle

def context_sample(X, indexes, left, right, newshape=None):
    batch = []
    for i in indexes:
        x_i = context_slice(
            value=X, index=i, left=left, right=right, fill_value=0)
        if newshape:
            x_i = np.reshape(x_i, newshape=newshape)
        batch.append(x_i)
    return np.asarray(batch)

def load_label_enums(cqt_array, lab_file, cqt_params, label_map):
    labels = chordutils.align_lab_file_to_array(
        cqt_array, lab_file, cqt_params.get("framerate"))

    return np.asarray([label_map.get(l, -1) for l in labels])

def collect_model_files(model_directory):
    # Find the model definition.
    def_files = glob.glob(os.path.join(model_directory, "*.definition"))
    assert len(def_files) == 1, \
        "More than one definition file found? %s" % def_files

    # Collect parameters.
    param_files = glob.glob(os.path.join(model_directory, "*.params"))
    assert len(param_files) > 0, "No parameter files matched?"
    param_files.sort()

    # Find the configuration file. 
    config_params = glob.glob(os.path.join(model_directory, "*.config"))
    assert len(config_params) == 1, \
        "More than one configuration file found? %s" % config_params
    return def_files[0], param_files, config_params[0]

def score(posterior, y_true):
    """
    Returns true positives.
    """
    y_pred = posterior.argmax(axis=1)
    return np.sum(y_true == y_pred)


def main(args):
    def_file, param_files, config_file = collect_model_files(args.model_directory)

    dnet = Network.load(def_file)
    dnet.compile()
    train_params = utils.json_load(config_file).get("train_params")

    cqt_params = json.load(
        open(os.path.join(args.cqt_directory, "cqt_params.txt")))
    label_map = load_label_enum_map(args.label_map)

    index_map = {}
    inputs = dnet.empty_inputs()
    stats_file = os.path.join(args.model_directory, "validation-stats.txt")
    fh = open(stats_file, "w")
    fh.close()
    for param_file in param_files:
        dnet.param_values = cPickle.load(open(param_file))
        true_positives, total = 0.0, 0.0
        print "[%s] Evaluating %s" % (time.asctime(), filebase(param_file))
        for n, line in enumerate(open(args.splitlist)):
            fbase = line.strip("\n")

            cqt_file = expand_filebase(fbase, args.cqt_directory, ".npy")
            assert os.path.exists(cqt_file)

            lab_file = expand_filebase(fbase, args.lab_directory, ".lab")
            assert os.path.exists(lab_file)

            cqt_array = np.load(cqt_file)
            y_true = load_label_enums(cqt_array, lab_file, cqt_params, label_map)
            if not lab_file in index_map:
                index_map[lab_file] = np.random.permutation(
                    len(y_true))[:args.num_samples]

            inputs[dnet.input_name] = context_sample(
                cqt_array,
                index_map.get(lab_file),
                left=train_params.get("left"),
                right=train_params.get("right"),
                newshape=dnet.input_shape)

            y_true = context_sample(
                y_true, index_map.get(lab_file), left=0, right=0).squeeze()

            y_pred = dnet(inputs).argmax(axis=1)
            true_positives += (y_true == y_pred).sum()
            total += len(y_true)
            if (n % 20) == 0:
                print "[%s] After %3d: %0.3f" % \
                    (time.asctime(), n, 100 * true_positives / total)

        print "[%s] Final: %0.3f" % \
                (time.asctime(), 100 * true_positives / total)
        fh = open(stats_file, 'a')
        fh.write("%s\t%0.3f\n" % (filebase(param_file),
                                  100 * true_positives / total))
        fh.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("model_directory",
                        metavar="model_directory", type=str,
                        help="Directory containing a '*.definition', a set "
                        "of files matching '*.params', and a '*.config' file.")

    parser.add_argument("splitlist",
                        metavar="splitlist", type=str,
                        help="Text file of filebases.")

    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Directory to look for CQT arrays.")

    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory to look for CQT arrays.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON dictionary mapping chords to index.")

    parser.add_argument("num_samples",
                        metavar="num_samples", type=int,
                        help="Number of samples per file to evaluate.")

    main(parser.parse_args())
