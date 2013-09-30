#!/usr/bin/env python
"""Consumes a filelist of cqt-arrays, producing posteriorgrams.

Sample Call:
MODELDIR=/home/ejhumphrey/chords/models/majmin_chord_classifier_000
ipython ejhumphrey/scripts/predict_chords.py \
/home/ejhumphrey/chords/cqt_list.txt \
$MODELDIR/majmin_chord_classifier_000.definition \
$MODELDIR/majmin_chord_classifier_000_final-20130928.params \
$MODELDIR/majmin_chord_classifier_000-train_params.txt \
"""


import argparse
import numpy as np

from marl.hewey.core import context_slice
from ejhumphrey.dnn.core.graphs import load

import json
import os
import time

def context_slicer(X, left, right, batch_size=100, newshape=None):
    """Generator to step through a CQT array as batches of datapoints.

    Parameters
    ----------
    X : np.ndarray
        CQT array
    left : int
        Previous context.
    right : int
        Subsequent context.
    batch_size : int
        Number of datapoints to return each iteration.
    newshape : array_like
        New shape for each datapoint.

    Yields
    ------
    batch : np.ndarray
        Array of datapoints. First dimension is batch_size, with the potential
        exception of the last iteration.
    """
    batch = list()
    for i in range(len(X)):
        x_i = context_slice(
            value=X, index=i, left=left, right=right, fill_value=0)
        if newshape:
            x_i = np.reshape(x_i, newshape=newshape)
        batch.append(x_i)
        if len(batch) == batch_size:
            yield np.array(batch)
            batch = list()
    yield np.array(batch)

def predict_cqt(cqt_array, dnet, train_params, batch_size=100):
    """
    Parameters
    ----------
    cqt_array : np.ndarray
        CQT array.
    dnet : dnn.core.graphs.Network()
        Instantiated and compiled net.
    train_params : dict
        Must have a "left" and "right" value.

    Returns
    -------
    posterior : np.ndarray
        Output prediction surface.
    """
    gen = context_slicer(X=cqt_array,
                         left=train_params.get("left"),
                         right=train_params.get("right"),
                         batch_size=batch_size,
                         newshape=dnet.input_shape)
    posterior = None
    inputs = dnet.empty_inputs()
    for i, batch in enumerate(gen):
        inputs[dnet.input_name] = batch
        batch_posterior = dnet(inputs)
        if posterior is None:
            posterior = np.zeros([len(cqt_array), batch_posterior.shape[1]])
        idx0, idx1 = i * batch_size, i * batch_size + len(batch_posterior)
        posterior[idx0:idx1] = batch_posterior
    return posterior


def main(args):
    dnet = load(args.def_file, args.param_file)
    model_name = os.path.split(os.path.splitext(args.def_file)[0])[-1]
    train_params = json.load(open(args.train_params))
    for cqt_file in open(args.filelist):
        cqt_file = cqt_file.strip("\n")
        posterior = predict_cqt(np.load(cqt_file), dnet, train_params)
        output_file = "%s-%s-posterior.npy" % (cqt_file.strip("-cqt.npy"),
                                               model_name)
        np.save(output_file, posterior)
        print "[%s] Finished: %s" % (time.asctime(), output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("def_file",
                        metavar="def_file", type=str,
                        help="JSON definition file of the network.")

    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Pickled dictionary of parameters.")

    parser.add_argument("train_params",
                        metavar="train_params", type=str,
                        help="JSON file of training parameters.")

#    parser.add_argument("output_directory",
#                        metavar="output_directory", type=str,
#                        help="Directory to save output posteriors.")

    main(parser.parse_args())
