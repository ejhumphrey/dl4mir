#!/usr/bin/env python
"""Consumes a filelist of cqt-arrays, producing posteriorgrams.

Importantly, a two additional files are assumed to exist in the same directory
as the provided parameters:
    1. A "*.definition" file.
    2. A "*-train_params.txt" file.

Sample Call:
MODELDIR=/media/attic/chords/models/majmin_chord_classifier_000-circ
ipython ejhumphrey/scripts/predict_chords.py \
/media/attic/chords/cqt_list.txt \
$MODELDIR/majmin_chord_classifier_000-circ_0050000-20131001_063439m662.params \
/media/attic/chords/posteriors \
"""

import argparse
import glob
import numpy as np
import os
import time

from marl.hewey.core import context_slice

from ejhumphrey.dnn.core.graphs import Network
from ejhumphrey.dnn import utils


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
    if len(batch):
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


def collect_model_files(param_file):
    def_files = glob.glob(os.path.join(os.path.split(param_file)[0],
                                       "*.definition"))
    assert len(def_files) == 1, \
        "More than one definition file found? %s" % def_files
    config_params = glob.glob(os.path.join(os.path.split(param_file)[0],
                                         "*.config"))
    assert len(config_params) == 1, \
        "More than one definition file found? %s" % config_params
    train_params = utils.json_load(config_params[0]).get("train_params")
    return def_files[0], train_params

def main(args):

    def_file, train_params = collect_model_files(args.param_file)

    dnet = Network.load(def_file, args.param_file)
    param_base = os.path.split(os.path.splitext(args.param_file)[0])[-1]

    output_dir = os.path.join(args.output_directory, param_base)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print "[%s] Output Directory: %s" % (time.asctime(), output_dir)
    for cqt_file in open(args.filelist):
        cqt_file = cqt_file.strip("\n")
        output_file = os.path.join(output_dir, os.path.split(cqt_file)[-1])
        if os.path.exists(output_file) and not args.overwrite:
            print "[%s] Skipping: %s" % (time.asctime(),
                                         os.path.split(output_file)[-1])
            continue
        posterior = predict_cqt(np.load(cqt_file), dnet, train_params)
        np.save(output_file, posterior)
        print "[%s] Finished: %s" % (time.asctime(),
                                     os.path.split(output_file)[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Pickled dictionary of parameters.")

    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output posteriors.")

    parser.add_argument("--overwrite",
                        action='store_true',
                        default=False, dest='overwrite',
                        help="Overwrite if output file exists.")

    main(parser.parse_args())
