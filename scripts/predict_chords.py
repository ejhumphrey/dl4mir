#!/usr/bin/env python
"""Train a deepnet for chord identification.

Sample Call:
bash$ ipython ejhumphrey/scripts/predict_chords.py \

"/Volumes/Audio/Chord_Recognition/labeled_tracks_20130926.txt"

"""


import argparse
import numpy as np

from ejhumphrey.datasets import chordutils
from marl.hewey.core import DataSequence, context_slice
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen

from ejhumphrey.dnn.core.graphs import load

import json
import os
import time

cqt_param_file = "/Volumes/Audio/Chord_Recognition/cqt_params.txt"
track_filelist = "/Volumes/Audio/Chord_Recognition/labeled_tracks_20130926.txt"
hewey_file = '/Volumes/speedy/chordrec.dsf'


def collect_track_tuples(filelist):
    """Compile a list of (audio_file, cqt_file, label_file) tuples.

    It is guaranteed that all files exist on disk.

    Returns
    -------
    results : list
        List of three item tuples; audio, cqt, and label.
    """
    results = []
    for line in open(filelist):
        audio_file = line.strip('\n')
        assert os.path.exists(audio_file), \
            "Audio file does not exist: %s" % audio_file
        cqt_file = os.path.splitext(audio_file)[0] + "-cqt.npy"
        assert os.path.exists(cqt_file), \
            "CQT file does not exist: %s" % cqt_file
        label_file = os.path.splitext(audio_file)[0] + ".lab"
        assert os.path.exists(label_file), \
            "Label file does not exist: %s" % label_file
        results.append((audio_file, cqt_file, label_file))

    return results


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

def predict_cqt(cqt_array, dnet, train_params):
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
                         newshape=dnet.input_shape)
    posterior = list()
    inputs = dnet.empty_inputs()
    for batch in gen:
        inputs[dnet.input_name] = batch
        posterior.append(dnet(inputs))
    return np.concatenate(posterior, axis=0)


def main(args):
    dnet = load(args.def_file, args.param_file)
    model_name = os.path.split(os.path.splitext(args.def_file)[0])[-1]
    train_params = json.load(open(args.train_params))
    for cqt_file in open(args.file_list):
        posterior = predict_cqt(cqt_file, dnet, train_params)
        output_file = "%s-%s_pred.npy" % (cqt_file.strip("-cqt.npy"),
                                          model_name)
        np.save(output_file, posterior)


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

    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output posteriors.")

    main(parser.parse_args())
