"""Fit a PCA-LDA model to a training stash file."""

from __future__ import print_function
import argparse
import biggie
import numpy as np
import optimus
from os import path
import time
from sklearn.lda import LDA
from sklearn.decomposition import PCA

import dl4mir.timbre.data as D
import dl4mir.common.fileutil as futil

PRINT_FREQ = 500


def fit_params(data, labels, n_components=256, n_out=3):
    data = np.log1p(np.asarray(data))
    shp = data.shape
    data = data.reshape(shp[0], np.prod(shp[1:]))
    pca = PCA(n_components=n_components, whiten=True).fit(data)
    lda = LDA(n_components=n_out).fit(pca.transform(data), labels)
    return {
        'pca.bias': pca.mean_,
        'pca.weights': pca.components_.T,
        'lda.bias': lda.xbar_,
        'lda.weights': lda.scalings_[:, :n_out]}


def pca_lda_graph(n_in=20, n_components=256, n_out=3):
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, n_in, 192))

    reshape = optimus.Flatten(name='flat', ndim=2)
    logscale = optimus.Log("logscale", 1.0)

    pca = optimus.CenteredAffine(
        name='pca',
        input_shape=(None, n_in*192),
        output_shape=(None, n_components),
        act_type='linear')

    lda = optimus.CenteredAffine(
        name='lda',
        input_shape=(None, n_components),
        output_shape=(None, n_out),
        act_type='linear')

    embedding = optimus.Output(name='embedding')

    base_edges = [
        (input_data, reshape.input),
        (reshape.output, logscale.input),
        (logscale.output, pca.input),
        (pca.output, lda.input),
        (lda.output, embedding)]

    predictor = optimus.Graph(
        name='pca-lda',
        inputs=[input_data],
        nodes=[logscale, reshape, pca, lda],
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[embedding],
        verbose=False)

    return predictor


def main(args):
    predictor = pca_lda_graph(20, args.n_components, 3)
    input_shape = list(predictor.inputs['cqt'].shape)
    time_dim = input_shape[2]
    input_shape[0] = args.num_points

    print("Opening {0}".format(args.training_file))
    stash = biggie.Stash(args.training_file, cache=True)
    stream = D.create_labeled_stream(
        stash, time_dim, working_size=1000, threshold=0.05)

    print("Starting '{0}'".format(args.trial_name))
    data, labels = np.zeros(input_shape), []
    for idx, x in enumerate(stream):
        data[idx, ...] = x.cqt
        labels.append(x.label)
        if len(labels) == args.num_points:
            break
        elif (len(labels) % PRINT_FREQ) == 0:
            print("[{0}] {1:5} / {2:5}"
                  "".format(time.asctime(), len(labels), args.num_points))

    predictor.param_values = fit_params(data, labels, args.n_components, 3)
    output_directory = futil.create_directory(args.output_directory)
    predictor_file = path.join(output_directory, args.predictor_file)
    param_file = predictor_file.replace(".json", ".npz")
    optimus.save(predictor, def_file=predictor_file, param_file=param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to a biggie Stash file for training.")
    parser.add_argument("num_points",
                        metavar="num_points", type=int,
                        help="Number to datapoints from which to fit params.")
    parser.add_argument("n_components",
                        metavar="n_components", type=int,
                        help="Dimensionality of PCA-subspace.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the .")
    parser.add_argument("trial_name",
                        metavar="trial_name", type=str,
                        help="Unique name for this training run.")
    parser.add_argument("predictor_file",
                        metavar="predictor_file", type=str,
                        help="Name for the resulting predictor graph.")
    main(parser.parse_args())
