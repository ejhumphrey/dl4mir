#!/usr/bin/env python
"""

Sample Call:
ipython
"""

import argparse
import numpy as np
import theano
import theano.tensor as T
import time
from scipy.spatial.distance import cdist
import cPickle


def data_stepper(data, step_size=1000):
    M = int(np.ceil(len(data) / float(step_size)))
    for m in xrange(M):
        yield data[m * step_size : (m + 1) * step_size]

class SphereicalKMeans(object):
    def __init__(self, k_centroids, standardize=True, whiten=False, step_size=1000):
        self.k_centroids = k_centroids
        self.standardize = standardize
        self.whiten = whiten
        self._step_size = step_size

        self.feature_dim = None
        self.num_samples = None
        self.centroids = np.zeros([])
        self.data = None
        self._compile_theano_dot()

    def _init_centroids(self):
        self.centroids = np.random.normal(
            size=[self.k_centroids, self.feature_dim])
        self._norm_centroids()
        self.initial_centroids = self.centroids.copy()
        self.percent_change = np.inf

    def _norm_centroids(self):
        u = cdist(self.centroids, np.zeros([1, self.feature_dim]), 'euclidean')
        self.centroids /= u

    def _compile_theano_dot(self):
        x_in = T.matrix(name="x")
        D_mat = T.matrix(name="D")
        self.dot = theano.function(inputs=[x_in, D_mat],
                               outputs=T.dot(x_in, D_mat.T),
                               allow_input_downcast=True)

    def update_centroids(self, data, verbose=True):
        new_centroids = self.centroids.copy()
        if verbose:
            print "[%s] Starting cluster assignment." % (time.asctime())

        assignment = []
        for x_m in data_stepper(data, self._step_size):
            S = self.dot(x_m, self.centroids)
            centroid_index = S.argmax(axis=1)
            assignment.extend(centroid_index.tolist())
            S_ij = S[np.arange(len(S)), centroid_index]
            new_centroids[centroid_index, :] += S_ij[:, np.newaxis] * x_m

        assignment = np.array(assignment)
        self.percent_change = 100 * np.mean(self.previous_assignment != assignment)
        self.previous_assignment = assignment
        self.centroids = new_centroids
        self._norm_centroids()
        if verbose:
            print "[%s] Finished update, %0.3f%% changed." % \
                (time.asctime(), self.percent_change)

    def fit(self, data, stopping_diff=0.00, max_epochs=1000, verbose=False):
        if self.standardize:
            self.mu = data.mean(axis=0)[np.newaxis, :]
            self.sigma = data.std(axis=0)[np.newaxis, :]
            data -= self.mu
            data /= self.sigma

        if self.whiten:
            raise NotImplementedError("ZCA whitening doesn't exist yet.")

        self.num_samples = data.shape[0]
        self.feature_dim = data.shape[1]

        self._init_centroids()
        self.previous_assignment = np.zeros(self.num_samples) - 1

        epoch = 0
        while epoch < max_epochs and self.percent_change >= stopping_diff:
            self.update_centroids(data, verbose=verbose)



def main(args):
    """Main routine for staging parallelization."""

    kmeans = SphereicalKMeans(args.k_centroids, standardize=args.standardize)
    data = np.load(args.input_file).astype('float32')
    print args
    try:
        kmeans.fit(data,
                   stopping_diff=args.stopping_diff,
                   max_epochs=args.max_epochs,
                   verbose=True)
    except KeyboardInterrupt:
        print "Stopping early."

    fh = open(args.output_file, 'w')
    cPickle.dump({'mu':kmeans.mu,
                  'sig':kmeans.sigma,
                  'weights':kmeans.centroids}, fh)
    fh.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_file",
                       metavar="input_file", type=str,
                      help="Numpy file of input data, shaped (samples, features).")
    parser.add_argument("k_centroids",
                        metavar="k_centroids", type=int,
                        help="Number of centroids to compute.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File to save the result.")
    parser.add_argument("--max_epochs", action="store", dest="max_epochs",
                        type=int, default=10000,
                        help="Maximum number of epochs before stopping.")
    parser.add_argument("--stopping_diff", action="store", dest="stopping_diff",
                        type=float, default=0.01,
                        help="Maximum number of epochs before stopping.")
    parser.add_argument("--standardize", action="store", dest="standardize",
                default=True, help="Standardize data.")
#    parser.add_argument("--whiten", action="store", dest="whiten",
#                default=False, help="Perform ZCA whitening.")
    main(parser.parse_args())
