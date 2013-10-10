#!/usr/bin/env python
"""

Sample Call:
ipython
"""

import argparse
from multiprocessing import Pool
import numpy as np
import theano
import theano.tensor as T
import time
from scipy.spatial.distance import cdist, euclidean


# Global dict
class Globals():
    def __init__(self):
        self.k_centroids = 0
        self.feature_dim = None
        self.num_samples = None
        self.centroids = np.zeros([])
        self.assignment_buffer = []
        self.data = None

    def clear(self):
        self.assignment_buffer = []

    def load_data(self, input_file):
        self.data = np.load(input_file).astype('float32')
        self.data -= self.data.mean(axis=0)
        self.data /= self.data.std(axis=0)
        self.num_samples = self.data.shape[0]
        self.feature_dim = self.data.shape[1]

    def init_centroids(self):
        self.centroids = np.random.normal(
            size=[self.k_centroids, self.feature_dim])
        self.norm_centroids()
        self.initial_centroids = self.centroids.copy()

    def norm_centroids(self):
        u = cdist(self.centroids, np.zeros([1, self.feature_dim]), 'euclidean')
        self.centroids /= u

glowballz = Globals()


def compile_theano_dot():
    x_in = T.matrix(name="x")
    D_mat = T.matrix(name="D")
    return theano.function(inputs=[x_in, D_mat],
                           outputs=T.dot(x_in, D_mat.T),
                           allow_input_downcast=True)

tdot = compile_theano_dot()


def data_stepper(data, step_size=1000):
    M = int(np.ceil(len(data) / float(step_size)))
    for m in xrange(M):
        yield data[m * step_size : (m + 1) * step_size]

def update_centroids():
    step_size = 1000
    new_centroids = glowballz.centroids.copy()
    print "[%s] Starting cluster assignment." % (time.asctime())
    for x_m in data_stepper(glowballz.data, step_size):
        S = tdot(x_m, glowballz.centroids)
        centroid_index = S.argmax(axis=1)
        S_ij = S[np.arange(len(S)), centroid_index]
        new_centroids[centroid_index, :] += S_ij[:, np.newaxis] * x_m

    glowballz.centroids = new_centroids
    glowballz.norm_centroids()
    print "[%s] Finished update." % time.asctime()



def main(args):
    """Main routine for staging parallelization."""

    glowballz.k_centroids = args.k_centroids
    glowballz.load_data(args.input_file)
    glowballz.init_centroids()

    for n in xrange(args.max_iterations):
        update_centroids()

    print [euclidean(u, v) for u, v in zip(glowballz.centroids, glowballz.initial_centroids)]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Numpy file of input data, shaped (samples, features).")
    parser.add_argument("k_centroids",
                        metavar="k_centroids", type=int,
                        help="Number of centroids to compute.")
    parser.add_argument("max_iterations",
                        metavar="max_iterations", type=int,
                        help="Maximum number of iterations before quitting.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File to save the result.")
    main(parser.parse_args())
