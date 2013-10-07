'''
Created on Oct 7, 2013

@author: ejhumphrey
'''

import numpy as np
import theano
import theano.tensor as T
from ejhumphrey.dnn.core import functions
from collections import OrderedDict
from ejhumphrey.dnn.core.functions import euclidean_proj, l2norm

def update_function(x_in, W, eta):
    dist_mat = euclidean_proj(x_in, W)
    result, _updates = theano.scan(fn=set_vector_at_index,
                              outputs_info=None,
                              sequences=[dist_mat.argmin(axis=0), x_in],
                              non_sequences=W)
    updates = OrderedDict()
    updates[W] = l2norm(W + eta * T.mean(result, axis=0))
    return theano.function(inputs=[x_in, eta],
                           outputs=T.mean(dist_mat.min(axis=0)),
                           updates=updates,
                           allow_input_downcast=True)

def constraint_function(W):
    updates = OrderedDict()
    updates[W] = l2norm(W)
    return theano.function(inputs=[], outputs=None, updates=updates)

def set_vector_at_index(idx, vector, matrix_model):
    zeros = T.zeros_like(matrix_model)
    zeros_subtensor = zeros[idx, :]
    return T.set_subtensor(zeros_subtensor, vector)

class KMeans(object):

    def __init__(self, k, spherical=True, distance_fx=functions.euclidean):
        self._k = k
        self._spherical = spherical
        self._x_input = T.matrix(name='input', dtype='float32')
        self._distance_fx = distance_fx


    def fit(self, X, n_iter, eta):
        self._feature_dim = X.shape[1]
        init_vals = np.random.normal(
            loc=0.0, scale=1.0, size=(self._k, self._feature_dim))
        self._W = theano.shared(value=init_vals, name='dictionary')
        self._eta = T.scalar(name="eta", dtype=None)







