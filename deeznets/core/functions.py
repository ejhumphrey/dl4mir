"""Pointwise theano functions and other such malarkey.

Note: Unless these are re-written as classes, which isn't the worst idea,
a dictionary is neccesary so that the activation function can be serialized as
a string. If it were a class, it could / would have a name property.
"""

import theano.tensor as T


def linear(x):
    """ Write my LaTeX form."""
    return x


def relu(x):
    """ Write my LaTeX form."""
    return 0.5 * (x + T.abs_(x))


def tanh(x):
    """ Write my LaTeX form."""
    return T.tanh(x)


def sigmoid(x):
    """ Write my LaTeX form."""
    return T.nnet.sigmoid(x)


def soft_shrink(x, threshold, Q):
    """ Write my LaTeX form."""
    raise NotImplementedError("'soft_shrink' is not implemented yet.")


def hard_shrink(x, threshold):
    """ Write my LaTeX form."""
    raise NotImplementedError("'hard_shrink' is not implemented yet.")


def soft_hinge(x, margin, knee=1.0):
    """
    x : symbolic, or scalar
        typically, the independent variable
    margin : scalar, or symbolic
        typically, the margin or offset
    knee : scalar
        knee of the log-approx

    note: standard behavior is monotonically increasing; swapping a and b
        will flip the function horizontally.
    """
    return T.log(1 + T.exp(knee * (x - margin))) / knee


Activations = {'linear': linear,
               'relu': relu,
               'tanh': tanh,
               'sigmoid': sigmoid,
               'soft_shrink': soft_shrink,
               'hard_shrink': hard_shrink}


def l2norm(x):
    scalar = T.pow(T.pow(x, 2.0).sum(axis=1), 0.5)
    return x / scalar.dimshuffle(0, 'x')


def euclidean(a, b):
    """Row-wise euclidean distance between tensors.
    """
    a, b = a.flatten(2), b.flatten(2)
    return T.sqrt(T.sum(T.pow(a - b, 2.0), axis=1))


def manhattan(a, b):
    """Row-wise manhattan distance between tensors.
    """
    a, b = a.flatten(2), b.flatten(2)
    return T.sum(T.abs_(a - b), axis=1)


def euclidean_proj(a, b):
    """Projected Euclidean distance between tensors.
    """
    a = a.flatten(2).dimshuffle("x", 0, 1)
    b = b.flatten(2).dimshuffle(0, "x", 1)
    return T.sqrt(T.sum(T.pow(a - b, 2.0), axis=-1))


def manhattan_proj(a, b):
    """Projected Manhattan distance between tensors.
    """
    a = a.flatten(2).dimshuffle("x", 0, 1)
    b = b.flatten(2).dimshuffle(0, "x", 1)
    return T.sum(T.abs_(a - b), axis=-1)
