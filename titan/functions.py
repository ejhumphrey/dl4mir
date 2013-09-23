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
    raise NotImplementedError("SoftShrink is not implemented yet.")

def hard_shrink(x, threshold):
    """ Write my LaTeX form."""
    raise NotImplementedError("SoftShrink is not implemented yet.")

Activations = {'linear':linear,
               'relu':relu,
               'tanh':tanh,
               'sigmoid':sigmoid,
               'soft_shrink':soft_shrink,
               'hard_shrink':hard_shrink}
