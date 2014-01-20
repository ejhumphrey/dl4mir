"""
"""

import os
import theano
import theano.tensor as T
from . import FLOATX
from collections import OrderedDict
from ejhumphrey.dnn.core import functions


def sgd(scalar_loss, params):
    """
    Parameters
    ----------
    scalar_loss : theano symb scalar
        Scalar loss to differentate params.
    params : dict
        Full names and corresponding symbolic parameters.

    Returns
    -------
    updates : dict
        Update rules for the given parameters.
    hyperparams : dict
        Scalar hyperparameters produced as a result of this call.
    """

    # Differentiate wrt those the given parameters
    gparams = OrderedDict([(k, T.grad(scalar_loss, p)) \
                           for k, p in params.iteritems()])

    # generate the list of updates
    hyperparams = []
    updates = OrderedDict()
    for k in gparams.keys():
        param, gparam = params.get(k), gparams.get(k)
        eta_name = os.path.join(k, "learning_rate")
        eta = T.scalar(name=eta_name, dtype=FLOATX)
        updates[param] = param - eta * gparam
        hyperparams.append(eta)

    return updates, hyperparams


class Update(dict):

    def __init__(self, params, givens):
        pass


class SGD(object):

    def __init__(self):
        self._inputs = []
        self._outputs = {}
        self._updates = OrderedDict()
        self._fx = None
        self.iteration = 0

    @property
    def inputs(self):
        return list(self._inputs)

    @property
    def outputs(self):
        return dict(self._outputs)

    def compute_updates(self, loss, params):
        """
        Parameters
        ----------
        loss : instance of Loss class
            Must have the following attributes:
                'total'
                'inputs'
        params : dict
            Symbolic parameters to differentiate wrt. Must be part of the graph.
        """
        scalar_loss = loss.total
        self._inputs = []
        self._inputs.extend(loss.inputs)
        self._outputs['loss'] = scalar_loss
        # Differentiate wrt those the given parameters
        gparams = OrderedDict([(k, T.grad(scalar_loss, p)) \
                               for k, p in params.iteritems()])

        # generate the list of updates
        for k in gparams.keys():
            param, gparam = params.get(k), gparams.get(k)
            eta_name = os.path.join(k, "learning_rate")
            eta = T.scalar(name=eta_name, dtype=FLOATX)
            self._updates[param] = param - eta * gparam
            self._inputs.append(eta)


    def compile(self):
        self._fx = theano.function(inputs=self.inputs,
                                   outputs=self.outputs.get('loss'),
                                   allow_input_downcast=True,
                                   updates=self._updates,
                                   on_unused_input='warn')

    def __call__(self, inputs):
        """Dictionary of kwargs.
        """
        if self._fx is None:
            self.compile()
        self.iteration += 1
        return self._fx(**inputs)

    def empty_inputs(self, fill_value=0.0):
        return dict([(x.name, fill_value) for x in self.inputs])


def L2NormTensor4(tensor4, **ignored_args):
    """Each tensor3 along the first dimension is scaled to unit norm.
    """
    scalar = T.pow(T.pow(tensor4.flatten(2), 2.0).sum(axis=1), 0.5)
    return tensor4 / scalar.dimshuffle(0, 'x', 'x', 'x')

def L2NormMatrix(weight_matrix, **ignored_args):
    """
    It is assumed that the first dimension of the weight_matrix is the input,
    and second is the output. The contributions of the input are supposed to be
    bounded, so the axes are transposed compared to the normal case, i.e.
    L2-norm the columns.
    """
    scalar = T.pow(T.pow(weight_matrix, 2.0).sum(axis=0), 0.5)
    return weight_matrix / scalar.dimshuffle('x', 0)


RegisteredConstraints = {'l2norm-matrix': L2NormMatrix,
                         'l2norm-tensor4': L2NormTensor4}

class Constraints(object):

    def __init__(self):
        self._updates = OrderedDict()
        self._fx = None

    def register(self, param, args):
        """
        Parameters
        ----------
        param : theano.shared type
            Symbolic parameter to constrain.
        args : dict
            Arguments for the given constraint. Must contain at least a "name".
        """
        assert not param in self._updates, \
            "%s already has an associated constraint, consider merging." % param

        self._updates[param] = RegisteredConstraints.get(
            args.get("name"))(param, **args)

    def compile(self):
        self._fx = theano.function(inputs=[],
                                   outputs=None,
                                   allow_input_downcast=True,
                                   updates=self._updates)

    def apply(self):
        """"""
        if self._fx is None:
            self.compile()
        self._fx()
