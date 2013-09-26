"""
"""

import os
import theano
import theano.tensor as T
from . import FLOATX
from collections import OrderedDict


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
