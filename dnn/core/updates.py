"""
"""

import os
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
    hyperparams = OrderedDict()
    updates = OrderedDict()
    for k in gparams.keys():
        param, gparam = params.get(k), gparams.get(k)
        eta_name = os.path.join(os.path.split(k)[0], "learning_rate")
        eta = T.scalar(name=eta_name, dtype=FLOATX)
        updates[param] = param - eta * gparam
        hyperparams[eta_name]

    return updates, hyperparams

