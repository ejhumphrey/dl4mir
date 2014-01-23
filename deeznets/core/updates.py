"""
"""

import theano.tensor as T
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
    pass


class SGD(OrderedDict):

    PREFIX = 'learning_rate:'

    def __init__(self, params, scalar_loss):
        OrderedDict.__init__(self)
        self._inputs = dict()
        # Differentiate wrt those the given parameters
        if scalar_loss == 0:
            return

        param_list = params.values()
        gparams = [T.grad(scalar_loss, p) for p in param_list]
        for param, gparam in zip(param_list, gparams):
            eta = T.scalar(name="%s%s" % (self.PREFIX, param.name))
            self[param] = param - eta * gparam
            self._inputs[eta.name] = eta

    @property
    def inputs(self):
        return self._inputs


def UnitL2NormTensor4(tensor4, **ignored_args):
    """Each tensor3 along the first dimension is scaled to unit norm.
    """
    scalar = T.pow(T.pow(tensor4.flatten(2), 2.0).sum(axis=1), 0.5)
    return tensor4 / scalar.dimshuffle(0, 'x', 'x', 'x')


def UnitL2NormMatrix(weight_matrix, **ignored_args):
    """
    It is assumed that the first dimension of the weight_matrix is the input,
    and second is the output. The contributions of the input are supposed to be
    bounded, so the axes are transposed compared to the normal case, i.e.
    L2-norm the columns.
    """
    scalar = T.pow(T.pow(weight_matrix, 2.0).sum(axis=0), 0.5)
    return weight_matrix / scalar.dimshuffle('x', 0)

'''
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
            "%s already has an associated constraint, consider merging."
                     % param

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
'''
