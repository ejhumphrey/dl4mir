"""
"""
import json
import theano.tensor as T

# from ejhumphrey.dnn.core import FLOATX


def LossFactory(args):
    """Uses 'type' in the node_args dictionary to build a general loss."""
    local_args = dict(args)
    loss_type = local_args.pop(Loss.TYPE)
    return eval("%s(**local_args)" % loss_type)


class Loss(dict):
    TYPE = 'type'
    INPUTS = 'inputs'
    GIVENS = 'givens'

    def __init__(self, inputs, givens):
        self.inputs = inputs
        self.givens = givens
        self.type = self.type

    def __str__(self):
        return json.dumps(self, indent=2)

    @property
    def type(self):
        return self.__class__.__name__

    @type.setter
    def type(self, value):
        self[self.TYPE] = value

    @property
    def inputs(self):
        return self.get(self.INPUTS)

    @inputs.setter
    def inputs(self, value):
        self[self.INPUTS] = value

    @property
    def givens(self):
        return self.get(self.GIVENS)

    @givens.setter
    def givens(self, value):
        self[self.GIVENS] = value

    def loss(self, inputs):
        for url in self.inputs.values():
            assert url in inputs, "Expected '$%s' in 'inputs'" % url


class NegativeLogLikelihood(Loss):
    POSTERIOR = 'posterior'
    TARGET_IDX = 'target_idx'

    def __init__(self, inputs, givens):
        # Input Validation
        assert self.POSTERIOR in inputs, \
            "Expected '%s' in 'inputs'." % self.POSTERIOR
        assert self.TARGET_IDX in givens, \
            "Expected '%s' in 'givens'." % self.TARGET_IDX

        Loss.__init__(self, inputs, givens)

    def loss(self, inputs):
        """
        inputs : dict
            Set of URL-keyed variables from which to select.
        """
        Loss.loss(self, inputs)
        # Create the local givens.
        target_name = self.givens[self.TARGET_IDX]
        target_idx = T.ivector(name=target_name)

        posterior_url = self.inputs.get(self.POSTERIOR)
        posterior = inputs.get(posterior_url)
        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        scalar_loss = T.mean(-T.log(posterior)[batch_idx, target_idx])
        return scalar_loss, {target_idx.name: target_idx}


# def mean_squared_error(name, inputs):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     INPUT_KEY = 'prediction'
#     assert INPUT_KEY in inputs, \
#         "Function expected a key named '%s' in 'inputs'." % INPUT_KEY
#     target = T.matrix(name=urls.append_param(name, 'target'))
#     raise NotImplementedError("Haven't finished this yet.")


# def l2_penalty(name, inputs):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     INPUT_KEY = 'input'
#     assert INPUT_KEY in inputs, \
#         "Function expected a key named '%s' in 'inputs'." % INPUT_KEY
#     hyperparam_name = urls.append_param(name, 'l2_penalty')
#     weight_decay = T.scalar(hyperparam_name, dtype=FLOATX)
#     scalar_loss = weight_decay * T.sum(T.pow(inputs[INPUT_KEY], 2.0))
#     return scalar_loss, {weight_decay.name: weight_decay}


# def l1_penalty(x_input):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     hyperparam_name = os.path.join(x_input.name, 'l1_penalty')
#     sparsity = T.scalar(hyperparam_name, dtype=FLOATX)
#     scalar_loss = sparsity * T.sum(T.abs_(x_input))
#     return scalar_loss, [sparsity]
