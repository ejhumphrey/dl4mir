"""
"""


import json
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample

from ejhumphrey.dnn.core import FLOATX
from ejhumphrey.dnn.core import functions
from ejhumphrey.dnn import urls


def NodeFactory(args):
    """Node factory; uses 'type' in the node_args dictionary."""
    local_args = dict(args)
    node_type = local_args.pop(Node.TYPE)
    return eval("%s(**local_args)" % node_type)


# Argument Classes no longer exist because that was dumb.
'''
class MultiSoftmaxArgs(AffineArgs):
    """
    """
    def __init__(self, name,
                 input_shape,
                 output_shape,
                 activation='linear'):
        """
        Parameters
        ----------
        name : str
            Identifier for this layer.
        input_shape : tuple
            Input shape, flattened to 1D.
        output_shape : tuple
            (n_softmaxes, output_dim)
        """
        n_in = np.prod(input_shape, dtype=int)
        assert len(output_shape) == 2
        weight_shape = (output_shape[0], n_in, output_shape[1])
        NodeArgs.__init__(self, name=name,
                          input_shape=(n_in,),
                          param_shapes=dict(weights=weight_shape,
                                            bias=output_shape),
                          activation=activation)
        self.update(input_shape=self.input_shape,
                    output_shape=self.output_shape)

    @property
    def output_shape(self):
        return (self.weight_shape[0], self.weight_shape[2])

    @property
    def weight_shape(self):
        return self.param_shapes.get("weights")


class RBFArgs(AffineArgs):
    """
    """
    def __init__(self, name,
                 input_dim,
                 output_dim,
                 lp_norm='l1',
                 activation='linear'):
        """
        """
        AffineArgs.__init__(self, name,
                            input_shape=(input_dim,),
                            output_shape=(output_dim,),
                            activation=activation)
        del self['param_shapes']['bias']
        self.update(lp_norm=lp_norm)
'''


# --- Node Implementations ------
class Node(dict):
    """
    Nodes in the graph perform parameter management and micro-math operations.

    Note that a "Node" is merely a base class, and cannot be instantiated.
    Or, empty dictionaries? Which makes more sense...? Loud failure of a class
    that shouldn't be used will certainly prevent misuse...

    MUST implement three property getters:
        self._input_shapes
        self._output_shapes
        self._param_shapes

    Param name defines should come from here, not the args class.
    Move arg classes to separate module, they exist for conveience of creating
    properly formatted dictionaries.
    """
    _NAME = 'name'
    _ACTIVATION = 'activation'
    _TYPE = 'type'

    def __init__(self, name, activation):
        """
        node_args: dict
            Needs input validation.
        """
        # Serialize inputs via update.
        self.update(name=name, activation=activation, type=self.type)

        self.numpy_rng = np.random.RandomState()
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        # Keys are relative ('x_input', 'weights', etc)
        self._inputs = dict([(k, None) for k in self._input_shapes.keys()])
        self._outputs = dict([(k, None) for k in self._output_shapes.keys()])
        self._params = dict([(k, None) for k in self._param_shapes.keys()])

        # TODO(ejhumphrey): Make the dropout variable more agnostic, if this
        #     is the way things are going to proceed.
        dropout = T.scalar(name=self._own("dropout"), dtype=FLOATX)
        self._scalars = dict(dropout=dropout)

    def __str__(self):
        return json.dumps(self, indent=4)

    # --- Private Properties ---
    @property
    def _input_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        raise NotImplementedError("Write me!")

    @property
    def _output_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        raise NotImplementedError("Write me!")

    @property
    def _param_shapes(self):
        raise NotImplementedError("Write me!")

    @property
    def _activation(self):
        return self.get(self._ACTIVATION)

    # --- Public Properties ---
    @property
    def type(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.get(self._NAME)

    # @name.setter
    # def name(self, value):
    #     rename all the things

    @property
    def input_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return dict([(self._own(k), v)
                     for k, v in self._input_shapes.items()])

    @property
    def inputs(self):
        """Return the URLs of all inputs to this node."""
        return self.input_shapes.keys()

    @property
    def output_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return dict([(self._own(k), v)
                     for k, v in self._output_shapes.items()])

    @property
    def outputs(self):
        """Return the URLs of all outputs of this node."""
        return self.output_shapes.keys()

    @property
    def param_shapes(self):
        return dict([(self._own(k), v)
                     for k, v in self._param_shapes.items()])

    @property
    def params(self):
        return dict([(self._own(k), v) for k, v in self._params.items()])

    @property
    def param_values(self):
        """
        The numeric parameters of the layer.

        Returns
        -------
        values : dict
            np.ndarray values of the parameters, keyed by full-name.

        """
        return dict([(k, v.get_value()) for k, v in self.params.iteritems()])

    @param_values.setter
    def param_values(self, param_values):
        """
        Parameters
        ----------
        param_values : dict
            key/value pairs of parameter url and np.ndarray

        """
        for url, value in param_values.items():
            network, node, param = urls.parse(url)
            # Bypass all values that do not correspond to this layer.
            if self.name != node:
                continue
            if not param in self._params:
                # Catch undeclared parameters.
                raise ValueError("Undeclared parameter: %s" % param)
            elif self._params[param] is None:
                # Declared but uninitialized; safe to do so now.
                self._params[param] = theano.shared(
                    value=value.astype(FLOATX), name=url)
            else:
                # Initialized, but changing value.
                self._params[param].set_value(value.astype(FLOATX))

    @property
    def scalars(self):
        return dict([(self._own(k), v) for k, v in self._scalars.items()])

    @property
    def activation(self):
        return functions.Activations.get(self.get(self._ACTIVATION))

    # TODO(ejhumphrey): Is this deprecated / the best way to do this?
    @property
    def dropout(self):
        """
        Used as a probability.
        """
        return self._scalars.get("dropout")

    def _own(self, key):
        url = key
        if not self.name in url:
            url = urls.append_param(self.name, key)
        return url

    def validate_inputs(self, inputs):
        """Determine if the inputs dictionary contains the necessary keys."""
        # TODO(ejhumphrey): Look into this method. It just failed rather
        # opaquely, and I'm not sure why.
        valid_inputs = []
        for url in inputs:
            network, node, param = urls.parse(url)
            valid_inputs.append(urls.append_param(node, param) in self.inputs)
        return not False in valid_inputs

    def filter_inputs(self, inputs):
        """Extract the relevant input items, iff all are present."""
        assert self.validate_inputs(inputs)
        return dict([(k, inputs.pop(k)) for k in self.inputs])

    def transform(self):
        """
        """
        raise NotImplementedError("Subclass me!")


class Affine(Node):
    """
    Affine Transform Layer
      (i.e., a fully-connected non-linear projection)

    """
    _N_IN = 'n_in'
    _N_OUT = 'n_out'
    _INPUT = 'input'
    _OUTPUT = 'output'
    _WEIGHTS = 'weights'
    _BIAS = 'bias'

    def __init__(self, name, n_in, n_out, activation):
        """

        """
        # Preamble before calling super-init
        self.update(dict(n_in=n_in, n_out=n_out))
        Node.__init__(self, name, activation)
        weight_shape = self._param_shapes[self._WEIGHTS]
        weight_values = self.numpy_rng.normal(
            loc=0.0,
            scale=np.sqrt(1.0 / np.sum(weight_shape)),
            size=weight_shape)
        bias_values = np.zeros(n_out)
        self.param_values = {self._own(self._WEIGHTS): weight_values,
                             self._own(self._BIAS): bias_values, }

    # --- Over-ridden private properties ---
    @property
    def _input_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return {self._INPUT: (self.get(self._N_IN), )}

    @property
    def _output_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return {self._OUTPUT: (self.get(self._N_OUT), )}

    @property
    def _param_shapes(self):
        return {self._WEIGHTS: (self.get(self._N_IN), self.get(self._N_OUT)),
                self._BIAS: (self.get(self._N_OUT), )}

    def transform(self, inputs):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))

        Parameters
        ----------
        inputs: dict
            Must contain all known data inputs to this node, keyed by full
            URLs. Will fail loudly otherwise.

        Returns
        -------
        outputs: dict
            Will contain all outputs generated by this node, keyed by full
            name. Note that the symbolic outputs will take this full name
            internal to each object.
        """
        assert self.validate_inputs(inputs)
        # Since there's only the one, just take the single item.
        x_in = inputs.get(self.inputs[0])
        weights = self._params[self._WEIGHTS]
        bias = self._params[self._BIAS].dimshuffle('x', 0)

        # TODO(ejhumphrey): This isn't very stable, is it.
        x_in = T.flatten(x_in, outdim=2)
        z_out = self.activation(T.dot(x_in, weights) + bias)

        output_shape = self._output_shapes[self._OUTPUT]
        selector = self.theano_rng.binomial(size=output_shape,
                                            p=1.0 - self.dropout,
                                            dtype=FLOATX)
        z_out *= selector.dimshuffle('x', 0) * (self.dropout + 0.5)
        z_out.name = self.outputs[0]
        return {z_out.name: z_out}


class Conv3D(Node):
    """ (>^.^<) """
    # Reserved Keys
    _INPUT = 'input'
    _OUTPUT = 'output'
    _WEIGHTS = 'weights'
    _BIAS = 'bias'

    # Arguments
    _INPUT_SHAPE = 'input_shape'
    _WEIGHT_SHAPE = 'weight_shape'
    _POOL_SHAPE = 'pool_shape'
    _DOWNSAMPLE_SHAPE = 'downsample_shape'
    _BORDER_MODE = 'border_mode'

    def __init__(self, name,
                 input_shape,
                 weight_shape,
                 pool_shape=(1, 1),
                 downsample_shape=(1, 1),
                 activation='relu',
                 border_mode='valid'):
        """
        Parameters
        ----------
        input_shape : tuple
            Shape of the input data, as (in_maps, in_dim0, in_dim1).
        weight_shape : tuple
            Shape for all kernels, as (num_kernels, w_dim0, w_dim1).
        pool_shape : tuple
            2D tuple to pool over each feature map, as (p_dim0, p_dim1).
        downsample_shape : tuple
            2D tuple for downsampling each feature map, as (p_dim0, p_dim1).
        activation : str
            Name of the activation function to use.
        border_mode : str
            Convolution method for dealing with the edge of a feature map.

        """
        # Make sure the weight_shape argument is formatted properly.
        w = list(weight_shape)
        if len(w) == 3:
            w.insert(1, input_shape[0])
        elif len(w) == 4:
            assert w[1] == input_shape[0], \
                "weight_shape[1] must align with input_shape[0]: " \
                "%d!=%d." % (w[1], input_shape[0])
        else:
            raise ValueError("'weight_shape' must be length 3 or 4.")
        weight_shape = tuple(w)

        self.update(
            input_shape=input_shape,
            weight_shape=weight_shape,
            pool_shape=pool_shape,
            downsample_shape=downsample_shape,
            border_mode=border_mode)

        Node.__init__(self, name=name, activation=activation)

        # Create all the weight values at once
        fan_in = np.prod(weight_shape[1:])
        weight_values = self.numpy_rng.normal(
            loc=0.0, scale=np.sqrt(3. / fan_in), size=weight_shape)

        if self._activation == 'sigmoid':
            weight_values *= 4

        bias_values = np.zeros(weight_shape[:1])
        self.param_values = {self._own(self._WEIGHTS): weight_values,
                             self._own(self._BIAS): bias_values, }

    @property
    def _input_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return {self._INPUT: self.get(self._INPUT_SHAPE)}

    @property
    def _output_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        d0_in, d1_in = self._input_shapes.values()[0][-2:]
        if self._border_mode == 'valid':
            d0_out = int(d0_in - self._weight_shape[-2] + 1)
            d0_out /= self._pool_shape[0]
            d1_out = int(d1_in - self._weight_shape[-1] + 1)
            d1_out /= self._pool_shape[1]
        elif self._border_mode == 'same':
            d0_out, d1_out = d0_in, d1_in
        elif self._border_mode == 'full':
            raise NotImplementedError("Being lazy, haven't done this yet.")
        output_shape = (self._weight_shape[0], d0_out, d1_out)
        return {self._OUTPUT: output_shape}

    @property
    def _param_shapes(self):
        return {self._WEIGHTS: self._weight_shape,
                self._BIAS: self._weight_shape[:1]}

    @property
    def _weight_shape(self):
        return self.get(self._WEIGHT_SHAPE)

    @property
    def _border_mode(self):
        return self.get(self._BORDER_MODE)

    @property
    def _pool_shape(self):
        return self.get(self._POOL_SHAPE)

    @property
    def _downsample_shape(self):
        """
        Returns
        -------
        shapes : dict
        """
        return self.get(self._DOWNSAMPLE_SHAPE)

    def transform(self, inputs):
        """

        """
        self.validate_inputs(inputs)
        x_in = inputs.get(self.inputs[0])
        weights = self._params[self._WEIGHTS]
        bias = self._params[self._BIAS].dimshuffle('x', 0, 'x', 'x')

        z_out = T.nnet.conv.conv2d(
            input=x_in,
            filters=weights,
            filter_shape=self._weight_shape,
            border_mode=self._border_mode)

        selector = self.theano_rng.binomial(
            size=self._output_shapes[self._OUTPUT][:1],
            p=1.0 - self.dropout,
            dtype=FLOATX)

        z_out = self.activation(z_out + bias)
        z_out *= selector.dimshuffle('x', 0, 'x', 'x') * (self.dropout + 0.5)
        z_out = downsample.max_pool_2d(
            z_out, self._pool_shape, ignore_border=False)
        z_out.name = self.outputs[0]
        return {z_out.name: z_out}


class Conv2D(Node):
    """ . """

    def __init__(self, layer_args):
        """
        layer_args : ConvArgs

        """
        Node.__init__(self, layer_args)

        # Create all the weight values at once
        weight_shape = self.param_shapes.get("weights")
        fan_in = np.prod(weight_shape[1:])
        weights = self.numpy_rng.normal(loc=0.0,
                                        scale=np.sqrt(3. / fan_in),
                                        size=weight_shape)

        if self.get("activation") == 'sigmoid':
            weights *= 4

        bias = np.zeros(weight_shape[0])
        self.param_values = {self.own('weights'): weights,
                             self.own('bias'): bias, }

    def transform(self, x_in):
        """

        """
        W = self._theta['weights']
        b = self._theta['bias']
        weight_shape = self.param_shapes.get("weights")
        z_out = T.nnet.conv.conv2d(input=x_in,
                                   filters=W,
                                   filter_shape=weight_shape,
                                   border_mode=self.get("border_mode"))

        selector = self.theano_rng.binomial(size=self.output_shape[:1],
                                            p=1.0 - self.dropout,
                                            dtype=FLOATX)

        z_out = self.activation(z_out + b.dimshuffle('x', 0, 'x', 'x'))
        z_out *= selector.dimshuffle('x', 0, 'x', 'x') * (self.dropout + 0.5)
        return downsample.max_pool_2d(
            z_out, self.get("pool_shape"), ignore_border=False)


class Softmax(Affine):
    """
    """

    def __init__(self, name, n_in, n_out, activation):
        """
        """
        Affine.__init__(self, name, n_in, n_out, activation)
        # Softmax layers don't typically have dropout.
        self._scalars.clear()

    def transform(self, inputs):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))
        """
        assert self.validate_inputs(inputs)
        # Since there's only the one, just take the single item.
        x_in = inputs.get(self.inputs[0])
        weights = self._params[self._WEIGHTS]
        bias = self._params[self._BIAS].dimshuffle('x', 0)

        # TODO(ejhumphrey): This isn't very stable, is it.
        x_in = T.flatten(x_in, outdim=2)
        z_out = T.nnet.softmax(self.activation(T.dot(x_in, weights) + bias))
        z_out.name = self.outputs[0]
        return {z_out.name: z_out}


class RBF(Node):
    """
    Radial Basis Function Layer
      (i.e. distance layer)

    """
    param_names = ["weights"]

    def __init__(self, layer_args):
        """
        layer_args : RBFArgs

        """
        Node.__init__(self, layer_args)
        w_shape = self.param_shapes.get("weights")
        weights = self.numpy_rng.normal(loc=0.0,
                                        scale=np.sqrt(1. / np.sum(w_shape)),
                                        size=w_shape)
        self.param_values = {self.own('weights'): weights, }

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))

        """
        W = self._theta["weights"].T

        # TODO(ejhumphrey): This isn't very stable, is it.
        x_in = T.flatten(x_in, outdim=2)
        if self.get("lp_norm") == "l1":
            z_out = T.abs_(x_in.dimshuffle(0, 'x', 1) - W.dimshuffle('x', 0, 1))
        elif self.get("lp_norm") == "l2":
            z_out = T.pow(x_in.dimshuffle(0, 'x', 1) - W.dimshuffle('x', 0, 1), 2.0)
        else:
            raise NotImplementedError(
                "Lp_norm type '%s' unsupported." % self.get("lp_norm"))

        selector = self.theano_rng.binomial(size=self.output_shape,
                                            p=1.0 - self.dropout,
                                            dtype=FLOATX)
        return T.sum(z_out, axis=2) * selector.dimshuffle('x', 0) * (self.dropout + 0.5)


class SoftMask(Node):
    """
    """
    param_names = ["weights", "bias", "templates"]

    def __init__(self, layer_args):
        """
        """
        Node.__init__(self, layer_args)
        weight_shape = self.param_shapes.get("weights")
#        scale = np.sqrt(6. / np.sum(weight_shape))

#        weights = self.numpy_rng.normal(loc=1.0,
#                                        scale=0.05,
#                                        size=weight_shape)
        weights = self.numpy_rng.uniform(low=0, high=1.0, size=weight_shape)
        templates = np.ones(weights.shape)
        bias = np.zeros(self.output_shape)

        self.param_values = {self.own('weights'): weights,
                             self.own('bias'): bias,
                             self.own('templates'): templates, }
        self._scalars.clear()

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))
        """
        # TODO(ejhumphrey): This isn't very stable, is it.
        x_in = x_in.flatten(2)
        W = self._theta["weights"] * self._theta["templates"]
        b = self._theta["bias"].dimshuffle('x', 0)
        return T.nnet.softmax(self.activation(T.dot(x_in, W) + b))


class MultiSoftmax(Node):
    """
    Multi-softmax Layer


    """
    param_names = ["weights", "bias"]

    def __init__(self, layer_args):
        """
        layer_args : AffineArgs

        """
        Node.__init__(self, layer_args)
        w_shape = self.param_shapes.get("weights")
        weights = self.numpy_rng.normal(loc=0.0,
                                        scale=np.sqrt(1. / np.sum(w_shape)),
                                        size=w_shape)
        bias = np.zeros(self.output_shape)
        self.param_values = {self.own('weights'): weights,
                             self.own('bias'): bias, }
        self._scalars.clear()

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))

        """
        W = self._theta["weights"]
        b = self._theta['bias']

        x_in = T.flatten(x_in, outdim=2)
        output = []
        for i in range(self.output_shape[0]):
            z_i = self.activation(T.dot(x_in, W[i]) + b[i].dimshuffle('x', 0))
            output.append(T.nnet.softmax(z_i).dimshuffle(0, 1, 'x'))

        return T.concatenate(output, axis=2)


class EnergyPDF(Node):
    """
    """
    param_names = []

    def __init__(self, layer_args):
        """
        """
        Node.__init__(self, layer_args)
        self.param_values = {}
        self._scalars.clear()

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))
        """
        return T.nnet.softmax(-1.0 * x_in)
