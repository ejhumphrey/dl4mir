"""
"""


import json
import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample

from ejhumphrey.dnn.core import FLOATX
from ejhumphrey.dnn.core import functions


def Layer(layer_args):
    """Layer factory; uses 'type' in the layer_args dictionary."""
    return eval("%s(layer_args)" % layer_args.get("type"))

# --- Layer Argument Classes --- 
class BaseLayerArgs(dict):
    """
    Base class for all layer arguments
    """
    def __init__(self, name,
                 input_shape=None,
                 param_shapes=None,
                 activation="linear"):
        """
        Parameters
        ----------
        name : str
            Unique name for the layer.
        input_shape : tuple
            shape of input array, regardless of batch size
        activation : string
            Name of the activation function.
        """

        if param_shapes is None:
            param_shapes = dict()

        self._input_shape = input_shape

        args = {'type':self.type,
                'name':name,
                'param_shapes':param_shapes,
                'activation':activation, }

        self.update(args)

    def __str__(self):
        return json.dumps(self)

    @property
    def type(self):
        return self.__class__.__name__.split("Args")[0]

    @property
    def input_shape(self):
        """
        Returns
        -------
        shp : tuple
        """
        return self._input_shape

    @property
    def output_shape(self):
        """
        Returns
        -------
        shp : tuple
        """
        raise NotImplementedError("Output shape is undefined.")

    @property
    def activation(self):
        return self.get('activation')

    @property
    def name(self):
        return self.get('name')

    @property
    def param_shapes(self):
        return self.get('param_shapes')

    @param_shapes.setter
    def param_shapes(self, shapes):
        """
        Parameters
        ----------
        shapes : dict
        """
        self['param_shapes'].update(shapes)



class AffineArgs(BaseLayerArgs):
    """
    """

    def __init__(self, name,
                 input_shape,
                 output_shape,
                 activation="tanh"):

        """
        Parameters
        ----------

        """
        n_in = np.prod(input_shape, dtype=int)
        assert len(output_shape) == 1
        weight_shape = (n_in, output_shape[0])
        BaseLayerArgs.__init__(self, name=name,
                               input_shape=(n_in,),
                               param_shapes=dict(weights=weight_shape,
                                                 bias=output_shape),
                               activation=activation)
        self.update(input_shape=self.input_shape,
                    output_shape=self.output_shape)

    @property
    def output_shape(self):
        return self.weight_shape[1:]

    @property
    def weight_shape(self):
        return self.param_shapes.get("weights")


class Conv3DArgs(BaseLayerArgs):

    def __init__(self, name,
        input_shape,
        weight_shape,
        pool_shape=(1, 1),
        downsample_shape=(1, 1),
        activation="tanh",
        border_mode='valid'):
        """
        input_shape : tuple
            (in_maps, in_dim0, in_dim1), the last three dims of a 4d tensor
            with a typical shape (n_points, in_maps, in_dim0, in_dim1)
        weight_shape : tuple
            (out_maps, w_dim0, w_dim1)

        """
        # If input_shape is provided, must make sure the weight_shape is
        # consistent.
        if input_shape:
            w = list(weight_shape)
            if len(w) == 3:
                w.insert(1, input_shape[0])
            elif len(w) == 4:
                w[1] = input_shape[0]
            weight_shape = tuple(w)

        param_shapes = {"weights":weight_shape,
                        "bias":weight_shape[:1], }
        BaseLayerArgs.__init__(self, name,
                               input_shape=input_shape,
                               param_shapes=param_shapes,
                               activation=activation)
        self.update(pool_shape=pool_shape,
                    downsample_shape=downsample_shape,
                    border_mode=border_mode)
        self.update(input_shape=self.input_shape,
                    output_shape=self.output_shape)

    @property
    def output_shape(self):
        d0_in, d1_in = self.input_shape[1:]
        d0_out = (d0_in - self.weight_shape[-2] + 1) / self.pool_shape[0]
        d1_out = (d1_in - self.weight_shape[-1] + 1) / self.pool_shape[1]
        return (self.weight_shape[0], d0_out, d1_out)

    @property
    def pool_shape(self):
        return self.get('pool_shape')

    @property
    def downsample_shape(self):
        return self.get('downsample_shape')

    @property
    def weight_shape(self):
        return self.param_shapes.get("weights")


class SoftmaxArgs(AffineArgs):
    """
    """
    def __init__(self, name,
                 input_dim,
                 output_dim):
        """
        """
        AffineArgs.__init__(self, name,
                            input_shape=(input_dim,),
                            output_shape=(output_dim,),
                            activation="linear")

# --- Layer Class Implementations ------
class BaseLayer(dict):
    """
    Layers are in charge of parameter management and micro-math operations.
    """
    param_names = []
    def __init__(self, layer_args):
        self.update(layer_args)
        """
        Takes a LayerArg dictionary.
        """
        self.numpy_rng = np.random.RandomState()
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        # Theta is the local set of all symbolic parameters in this layer.
        self._theta = dict([(k, None) for k in self.param_names])
        self._scalars = dict(dropout=T.scalar(name=self.own("dropout"),
                                              dtype=FLOATX))

    def __str__(self):
        return json.dumps(self, indent=2)

    @property
    def scalars(self):
        return [v for v in self._scalars.itervalues()]

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self.get("name")

    @property
    def activation(self):
        return functions.Activations.get(self.get("activation"))

    @property
    def params(self):
        """
        The symbolic parameters of the layer.

        Returns
        -------
        params : dict
            Symbolic parameters of the layer, keyed by full name.
        """

        return dict([(self.own(k), v) for k, v in self._theta.iteritems()])

    @property
    def param_values(self):
        """
        The numeric parameters of the layer.

        Returns
        -------
        values : dict
            np.ndarray values of the layer, keyed by full-name.

        """
        return dict([(k, v.get_value()) for k, v in self.params.iteritems()])

    @param_values.setter
    def param_values(self, param_values):
        """
        Parameters
        ----------
        param_values : dict
            key/value pairs of parameter name and np.ndarray

        """
        for full_name, value in param_values.items():
            layer_name, param_name = os.path.split(full_name)
            # Bypass all values that do not correspond to this layer. 
            if self.name != layer_name:
                continue
            if not param_name in self._theta:
                # Catch undeclared parameters.
                raise ValueError("Undeclared parameter: %s" % param_name)
            elif self._theta[param_name] is None:
                # Declared but uninitialized; safe to do so now. 
                self._theta[param_name] = theano.shared(
                    value=value.astype(FLOATX), name=full_name)
            else:
                # Initialized, but changing value.
                self._theta[param_name].set_value(value.astype(FLOATX))

    @property
    def input_shape(self):
        """
        Returns
        -------
        shp : tuple
        """
        return self.get("input_shape")

    @property
    def output_shape(self):
        """
        Returns
        -------
        shp : tuple
        """
        return self.get("output_shape")

    @property
    def param_shapes(self):
        return self.get("param_shapes")

    def transform(self, x_in):
        """
        x_in : symbolic theano variable

        """
        raise NotImplementedError("Subclass me!")

    def own(self, name):
        return os.path.join(self.name, name)

    @property
    def dropout(self):
        """
        Used as a probability.
        """
        return self._scalars.get("dropout")


class Affine(BaseLayer):
    """
    Affine Transform Layer
      (i.e., a fully-connected non-linear projection)

    """
    param_names = ["weights", "bias"]

    def __init__(self, layer_args):
        """
        layer_args : AffineArgs

        """
        BaseLayer.__init__(self, layer_args)
        weights = np.zeros(self.param_shapes.get("weights"))
        bias = np.zeros(self.output_shape)
        self.param_values = {self.own('weights'):weights,
                             self.own('bias'):bias, }

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))

        """
        W = self._theta["weights"]
        b = self._theta['bias'].dimshuffle('x', 0)
        # TODO(ejhumphrey): This sucks.
        x_in = T.flatten(x_in, outdim=2)
#        selector = self.theano_rng.binomial(size=self.input_shape,
#                                            p=1.0 - self.dropout_prob,
#                                            dtype=FLOATX)
#        W = W * selector.dimshuffle(0, 'x') * self.dropout_scalar
        return self.activation(T.dot(x_in, W) + b)



class Conv3D(BaseLayer):
    """ . """
    param_names = ["weights", "bias"]

    def __init__(self, layer_args):
        """
        layer_args : ConvArgs

        """
        BaseLayer.__init__(self, layer_args)

        # Create all the weight values at once
        weight_shape = self.param_shapes.get("weights")
        fan_in = np.prod(weight_shape[1:])
        weights = self.numpy_rng.normal(loc=0.0,
                                        scale=np.sqrt(3. / fan_in),
                                        size=weight_shape)

        if self.get("activation") == 'sigmoid':
            weights *= 4

        bias = np.zeros(weight_shape[0])
        self.param_values = {self.own('weights'):weights,
                             self.own('bias'):bias, }


    def transform(self, x_in):
        """

        """
        W = self._theta['weights']
        b = self._theta['bias']
        weight_shape = self.param_shapes.get("weights")
#        W = W*selector.dimshuffle(0,'x','x','x')/scalar
        z_out = T.nnet.conv.conv2d(input=x_in,
                                   filters=W,
                                   filter_shape=weight_shape,
                                   border_mode=self.get("border_mode"))

        selector = self.theano_rng.binomial(size=self.output_shape[:1],
                                            p=1.0 - self.dropout,
                                            dtype=FLOATX)
#        scalar = selector.sum()
        scalar = 1.0# self.dropout_scalar

        z_out = (z_out + b.dimshuffle('x', 0, 'x', 'x')) * selector.dimshuffle('x', 0, 'x', 'x') * scalar
        z_out = self.activation(z_out)
        return downsample.max_pool_2d(z_out,
                                      self.get("pool_shape"),
                                      ignore_border=False)


class Softmax(BaseLayer):
    """
    """
    param_names = ["weights", "bias"]

    def __init__(self, layer_args):
        """
        """
        BaseLayer.__init__(self, layer_args)
        weight_shape = self.param_shapes.get("weights")
        scale = np.sqrt(6. / np.sum(weight_shape))

        weights = self.numpy_rng.normal(loc=0.0, scale=scale, size=weight_shape)
        bias = np.zeros(self.output_shape)

        self.param_values = {self.own('weights'):weights,
                             self.own('bias'):bias, }

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))
        """
        # TODO(ejhumphrey): This sucks.
        x_in = x_in.flatten(2)
        W = self._theta["weights"]
        b = self._theta["bias"]
        return T.nnet.softmax(T.dot(x_in, W) + b.dimshuffle('x', 0))


