"""
"""

ALLOW_DOWNCAST = True
import os
import numpy as np
import theano
import theano.tensor as T
from .functions import Activations
from .parameters import Parameter

node_def = """class: Affine 
name: affine
parameters: [ name: weights shape: (100, 300),
              name: bias shape: (300,) ]
activation: tanh
args: []"""

node_dict = {"class": "Affine",
             "name": "affine",
             "parameters": [{"name":"weights", "shape": (100, 300), },
                             {"name":"bias", "shape": (300,), }],
             "activation": "tanh",
             "args": [], }

node_def = """class: Affine 
name: sparse_affine
parameters: [ name: weights shape: (100, 300),
              name: bias shape: (300,) ]
activation: hard_shrink
args: [theata: 0.25]"""

node_def = """class: Conv 
name: conv3d
parameters: [ name: kernel shape: (1, 50, 9, 5),
              name: bias shape: (50,),
              name: pool shape: (1, 1) ]
activation: relu
args: []"""

node_def = """class: Conv 
name: conv2d
parameters: [ name: kernel shape: (128, 9, 5),
              name: bias shape: (128,),
              name: pool shape: (1, 1) ]
activation: sigmoid
args: []"""

node_def = """class: L2Distance 
name: euclidean
parameters: []
activation: linear
args: []"""

node_def = """class: InputStacker 
name: stacker
parameters: []
activation: linear
args: []"""

node_def = """class: Normalization 
name: stdizer
parameters: [ name: mean shape: (100, 100),
              name: scalar shape: (100, 100) ]
activation: linear
args: []"""

# dot-separators are attributes
# slashes indicate ownership
# colons represent I/O
manifest = """
affine0.class = Affine 
affine0.name = affine0
affine0/bias.name = b
affine0/bias.param -> symbolic
affine0/bias.shape = (300,)
affine0/bias.value = ?
affine0/weights.name = W 
affine0/weights.param -> symbolic
affine0/weights.shape = (100, 300) 
affine0/weights.value = ?
affine0:x.input -> symbolic
affine0:z.output -> symbolic
"""


class Node(dict):

    def __init__(self, name=None, parameters=None, activation=None, args=None):
        """Base class for all nodes (points on the graph)
        
        Parameters
        ----------
        name : string
            Unique identifier of this node.
        parameters : list
            List of initialized Parameters for this node.
        activation : string
            Name of the activation function.
        args : ?
            Should probably try to do away with this at the base class level.
        """
        self._name = name
        self._parameters = parameters
        self.activation = activation
        self._args = args

        self.update({"name":self.name,
                     "parameters":self.parameters,
                     "activation":self.activation,
                     "args":self.args})

        self.inputs = {}
        self.outputs = {}
    
    def own(self, name):
        return self.name + "/" + name

    @property
    def name(self):
        return self._name

    @property
    def activation(self):
        """Name of the activation function."""
        return self._activation_name

    @activation.setter
    def activation(self, name):
        self._activation_fx = Activations.get(name)
        self._activation_name = name

    @property
    def parameters(self):
        """Dictionary of all parameters in this node.
        
        Conceptually, this is Theta. Each parameter is extended to the full
        name of thise node.
        """
        local_dict = dict()
        for k,v in self._parameters.iteritems():
            local_dict[self.own(k)] = v
        return local_dict

    @parameters.setter
    def set_parameters(self, parameters):
        local_dict = dict()
        for k,v in parameters.iteritems():
            local_dict[os.path.split(k)[-1]] = v
        self._parameters.update(parameters)
    
    @property
    def values(self):
        pass
    
    @property
    def args(self):
        return self._args

    

class Affine(Node):
    """
    """
    WEIGHTS = "weights"
    BIAS = "bias"
    INPUT = "x"
    OUTPUT = "z"

    def __init__(self, name=None,
                 weight_shape=None,
                 activation='linear',
                 values=None):
        """
        """

        assert weight_shape is None or values is None, \
            "One of weight_shape or values must be None."
        if not weight_shape is None:
            self.weights = Parameter(name=Affine.WEIGHTS,
                                     shape=weight_shape)
            self.bias = Parameter(Affine.BIAS,
                                  shape=weight_shape[1:])
        else:
            assert Affine.WEIGHTS in values and Affine.BIAS in values, \
                "Values must contain keys '%s' and '%s'." % (Affine.WEIGHTS,
                                                             Affine.BIAS)
            self.weights = Parameter(name=Affine.WEIGHTS,
                                     value=values.get(Affine.WEIGHTS))
            self.bias = Parameter(name=Affine.BIAS,
                                  value=values.get(Affine.BIAS))

        Node.__init__(self, name=name,
                      parameters={Affine.WEIGHTS:self.weights,
                                  Affine.BIAS:self.bias},
                      activation=activation,
                      args={})

        self.inputs[Affine.INPUT] = None
        self.outputs[Affine.OUTPUT] = None

    def __call__(self, x):
        self.inputs[Affine.INPUT] = x
        z = T.dot(x, self.weights.variable) + self.bias.variable
        self.outputs[Affine.OUTPUT] = self._activation_fx(z)
        return self.outputs[Affine.OUTPUT]
    
    def 
