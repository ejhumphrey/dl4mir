'''
Created on Nov 6, 2012

@author: ejhumphrey
'''

import cPickle
import json
import os
import time

import theano
import theano.tensor as T

from . import FLOATX
from .layers import Layer

TIME_FMT = "%Y%m%d_%H%M%S"
DEF_EXT = "definition"
PARAMS_EXT = "params"

def timestamp():
    """Returns a string representation of the time, like:
    YYYYMMDD_HHMMSSmMMM
    """
    return time.strftime(TIME_FMT) + "m%03d" % int((time.time() % 1) * 1000)

def save_params(net, filebase, add_time=True):
    """Serialize a network to disk.

    Parameters
    ----------
    net : Network
        Instantiated network to serialize.
    filebase : string
        Path to write the appropriate information. Two time-stamped files are
        created:
        1. A human-readable json dump of the network architecture.
        2. A pickled dictionary of the networks numerical parameters.
    """
    model_directory = os.path.split(filebase)[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    nowstamp = ""
    if add_time:
        nowstamp += "-" + timestamp()
    # Save pickled parameters.
    model_params_file = "%s%s.%s" % (filebase, nowstamp, PARAMS_EXT)
    model_params = open(model_params_file, "w")
    cPickle.dump(net.param_values, model_params)
    model_params.close()

def save_definition(net, filebase, add_time=True):
    """Serialize a network to disk.

    Parameters
    ----------
    net : Network
        Instantiated network to serialize.
    filebase : string
        Path to write the appropriate information. Two time-stamped files are
        created:
        1. A human-readable json dump of the network architecture.
        2. A pickled dictionary of the networks numerical parameters.
    """
    model_directory = os.path.split(filebase)[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    nowstamp = ""
    if add_time:
        nowstamp += "-" + timestamp()
    # Save json-encoded architecture.
    model_def_file = "%s%s.%s" % (filebase, nowstamp, DEF_EXT)
    model_def = open(model_def_file, "w")
    json.dump(net.layers, model_def, indent=2)
    model_def.close()

def load(definition_file, param_file):
    """Load a network from disk.

    Parameters
    ----------
    filebase : string
        Path to a file that matches a definition and parameter file.
    """

    layer_args = convert(json.load(open(definition_file)))
    net = Network([Layer(args) for args in layer_args])
    net.param_values = cPickle.load(open(param_file))
    net.compile()
    return net

def convert(obj):
    """Convert unicode to strings.

    Known issue: Uses dictionary comprehension, and is incompatible with 2.6.
    """
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [convert(element) for element in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj


class Network(object):
    """
    Feed-forward graph.
    """
    def __init__(self, layers, name=""):
        """
        layers : list
            List of layers.

        """
        self.name = name
        self.layers = layers
        self.input_name = "x_input"
        self.output_name = "z_output"
        self._inputs = []
        self._outputs = {}
        self._fx = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def __str__(self):
        return json.dumps(self.layers, indent=2)

    def symbolic_input(self, name):
        """
        Return a symbolic theano variable fitting this network.

        Parameters
        ----------
        name : str
            string for the variable. must be unique to calling entity,
            because it will be live for subsequent function calls.
        """
        n_dim = len(self.input_shape)
        if n_dim == 1:
            x_in = T.matrix(name=name, dtype=FLOATX)
        elif n_dim == 2:
            x_in = T.tensor3(name=name, dtype=FLOATX)
        elif n_dim == 3:
            x_in = T.tensor4(name=name, dtype=FLOATX)
        else:
            raise ValueError("Unsupported input dimensionality: %d" % n_dim)

        return x_in

    def symbolic_output(self, name):
        """
        Return a symbolic theano variable fitting this network

        Parameters
        ----------
        name : str
            string for the variable. must be unique to calling entity,
            because it will be live for subsequent function calls.
        """
        n_dim = len(self.output_shape)
        if n_dim == 1:
            x_in = T.matrix(name=name, dtype=FLOATX)
        elif n_dim == 2:
            x_in = T.tensor3(name=name, dtype=FLOATX)
        elif n_dim == 3:
            x_in = T.tensor4(name=name, dtype=FLOATX)
        else:
            raise ValueError("Unsupported input dimensionality: %d" % n_dim)

        return x_in

    @property
    def params(self):
        """
        The symbolic parameters in a flat dictionary.

        Returns
        -------
        params : dict
            Symbolic variables keyed by full parameter names,
            i.e. 'layer_name/param_name'
        """
        all_params = dict()
        [all_params.update(layer.params) for layer in self.layers]
        return all_params

    @property
    def param_values(self):
        """
        The numerical parameters in a flat dictionary.

        Returns
        -------
        param_values : dict
            Numpy arrays keyed by full parameter names,
            i.e. 'layer_name/param_name'
        """

        param_values = {}
        for k, v in self.params.iteritems():
            param_values[k] = v.get_value()
        return param_values

    @param_values.setter
    def param_values(self, param_values):
        """
        Parameters
        ----------
        param_values : dict
            Flat dictionary of values, keyed by full parameter names.
        """
        for layer in self.layers:
            layer.param_values = param_values

    def transform(self, x_in):
        """
        Forward transform an input through the network

        Parameters
        ----------
        x_in : theano symbolic type
            we're not doing any checking, so it'll die if it's not correct

        Returns
        -------
        z_out : theano symbolic type

        """
        self._inputs = list()
        self._outputs.clear()
        self.input_name = x_in.name
        self._inputs.append(x_in)
        layer_input = x_in
        for layer in self.layers:
            layer_input = layer.transform(layer_input)
            layer_key = os.path.join(layer.name, "output")
            self._outputs[layer_key] = layer_input
        self._inputs.extend(self.scalars)
        return self._outputs[layer_key]

    @property
    def inputs(self):
        return list(self._inputs)

    @property
    def outputs(self):
        return dict(self._outputs)

    @property
    def scalars(self):
        all_scalars = list()
        [all_scalars.extend(layer.scalars) for layer in self.layers]
        return all_scalars


    def compile(self, input_name=None, output_name=None):
        if input_name is None:
            input_name = self.input_name
        if output_name is None:
            output_name = self.output_name

        self.output_name = output_name
        self._outputs[output_name] = self.transform(
            self.symbolic_input(input_name))
        self._fx = theano.function(inputs=self.inputs,
                                   outputs=self.outputs.get(output_name),
                                   allow_input_downcast=True,
                                   on_unused_input='warn')

    def __call__(self, inputs):
        if self._fx is None:
            self.compile()
        return self._fx(**inputs)

    def empty_inputs(self, fill_value=0.0):
        return dict([(x.name, fill_value) for x in self.inputs])

    @property
    def vars(self):
        """Return all symbolic variables
        """
        all_vars = dict()
        all_vars.update(self.outputs)
        all_vars.update(self.params)
        return all_vars

    def save_params(self, filebase, add_time):
        save_params(self, filebase, add_time)

    def save_definition(self, filebase, add_time):
        save_definition(self, filebase, add_time)


