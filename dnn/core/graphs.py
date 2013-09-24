'''
Created on Nov 6, 2012

@author: ejhumphrey
'''

import cPickle
import json
import os
import time

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
    return time.strftime(TIME_FMT) + "m%3d" % int((time.time() % 1) * 1000)

def save(net, filebase):
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

    nowstamp = timestamp()
    # Save json-encoded architecture.
    model_def_file = "%s-%s.%s" % (filebase, nowstamp, DEF_EXT)
    model_def = open(model_def_file, "w")
    json.dump(net.layers, model_def, indent=2)
    model_def.close()
    # Save pickled parameters.
    model_params_file = "%s-%s.%s" % (filebase, nowstamp, PARAMS_EXT)
    model_params = open(model_params_file, "w")
    cPickle.dump(net.param_values, model_params)
    model_params.close()

def load(filebase):
    """Load a network from disk.

    Parameters
    ----------
    filebase : string
        Path to a file that matches a definition and parameter file.
    """
    model_def = filebase + DEF_EXT
    assert os.path.exists(model_def), \
        "Model definition file '%s' does not exist." % model_def

    model_params = filebase + PARAMS_EXT
    assert os.path.exists(model_params), \
        "Model parameter file '%s' does not exist." % model_params

    net = Network([Layer(args) for args in json.load(open(model_def))])
    net.param_values = cPickle.load(open(model_params))
    return net


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
#        self.layers = [Layer(args) for args in layer_args]

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

    @property
    def dropout(self):
        """
        Dropout states over the graph.

        Returns
        -------

        """
        return [l.dropout for l in self.layers]

    @dropout.setter
    def dropout(self, states):
        """
        set dropout bools over the graph
        """
        for l, b in zip(self.layers, states):
            l.dropout = b

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
        layer_input = x_in
        for layer in self.layers:
            layer_input = layer.transform(layer_input)

        return layer_input

