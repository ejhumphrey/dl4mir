"""
"""

import cPickle
import json
import os

import theano
import theano.tensor as T

from ejhumphrey.dnn.core import FLOATX
from ejhumphrey.dnn.core import nodes
from ejhumphrey.dnn import utils


DEF_EXT = "definition"
PARAMS_EXT = "params"
TENSOR_TYPES = {1: T.matrix,
                2: T.tensor3,
                3: T.tensor4}


def save_params(net, filebase, add_time=True):
    """Serialize a network to disk.

    Parameters
    ----------
    net : Network
        Instantiated network to serialize.
    filebase : string
        Path to save a parameter dictionary, with optional timestamp.
    """
    model_directory = os.path.split(filebase)[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    nowstamp = ""
    if add_time:
        nowstamp += "-" + utils.timestamp()
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
        Path to save the model definition, with optional timestamp.
    """
    model_directory = os.path.split(filebase)[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    nowstamp = ""
    if add_time:
        nowstamp += "-" + utils.timestamp()
    # Save json-encoded architecture.
    model_def_file = "%s%s.%s" % (filebase, nowstamp, DEF_EXT)
    model_def = open(model_def_file, "w")
    json.dump(net.layers, model_def, indent=2)
    model_def.close()


def load_network_def(definition_file):
    """Load a network from disk.

    Parameters
    ----------
    definition_file : string
        Path to a file that matches a JSON-ed model definition.
    """

    network_def = utils.convert(json.load(open(definition_file)))
    assert "edges" in network_def
    assert "nodes" in network_def
    assert "input_dims" in network_def
    # return Network([Layer(args) for args in layer_args])


class Network(dict):
    """
    An acyclic graph.
    """
    NODES = "nodes"
    EDGES = "edges"
    INPUTS = "inputs"

    def __init__(self, input_names, nodes, edges):
        """
        Parameters
        ----------
        input_names : list
            asdf
        nodes : dict
            asdf
        edges : list
            asdf
        """
        self.nodes = nodes
        self.edges = edges
        self.input_names = input_names

        self._init_inputs()
        self._compute_outputs()

    @property
    def nodes(self):
        return self.get(self.NODES)

    @nodes.setter
    def nodes(self, value):
        self[self.NODES] = value

    @property
    def input_names(self):
        return self.get(self.INPUTS)

    @input_names.setter
    def input_names(self, value):
        self[self.INPUTS] = value

    @property
    def edges(self):
        return self.get(self.EDGES)

    @edges.setter
    def edges(self, value):
        self[self.EDGES] = value

    def _init_inputs(self):
        self.inputs = {}
        for full_name in self.input_names:
            node_name, var_name = os.path.split(full_name)
            ndim = len(self.nodes[node_name].input_shapes[full_name])
            self.inputs[full_name] = TENSOR_TYPES[ndim](
                name=full_name, dtype=FLOATX)

    def _compute_outputs(self):
        """Graph traversal logic to connect arbitrary, acyclic networks.
        """
        inputs = self.inputs.copy()
        connections = utils.edges_to_connections(self.edges)
        self.outputs = {}
        while connections or inputs:
            no_match = True
            for node in self.nodes.values():
                if not node.validate_inputs(inputs):
                    # Insufficient inputs; continue to the next.
                    continue
                no_match = False
                node_outputs = node.transform(node.filter_inputs(inputs))
                self.outputs.update(node_outputs)
                for new_output in node_outputs:
                    if not new_output in connections:
                        # Terminal output; move along.
                        continue
                    # List of new input names that are equivalent to the symb
                    # var produced as an output.
                    equiv_inputs = connections.pop(new_output)
                    # Associate the symb output with its input names.
                    new_inputs = dict([(equiv_input, node_outputs[new_output])
                                      for equiv_input in equiv_inputs])
                    inputs.update(new_inputs)
                break

            if no_match:
                raise ValueError("Caught infinite connection loop.")

    # TODO(ejhumphrey): These come from node-inspection
    # -------------------------------------------------
    # @property
    # def input_shapes(self):
    #     return self.layers[0].input_shape

    # @property
    # def output_shape(self):
    #     return self.layers[-1].output_shape
    # -------------------------------------------------

    def __str__(self):
        return json.dumps(self, indent=2)

    @classmethod
    def load(self, definition_file, param_file=None):
        """Load a network from disk.

        Parameters
        ----------
        definition_file : string
            Path to a file that matches a JSON-ed model definition.
        param_file : string
            Path to a pickled dictionary of parameters.
        """
        net = load_network_def(definition_file)
        if param_file:
            net.param_values = cPickle.load(open(param_file))
        return net

    @property
    def params(self):
        """
        The symbolic parameters in a flat dictionary.

        Returns
        -------
        params : dict
            Symbolic variables keyed by full parameter names,
            i.e. 'node_name/param_name'
        """
        all_params = dict()
        [all_params.update(node.params) for node in self.nodes.values()]
        return all_params

    @property
    def param_values(self):
        """
        The numerical parameters in a flat dictionary.

        Returns
        -------
        param_values : dict
            Numpy arrays keyed by full parameter names,
            i.e. 'node_name/param_name'
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
        for node in self.nodes:
            node.param_values = param_values

    @property
    def scalars(self):
        all_scalars = list()
        [all_scalars.extend(layer.scalars) for layer in self.layers]
        return all_scalars

    def compile(self, output_name=None):
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
