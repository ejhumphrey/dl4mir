"""
"""

import cPickle
import json
import os

import theano

from ejhumphrey.dnn.core import FLOATX
from ejhumphrey.dnn.core import TENSOR_TYPES
from ejhumphrey.dnn.core.nodes import NodeFactory
from ejhumphrey.dnn import utils
from ejhumphrey.dnn import urls


DEF_EXT = "definition"
PARAMS_EXT = "params"


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

    # return Network([Layer(args) for args in layer_args])


class Network(dict):
    """
    An acyclic graph.
    """
    NAME = "name"
    NODES = "nodes"
    EDGES = "edges"
    INPUTS = "input_names"

    def __init__(self, name, input_names, nodes, edges):
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
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._input_names = input_names

        self._init_inputs()
        self._own_symbolics()
        self.compute_outputs()

    @classmethod
    def from_args(cls, args):
        assert cls.NAME in args
        assert cls.NODES in args
        assert cls.EDGES in args
        assert cls.INPUTS in args
        nodes = dict([(k, NodeFactory(a)) for k, a in args[cls.NODES].items()])
        return cls(name=args[cls.NAME],
                   input_names=args[cls.INPUTS],
                   nodes=nodes,
                   edges=args[cls.EDGES])

    def own(self, path):
        url = path
        if not self.name in url:
            url = urls.append_node(self.name, path)
        return url

    @property
    def name(self):
        return self.get(self.NAME)

    @name.setter
    def name(self, value):
        self[self.NAME] = value

    @property
    def _nodes(self):
        return self.get(self.NODES)

    @_nodes.setter
    def _nodes(self, value):
        self[self.NODES] = value

    @property
    def nodes(self):
        return dict([(self.own(k), v) for k, v in self._nodes.items()])

    @property
    def _edges(self):
        return self.get(self.EDGES)

    @_edges.setter
    def _edges(self, value):
        self[self.EDGES] = value

    @property
    def _input_names(self):
        return self.get(self.INPUTS)

    @_input_names.setter
    def _input_names(self, value):
        self[self.INPUTS] = value

    @property
    def input_names(self):
        return self.inputs.keys()

    def _init_inputs(self):
        self._inputs = {}
        for node_path in self._input_names:
            node, param = urls.split_param(node_path)
            ndim = len(self._nodes[node].input_shapes[node_path])
            self._inputs[node_path] = TENSOR_TYPES[ndim](name=node_path,
                                                         dtype=FLOATX)

    def compute_outputs(self):
        """Graph traversal logic to connect arbitrary, acyclic networks.
        """
        inputs = self._inputs.copy()
        connections = utils.edges_to_connections(self._edges)
        self._outputs = {}
        while connections or inputs:
            no_match = True
            for node in self._nodes.values():
                if not node.validate_inputs(inputs):
                    # Insufficient inputs; continue to the next.
                    continue
                no_match = False
                node_outputs = node.transform(node.filter_inputs(inputs))
                self._outputs.update(node_outputs)
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

    def _own_symbolics(self):
        for param in self.params.values():
            # Only own unclaimed parameters.
            if not urls.network(param.name):
                param.name = self.own(param.name)

        for var in self.inputs.values():
            var.name = self.own(var.name)

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
        return json.dumps({self.name: self}, indent=4)

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
        for node in self.nodes.values():
            all_params.update(dict([(self.own(k), v)
                                    for k, v in node.params.items()]))
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
        return dict([(k, v.get_value()) for k, v in self.params.items()])

    @param_values.setter
    def param_values(self, param_values):
        """
        Parameters
        ----------
        param_values : dict
            Flat dictionary of values, keyed by full parameter names.
        """
        # Filter and validate.
        for node in self.nodes.values():
            node.param_values = param_values

    @property
    def scalars(self):
        scalars = dict()
        for node in self.nodes.values():
            scalars.update(dict([(self.own(k), v)
                                 for k, v in node.scalars.items()]))
        return scalars

    @property
    def inputs(self):
        inputs = dict([(self.own(k), v) for k, v in self._inputs.items()])
        inputs.update(self.scalars)
        return inputs

    @property
    def outputs(self):
        return dict([(self.own(k), v) for k, v in self._outputs.items()])

    def compile(self, output_name=None):
        self._fx = theano.function(inputs=self.inputs.values(),
                                   outputs=self._outputs.get(output_name),
                                   allow_input_downcast=True,
                                   on_unused_input='warn')

    def __call__(self, inputs):
        if self._fx is None:
            self.compile()
        return self._fx(**inputs)

    @property
    def variables(self):
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
