"""
"""

import theano

# from . import FLOATX
from . import TENSOR_TYPES
from .. import utils
from .. import urls


class Graph(object):
    """
    An acyclic graph.
    """
    # Unnecessary? Graphs probably shouldn't be involved in any serialization.
    # _NAME = "name"
    # _NODES = "nodes"
    # _EDGES = "edges"

    def __init__(self, nodes, edges):
        """
        Parameters
        ----------
        nodes : list of Nodes
            asdf
        edges : list of tuples
            asdf
        """
        self.inputs = dict()
        self.outputs = dict()
        self._params = set()
        self.scalars = dict()

        # self.name = name
        self.nodes = nodes
        self.edges = edges
        self.connect()

    @property
    def nodes(self):
        return self._nodes.values()

    @nodes.setter
    def nodes(self, nodes):
        self._input_shapes = dict()
        [self._input_shapes.update(n.input_shapes) for n in nodes]
        self._nodes = dict([(n.name, n) for n in nodes])

    def connect(self):
        """Graph traversal logic to connect arbitrary, acyclic networks."""
        assert self.edges
        assert self.nodes

        self.inputs.clear()
        self.outputs.clear()

        # Local data structures to consume.
        connections = utils.edges_to_connections(self.edges)

        # Recover inputs from the connection map.
        input_keys = connections.keys()
        inputs = dict()
        for key in input_keys:
            if urls.is_input(key):
                input_var = urls.parse_input(key)
                next_inputs = connections.pop(key)
                # Recover the input's ndim from its sink.
                # TODO(ejhumphrey): Stop being a hack and make sure they're
                #   all the same ndim.
                ndim = len(self._input_shapes[next_inputs[0]])
                self.inputs[input_var] = TENSOR_TYPES[ndim](name=input_var)
                # Unpack all inputs this variable maps to.
                inputs.update(dict([(k, self.inputs[input_var])
                                    for k in next_inputs]))

        # Having created the input variables, walk the edges until
        #   inputs is empty.
        while inputs:
            # Set a flag to catch a full loop where no inputs are consumed.
            # This should never happen when everything is going to plan.
            # print "Inputs: %s" % inputs
            # print "Connections: %s\n" % connections
            nothing_happened = True
            for node in self.nodes:
                if not node.validate_inputs(inputs):
                    # Insufficient inputs; try the next node.
                    continue
                nothing_happened = False
                node_outputs = node.transform(node.filter_inputs(inputs))
                [self._params.add(p) for p in node.params.values()]
                self.inputs.update(node.scalars)
                # self.scalars.update(node.scalars)
                for out_key, output in node_outputs.items():
                    if not out_key in connections:
                        # When would this ever happen? Never, right?
                        raise IOError("%s produced an extraneous output: %s" %
                                      (node.name, out_key))
                    # Associate the symb output with its input names, removing
                    # equivalences from the connection map.
                    for in_key in connections.pop(out_key):
                        if urls.is_output(in_key):
                            # - output leaves the graph, added to outputs
                            output.name = urls.parse_output(in_key)
                            self.outputs[output.name] = output
                        else:
                            # - output maps to a new input, added to 'inputs'
                            inputs[in_key] = output
                break
            if nothing_happened:
                raise ValueError("Caught infinite connection loop.")

    @property
    def params(self):
        """
        The symbolic parameters in a flat dictionary.

        Returns
        -------
        params : dict
            Theano symbolic variables
        """
        return dict([(p.name, p) for p in self._params])

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
        return dict([(k, p.get_value()) for k, p in self.params.items()])

    @param_values.setter
    def param_values(self, param_values):
        """
        Parameters
        ----------
        param_values : dict
            Flat dictionary of values, keyed by full parameter names.
        """
        # Filter and validate.
        for node in self.nodes:
            node.param_values = param_values

    def compile(self, output_name=None):
        return theano.function(
            inputs=self.inputs.values() + self.scalars.values(),
            outputs=self.outputs.get(output_name),
            allow_input_downcast=True,
            on_unused_input='warn')

    @property
    def variables(self):
        """Return all symbolic variables
        """
        all_vars = dict()
        all_vars.update(self.outputs)
        all_vars.update(dict([(k, p) for k, p in self.params.items()]))
        return all_vars


    # TODO(ejhumphrey): These come from node-inspection
    # -------------------------------------------------
    # @property
    # def input_shapes(self):
    #     return self.layers[0].input_shape

    # @property
    # def output_shape(self):
    #     return self.layers[-1].output_shape
    # -------------------------------------------------

    # def __str__(self):
    #     return json.dumps({self.name: self}, indent=4)

    # @classmethod
    # def load(self, definition_file, param_file=None):
    #     """Load a network from disk.

    #     Parameters
    #     ----------
    #     definition_file : string
    #         Path to a file that matches a JSON-ed model definition.
    #     param_file : string
    #         Path to a pickled dictionary of parameters.
    #     """
    #     net = load_network_def(definition_file)
    #     if param_file:
    #         net.param_values = cPickle.load(open(param_file))
    #     return net

    # @property
    # def scalars(self):
    #     scalars = dict()
    #     for node in self.nodes.values():
    #         scalars.update(dict([(self.own(k), v)
    #                              for k, v in node.scalars.items()]))
    #     return scalars

    # @property
    # def inputs(self):
    #     inputs = dict([(self.own(k), v) for k, v in self._inputs.items()])
    #     inputs.update(self.scalars)
    #     return inputs

    # @property
    # def outputs(self):
    #     return dict([(self.own(k), v) for k, v in self._outputs.items()])

    # def __call__(self, inputs):
    #     if self._fx is None:
    #         self.compile()
    #     return self._fx(**inputs)

    # def save_params(self, filebase, add_time):
    #     save_params(self, filebase, add_time)

    # def save_definition(self, filebase, add_time):
    #     save_definition(self, filebase, add_time)

'''
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

'''
