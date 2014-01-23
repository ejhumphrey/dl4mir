"""
"""
import json
from theano import function
from . import *
import time
# from .core.losses import Loss


class Driver(object):

    def __init__(self, nodes, edges, losses=None, constraints=None):
        self._graph_fx = None
        self._loss_fx = None
        self._update_fx = None

        self._graph = Graph(nodes=nodes, edges=edges)
        self._outputs = self._graph.outputs.values()

        if losses is None:
            losses = []
        self._losses = Accumulator(losses, self._graph.variables)

        self._updates = SGD(self._graph.params, self._losses.total)

        if constraints is None:
            constraints = []
        self.constraints = []
        # self.constraints = Constraints(constraints, graph.params)
        self._graph_compile()
        self._loss_compile()
        self._update_compile()
        self.monitor = None

    def __str__(self):
        args = dict(nodes=self._graph.nodes,
                    edges=self._graph.edges,
                    losses=self._losses,
                    constraints=self.constraints)
        return json.dumps(args, indent=4)

    @classmethod
    def from_file(cls, train_def):
        pass

    def apply_constraints(self, args):
        pass

    def _graph_compile(self):
        self._graph_inputs = self._graph.inputs.values()
        self._graph_fx = function(
            inputs=self._graph_inputs,
            outputs=self._outputs,
            allow_input_downcast=True)

    def _loss_compile(self):
        if not self._losses:
            return
        self._loss_inputs = self._graph_inputs + self._losses.inputs.values()
        self._loss_fx = function(
            inputs=self._loss_inputs,
            outputs=self._losses.total,
            allow_input_downcast=True)

    def _update_compile(self):
        if not self._updates:
            return
        self._update_inputs = self._loss_inputs + self._updates.inputs.values()
        self._update_fx = function(
            inputs=self._update_inputs,
            outputs=self._losses.total,
            updates=self._updates,
            allow_input_downcast=True)

    @property
    def graph_inputs(self):
        return [v.name for v in self._graph_inputs]

    @property
    def loss_inputs(self):
        return [v.name for v in self._loss_inputs]

    @property
    def update_inputs(self):
        return [v.name for v in self._update_inputs]

    def _validate(self, inputs, keys):
        return set(inputs.keys()) == set(keys)

    def transform(self, inputs):
        assert self._graph_fx
        self._validate(inputs, self.graph_inputs)
        # return self._graph_fx(**inputs)
        return dict([(v.name, res) for v, res in zip(self._outputs, outputs)])

    def loss(self, inputs):
        assert self._loss_fx
        self._validate(inputs, self.loss_inputs)
        return self._loss_fx(**inputs)

    def update(self, inputs):
        assert self._update_fx
        self._validate(inputs, self.update_inputs)
        return self._update_fx(**inputs)

    def train(self, inputs, n_iter, print_frequency=50):
        try:
            for n, data in enumerate(inputs):
                loss = self.update(data)
                if (n % print_frequency) == 0:
                    print "[%s] Iter: %07d\tloss: %0.5f" % (time.asctime(),
                                                            n, loss)
        except KeyboardInterrupt:
            print "[%s] Stopping - Iter: %07d\tloss: %0.5f" % (time.asctime(),
                                                               n, loss)


class Trainer(dict):
    def __init__(self):
        pass

    def run(self,):
        pass
