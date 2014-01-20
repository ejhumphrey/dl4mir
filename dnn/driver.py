"""
"""

from ejhumphrey.dnn.core.graphs import Network
from ejhumphrey.dnn.core.losses import Loss


class Driver(dict):
    GRAPHS = 'graphs'
    LOSSES = 'losses'
    CONSTRAINTS = 'constraints'
    UPDATES = 'update_rules'

    DATA = 'data_source'
    PARAMS = 'hyperparams'

    def __init__(self, nodes, losses=None, updates=None, constraints=None):
        self[self.GRAPHS] = dict()

        self.graphs = dict()
        if losses is None:
            losses = list()

        self.losses = losses

        self[self.UPDATES] = list()
        self[self.CONSTRAINTS] = list()

        self[self.DATA] = dict()
        self[self.PARAMS] = dict()

        self.monitor = None

    @classmethod
    def from_file(cls, train_def):
        pass

    @property
    def graphs(self):
        return self[self.GRAPHS]

    @graphs.setter
    def graphs(self, graph_args):
        """
        graph_args: dict of named graphs
        """
        for k, v in graph_args:
            self[self.GRAPHS][k] = v

    @property
    def losses(self):
        return self[self.LOSSES]

    @losses.setter
    def losses(self, value):
        self[self.LOSSES] = value

    def config_graph(self, graph_args):
        net = Network.from_args(graph_args)
        self.graphs[net.name] = net

    def config_losses(self, args):
        self._loss
        pass

    def config_updates(self, args):
        pass

    def config_constraints(self, args):
        pass

    def compile(self,):
        pass


class Trainer(dict):
    def __init__(self):
        pass

    def run(self,):
        pass
