"""
"""


from ejhumphrey.dnn.core.layers import Layer
from ejhumphrey.dnn.core.graphs import Network
from ejhumphrey.dnn.core.graphs import save
from ejhumphrey.dnn.core.modules import Loss
from ejhumphrey.dnn.core.updates import SGD
import os
import time

def select_update(base, newdict):
    """Update a dictionary with only the intersection of the two key sets."""
    for k, v in newdict.iteritems():
        if k in base:
            base[k] = v

class Trainer(object):

    def __init__(self, name, save_directory):
        self.network = None
        self.loss = None
        self.update = None
        self.monitor = None
        self.input_name, self.output_name = "input", "output"
        self.target_name = "y_target"
        self.name = name
        self.save_directory = save_directory


    def build_network(self, layer_args):
        """
        Parameters
        ----------
        layer_args : list of arg-like dicts
            Can be created with an argument factory or loaded from a json-file.
        """
        self.network = Network([Layer(args) for args in layer_args])
        self.network.compile(self.input_name, self.output_name)

    def configure_losses(self, loss_args):
        """
        Parameters
        ----------
        loss_args : list of tuples
            Each tuple must contain a string pointing to a representation in
            network.vars and a loss type declared in LossFunctions.
        """
        assert self.network, "Network must be built first."
        self.loss = Loss()
        for iname, ltype in loss_args:
            self.loss.register(self.network, iname, ltype)
        self.loss.compile()

    def configure_updates(self, update_args=None):
        """
        Parameters
        ----------
        update_args : list of strings, default None
            Names of parameters in the network to update.
        """
        assert self.loss, "Loss must be configured first."
        self.update = SGD()
        if update_args is None:
            update_args = self.network.params.keys()
        params = dict([(k, self.network.params.get(k)) for k in update_args])
        self.update.compute_updates(self.loss, params)
        self.update.compile()


    def run(self, sources, train_params, hyperparams):
        assert self.network, "Network must be built first."
        assert self.loss, "Loss must be configured first."
        assert self.update, "Update must be configured first."

        Done = False
        train_inputs = self.update.empty_inputs()
        select_update(train_inputs, hyperparams)

        loss_inputs = self.loss.empty_inputs()
        select_update(loss_inputs, hyperparams)

        while not Done:
            try:
                batch = sources['train'].next_batch(train_params.get("batch_size"))
                train_inputs[self.input_name] = batch.values
                train_inputs[self.target_name] = batch.labels

                train_loss = self.update(train_inputs)

                if (self.update.iteration % train_params["checkpoint_freq"]) == 0:
                    self.checkpoint(train_loss)

                if self.update.iteration >= train_params.get("max_iterations"):
                    Done = True

            except KeyboardInterrupt:
                break
        print "\nFinished.\n"

    def checkpoint(self, train_loss):
        save(self.network, os.path.join(self.save_directory, self.name))
        print "[%s] %d: %0.4f" % (time.asctime(),
                                  self.update.iteration,
                                  train_loss)

