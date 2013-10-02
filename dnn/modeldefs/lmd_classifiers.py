"""
"""

from ejhumphrey.dnn.core.layers import AffineArgs
from ejhumphrey.dnn.core.layers import SoftmaxArgs
from ejhumphrey.dnn.core.layers import Layer
from ejhumphrey.dnn.core.graphs import Network

from ejhumphrey.dnn import utils

def one_layer():
    classifier = Layer(SoftmaxArgs(name='classifier',
                                   input_dim=2400,
                                   output_dim=10))

    return Network([classifier])

def two_layer():
    layer_0 = Layer(AffineArgs(name='affine0',
                               input_shape=(2400,),
                               output_shape=(512,),
                               activation='tanh'))

    classifier = Layer(SoftmaxArgs(name='classifier',
                                   input_dim=layer_0.output_shape[0],
                                   output_dim=10))

    return Network([layer_0, classifier])

def three_layer():
    layer_0 = Layer(AffineArgs(name='affine0',
                               input_shape=(2400,),
                               output_shape=(512,),
                               activation='tanh'))

    layer_1 = Layer(AffineArgs(name='affine1',
                               input_shape=layer_0.output_shape,
                               output_shape=(64,),
                               activation='tanh'))

    classifier = Layer(SoftmaxArgs(name='classifier',
                                   input_dim=layer_1.output_shape[0],
                                   output_dim=10))

    return Network([layer_0, layer_1, classifier])


def generate_config_file(net, config_filename):
    """Produce configuration files for training.

    net : Network (must be compiled).
        Instantiated network to pull parameter names from.
    config_filename : str
        Filepath for writing output config file.
    """
    learning_rate = 0.02
    parameter_updates = net.params.keys()

    hyperparams = net.empty_inputs()
    del hyperparams[net.input_name]
    for param_name in parameter_updates:
        hyperparams["%s/learning_rate" % param_name] = learning_rate

    train_params = {"max_iterations":50000,
                    "checkpoint_freq":100,
                    "batch_size":50,
                    "left":0,
                    "right":0,
                    "value_shape":(2400,)}

    loss_tuples = [(net.output_name, 'nll'), ]

    config_params = {'hyperparams':hyperparams,
                     'train_params':train_params,
                     'loss_tuples':loss_tuples,
                     'parameter_updates':parameter_updates}

    utils.json_save(config_params, config_filename)
