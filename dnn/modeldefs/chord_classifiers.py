"""
"""

from ejhumphrey.dnn.core.layers import Conv3DArgs
from ejhumphrey.dnn.core.layers import AffineArgs
from ejhumphrey.dnn.core.layers import SoftmaxArgs
from ejhumphrey.dnn.core.layers import Layer
from ejhumphrey.dnn.core.graphs import Network

from ejhumphrey.dnn import utils

def base_model():
    layer_0 = Layer(Conv3DArgs(name='convlayer0',
                               input_shape=(1, 80, 192),
                               weight_shape=(16, 13, 17),
                               pool_shape=(2, 2),
                               activation='tanh'))

    layer_1 = Layer(Conv3DArgs(name='convlayer1',
                               input_shape=layer_0.output_shape,
                               weight_shape=(20, 13, 15),
                               pool_shape=(2, 2),
                               activation='tanh'))

    layer_2 = Layer(AffineArgs(name='affine2',
                               input_shape=layer_1.output_shape,
                               output_shape=(512,),
                               activation='tanh'))

    classifier = Layer(SoftmaxArgs(name='classifier',
                                   input_dim=layer_2.output_shape[0],
                                   output_dim=25))

    return Network([layer_0, layer_1, layer_2, classifier])


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
                    "left":40,
                    "right":39,
                    "value_shape": (1, 80, 192),
                    "bins_per_octave":24,
                    "transpose":True}

    loss_tuples = [(net.output_name, 'nll'), ]

    config_params = {'hyperparams':hyperparams,
                     'train_params':train_params,
                     'loss_tuples':loss_tuples,
                     'parameter_updates':parameter_updates}

    utils.json_save(config_params, config_filename)
