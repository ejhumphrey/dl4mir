#!/usr/bin/env python
"""Train a deepnet for chord identification.

Sample Call:
bash$ ipython ejhumphrey/dnn/models/chordrec.py \
majmin_chord_classifier_000 \
/home/ejhumphrey/chords/models \
/home/ejhumphrey/chords/chordrec.dsf \
/home/ejhumphrey/chords/MIREX09_chord_map.txt
"""

import argparse

from marl.hewey import file
from marl.hewey import sources

from ejhumphrey.dnn.core.framework import Trainer
from ejhumphrey.dnn.core.layers import Conv3DArgs
from ejhumphrey.dnn.core.layers import AffineArgs
from ejhumphrey.dnn.core.layers import SoftmaxArgs
from ejhumphrey.dnn.core.layers import Layer
import json

base_hyperparams = {'classifier/dropout': 0.0,
                    'affine2/dropout': 0.0,
                    'convlayer1/dropout': 0.0,
                    'convlayer0/dropout': 0.0,
                    'classifier/bias/learning_rate': 0.0,
                    'classifier/weights/learning_rate': 0.0,
                    'affine2/bias/learning_rate': 0.0,
                    'affine2/weights/learning_rate': 0.0,
                    'convlayer1/bias/learning_rate': 0.0,
                    'convlayer1/weights/learning_rate': 0.0,
                    'convlayer0/bias/learning_rate': 0.0,
                    'convlayer0/weights/learning_rate': 0.0, }

base_train_params = {"max_iterations":50000,
                     "checkpoint_freq":50,
                     "batch_size":50,
                     "left":40,
                     "right":39, }

N_DIM = base_train_params.get("left") + base_train_params.get("right") + 1
P_DIM = 192

# Model Configuration.
loss_pairs = [("output", 'nll'), ]

parameter_updates = ['affine2/bias',
                     'affine2/weights',
                     'classifier/bias',
                     'classifier/weights',
                     'convlayer0/bias',
                     'convlayer0/weights',
                     'convlayer1/bias',
                     'convlayer1/weights']

def build_layers():
    layer_0 = Layer(Conv3DArgs(name='convlayer0',
                               input_shape=(1, N_DIM, P_DIM),
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
    return [layer_0, layer_1, layer_2, classifier]


def set_all_learning_rates(hyperparams, eta):
    new_hyperparams = hyperparams.copy()
    for k in new_hyperparams:
        if k.count("learning_rate"):
            new_hyperparams[k] = eta
    return new_hyperparams


def training_source(filepath, train_params):
    file_handle = file.DataSequenceFile(filepath)
    return sources.SequenceSampler(dataset=file_handle,
                                   left=train_params.get("left"),
                                   right=train_params.get("right"),
                                   refresh_prob=0.1,
                                   cache_size=250)


def load_label_map(filepath):
    """JSON refuses to store integer zeros, so they are written as strings and
    interpreted as integers on load.
    """
    return dict([(k, int(v)) for k, v in json.load(open(filepath)).iteritems()])


def main(args):
    layers = build_layers()

    trainer = Trainer(name=args.name,
                      save_directory=args.save_directory)

    trainer.build_network(layers)
    trainer.configure_losses(loss_pairs)
    trainer.configure_updates(parameter_updates)
    print layers
    train_params = base_train_params.copy()
    dset = training_source(args.training_data, train_params)
    dset.set_value_shape((1, N_DIM, P_DIM))
    dset.set_label_map(load_label_map(args.label_map))

    sources = {'train':dset}
    hyperparams = set_all_learning_rates(base_hyperparams, 0.02)
    trainer.run(sources, train_params, hyperparams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deepnet training for Chord Recognition.")

    parser.add_argument("name",
                        metavar="name", type=str,
                        help="Name for this trained model.")

    parser.add_argument("save_directory",
                        metavar="save_directory", type=str,
                        help="Base directory to save data.")

    parser.add_argument("training_data",
                        metavar="training_data", type=str,
                        help="Hewey file to use for training.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON file mapping chords to integers.")

    main(parser.parse_args())

