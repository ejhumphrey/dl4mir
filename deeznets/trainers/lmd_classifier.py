#!/usr/bin/env python
"""Train a deepnet for chord identification.

Sample Call:
BASEDIR=/Volumes/Audio/LMD
ipython ejhumphrey/dnn/trainers/lmd_classifier.py \
test000 \
$BASEDIR/one_layer.definition \
$BASEDIR/one_layer.config \
$BASEDIR/models \
/Volumes/speedy/LMD_train00_20131001.dsf \
$BASEDIR/label_map.txt
"""

import argparse
import shutil

from marl.hewey import file
from marl.hewey import sources

from ejhumphrey.dnn.core.framework import Trainer
from ejhumphrey.dnn.utils import json_load
from ejhumphrey.datasets.utils import load_label_enum_map


def training_source(filepath, train_params):
    file_handle = file.DataSequenceFile(filepath)
    return sources.SequenceSampler(dataset=file_handle,
                                   left=train_params.get("left"),
                                   right=train_params.get("right"),
                                   refresh_prob=0.1,
                                   cache_size=500)

def main(args):
    name = args.name
    trainer = Trainer(name=name,
                      save_directory=args.save_directory)

    # Copy over input files:
    shutil.copy(args.definition, trainer.save_directory)
    shutil.copy(args.config, trainer.save_directory)

    trainer.build_network(json_load(args.definition))
    config = json_load(args.config)
    train_params = config.get("train_params")

    trainer.configure_losses(config.get("loss_tuples"))
    trainer.configure_updates(config.get("parameter_updates"))

    dset = training_source(args.training_datafile, train_params)
    dset.set_value_shape(train_params.get("value_shape"))
    label_map = load_label_enum_map(args.label_map)
    dset.set_label_map(label_map)

    sources = {'train':dset}
    trainer.run(sources, train_params, config.get("hyperparams"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deepnet training for Chord Recognition.")

    parser.add_argument("name",
                        metavar="name", type=str,
                        help="Name for this trained model.")

    parser.add_argument("definition",
                        metavar="definition", type=str,
                        help="JSON model definition.")

    parser.add_argument("config",
                        metavar="config", type=str,
                        help="Configuration file for training.")

    parser.add_argument("save_directory",
                        metavar="save_directory", type=str,
                        help="Base directory to save data.")

    parser.add_argument("training_datafile",
                        metavar="training_datafile", type=str,
                        help="Data file to use for training.")

    parser.add_argument("label_map",
                        metavar="label_map", type=str,
                        help="JSON file mapping chords to integers.")

    main(parser.parse_args())
