#!/usr/bin/env python
"""Train a deepnet for chord identification.

Sample Call:
bash$ ipython ejhumphrey/dnn/trainers/chordrec.py \
first_lcn_test \
/media/attic/chords/defs/base_model.definition \
/media/attic/chords/default.config \
/media/attic/chords/models \
/home/ejhumphrey/chords/chordrec_lcn_train0_20131002.dsf \
/home/ejhumphrey/chords/MIREX09_chord_map.txt

"""

import argparse
import numpy as np
import shutil

from marl.hewey import file
from marl.hewey import sources
from marl.hewey.core import Batch

from ejhumphrey.datasets import chordutils
from ejhumphrey.dnn.core.framework import Trainer
from ejhumphrey.dnn.utils import json_load
from ejhumphrey.datasets.utils import load_label_enum_map


def training_source(filepath, train_params, label_map, equivalence_map):
    """Open a DataSequenceFile for training.

    Parameters
    ----------
    filepath : str
        Path to a DataSequenceFile.
    train_params : dict
        Must contain at least a 'left' and 'right' key.
    """
    file_handle = file.DataSequenceFile(filepath)
    dset = sources.WeightedSampler(dataset=file_handle,
                                   left=train_params.get("left"),
                                   right=train_params.get("right"),
                                   label_map=label_map,
                                   equivalence_map=equivalence_map,
                                   weights={0:3, 1:1, 2:1, 3:1, 4:1, 5:1},
                                   refresh_prob=0.0,
                                   cache_size=650,
                                   MAX_LOCAL_INDEX=3000000)

    dset.set_value_shape(train_params.get("value_shape"))

    if train_params.get("transpose"):
        print "Randomly transposing chords is enabled."
        rotate_chord_batch = generate_rotation_function(
            label_map.get("N"), train_params.get("bins_per_octave"))
#        rotate_chord_batch = generate_rotate_and_mask_function(
#            label_map.get("N"), train_params.get("bins_per_octave"), 250)
        dset.set_transformer(rotate_chord_batch)

    return dset

#dset = sources.UniformSampler(dataset=file_handle,
#                                   left=train_params.get("left"),
#                                   right=train_params.get("right"),
#                                   label_map=label_map,
#                                   equivalence_map=equivalence_map,
#                                   refresh_prob=0.0,
#                                   cache_size=50,
#                                   MAX_LOCAL_INDEX=1000000)
##                                   MAX_LOCAL_INDEX=3000000)
#    return sources.SequenceSampler(dataset=file_handle,
#                                   left=train_params.get("left"),
#                                   right=train_params.get("right"),
#                                   refresh_prob=0.0,
#                                   cache_size=600)


def generate_rotation_function(no_chord_index, bins_per_octave):
    def rotate_chord_batch(batch):
        new_batch = Batch()
        for x, y in zip(batch.values, batch.labels):
            if y < 0:
                continue
            shp = x.shape
            x = x.squeeze()
            shift = np.random.randint(low= -12, high=13)
            xs, ys = chordutils.circshift_chord(
                x, y, shift, bins_per_octave, no_chord_index)
            new_batch.add_value(np.reshape(xs, newshape=shp))
            new_batch.add_label(ys)

        return new_batch
    return rotate_chord_batch

def generate_rotate_and_mask_function(no_chord_index, bins_per_octave, num_points):
    def distort_batch(batch):
        new_batch = Batch()
        for x, y in zip(batch.values, batch.labels):
            if y < 0:
                continue
            shp = x.shape
            x = x.squeeze()
            x *= chordutils.generate_cqt_mask(num_points,
                                              input_shape=x.shape,
                                              noise_shape=[9, 3],
                                              alpha=.5)
            shift = np.random.randint(low= -12, high=13)
            xs, ys = chordutils.circshift_chord(
                x, y, shift, bins_per_octave, no_chord_index)
            new_batch.add_value(np.reshape(xs, newshape=shp))
            new_batch.add_label(ys)

        return new_batch
    return distort_batch


def main(args):
    name = args.name
    trainer = Trainer(name=name,
                      save_directory=args.save_directory)

    # Copy over input files:
    shutil.copy(args.definition, trainer.save_directory)
    shutil.copy(args.config, trainer.save_directory)

    trainer.build_network(json_load(args.definition),
                          args.init_params)
    config = json_load(args.config)
    train_params = config.get("train_params")

    trainer.configure_losses(config.get("loss_tuples"))
    trainer.configure_updates(config.get("parameter_updates"))
    trainer.configure_constraints(config.get("constraints", {}))

    label_map = load_label_enum_map(args.label_map)
    equivalence_map = load_label_enum_map(args.equivalence_map)
    dset = training_source(args.training_datafile,
                           train_params,
                           label_map,
                           equivalence_map)


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

    parser.add_argument("equivalence_map",
                        metavar="equivalence_map", type=str,
                        help="JSON file mapping chords to equivalence classes.")

    parser.add_argument("--init_params", action="store", dest="init_params",
                        default='', help="Initial parameter pickle file.")

    main(parser.parse_args())
