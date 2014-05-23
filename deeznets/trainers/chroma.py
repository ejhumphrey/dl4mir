#!/usr/bin/env python
"""Train a deepnet for chroma representations

Sample Call:
python ejhumphrey/dnn/trainers/chroma.py \
chroma_1L_v061_F0 \
/Volumes/Audio/Chord_Recognition/defs/chroma_1L_RBF_v061.definition \
/Volumes/Audio/Chord_Recognition/chroma_L1.config \
/Volumes/Audio/Chord_Recognition/models/ \
/Volumes/speedy/onset_sync_chords_train0.dsf \
/Volumes/Audio/Chord_Recognition/TMC2013_chord_map_061_wBB.txt \
/Volumes/Audio/Chord_Recognition/TMC2013_chord_map_061_wBB_equivalence.txt \
--init_params=/Users/ejhumphrey/dltut/chroma_templates_v061.pk
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
    unique_qualities = set([int(l / 12) for l in label_map.values()])
    if -1 in unique_qualities:
        unique_qualities.remove(-1)
    weights = dict([(q, 1) for q in unique_qualities])

    file_handle = file.DataSequenceFile(filepath)
    dset = sources.WeightedSampler(dataset=file_handle,
                                   left=train_params.get("left"),
                                   right=train_params.get("right"),
                                   label_map=label_map,
                                   equivalence_map=equivalence_map,
                                   weights=weights,
                                   refresh_prob=0.0,
                                   cache_size=600,
                                   MAX_LOCAL_INDEX=2000000)

    dset.set_value_shape(train_params.get("value_shape"))

    if train_params.get("transpose"):
        print "Randomly transposing chords is enabled."
        rotate_chord_batch = generate_rotation_function(
            label_map.get("N"), train_params.get("bins_per_octave"))
        dset.set_transformer(rotate_chord_batch)

    return dset

def generate_rotation_function(no_chord_index, bins_per_octave):
    def rotate_chord_batch(batch):
        new_batch = Batch()
        for x, y in zip(batch.values, batch.labels):
            if y < 0:
                continue
            shp = x.shape
            shift = np.random.randint(low= -12, high=13)
            xs, ys = chordutils.circshift_chord(
                x, y, shift, bins_per_octave, no_chord_index)
            new_batch.add_value(np.reshape(xs, newshape=shp))
            new_batch.add_label(ys)

        return new_batch
    return rotate_chord_batch

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
