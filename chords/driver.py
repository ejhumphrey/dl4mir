"""write meeee"""
import argparse
import biggie
import optimus
from os import path
import marl.fileutils as futil
import numpy as np
import json

import dl4mir.chords.data as D
import dl4mir.common.streams as S
from dl4mir.chords import DRIVER_ARGS
from dl4mir.chords import models
import dl4mir.chords.lexicon as lex

DRIVER_ARGS['max_iter'] = 500000
VOCAB = lex.Strict(157)
LEARNING_RATE = 0.02
BATCH_SIZE = 50


def main(args):
    arch_key = args.arch_size
    if args.dropout:
        arch_key += '_dropout'

    trainer, predictor = models.MODELS[arch_key]()
    time_dim = trainer.inputs['data'].shape[2]

    if args.init_param_file:
        print "Loading parameters: %s" % args.init_param_file
        trainer.load_param_values(args.init_param_file)

    print "Opening %s" % args.training_file
    stash = biggie.Stash(args.training_file, cache=True)
    stream = D.create_chord_index_stream(
        stash, time_dim, max_pitch_shift=0, lexicon=VOCAB)

    # Load prior
    stat_file = "%s.json" % path.splitext(args.training_file)[0]
    prior = np.array(json.load(open(stat_file))['prior'], dtype=float)
    trainer.nodes['prior'].weight.value = 1.0 / prior.reshape(1, -1)

    stream = S.minibatch(stream, batch_size=BATCH_SIZE)

    print "Starting '%s'" % args.trial_name
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.output_directory)

    hyperparams = dict(learning_rate=LEARNING_RATE)
    if args.dropout:
        hyperparams.update(dropout=args.dropout)

    futil.create_directory(args.output_directory)
    predictor_file = path.join(driver.output_directory, args.predictor_file)
    print predictor_file
    return
    optimus.save(predictor, def_file=predictor_file)
    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to a biggie Stash file for training.")
    parser.add_argument("arch_size",
                        metavar="arch_size", type=str,
                        help="Size of the architecture, one of {X, XL, XXL}")
    parser.add_argument("dropout",
                        metavar="dropout", type=float,
                        help="Dropout parameter")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the training results.")
    parser.add_argument("trial_name",
                        metavar="trial_name", type=str,
                        help="Unique name for this training run.")
    parser.add_argument("predictor_file",
                        metavar="predictor_file", type=str,
                        help="Name for the resulting predictor graph.")
    parser.add_argument("--init_param_file",
                        metavar="--init_param_file", type=str, default='',
                        help="Path to a NPZ archive for initialization the "
                        "parameters of the graph.")
    main(parser.parse_args())
