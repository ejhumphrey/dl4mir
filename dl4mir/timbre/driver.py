"""Primary training driver for the NLSE models."""

from __future__ import print_function
import argparse
import biggie
import optimus
from os import path

import dl4mir.common.fileutil as futil
import dl4mir.common.streams as S
import dl4mir.timbre.data as D
from dl4mir.timbre import models

DRIVER_ARGS = dict(
    max_iter=25010,
    save_freq=1000,
    print_freq=50,
    nan_exceptions=100)
LEARNING_RATE = 0.02
BATCH_SIZE = 100
RADIUS = 12 ** 0.5


def main(args):
    sim_margin = -RADIUS * args.margin
    trainer, predictor, zerofilter = models.iX_c3f2_oY(20, 3, 'xlarge')
    time_dim = trainer.inputs['cqt'].shape[2]

    if args.init_param_file:
        print("Loading parameters: {0}".format(args.init_param_file))
        trainer.load_param_values(args.init_param_file)

    print("Opening {0}".format(args.training_file))
    stash = biggie.Stash(args.training_file, cache=True)
    stream = S.minibatch(
        D.create_pairwise_stream(stash, time_dim,
                                 working_size=100, threshold=0.05),
        batch_size=BATCH_SIZE)

    stream = D.batch_filter(
        stream, zerofilter, threshold=2.0**-16, min_batch=1,
        max_consecutive_skips=100, sim_margin=sim_margin, diff_margin=RADIUS)

    print("Starting '{0}'".format(args.trial_name))
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=futil.create_directory(args.output_directory))

    hyperparams = dict(
        learning_rate=LEARNING_RATE,
        sim_margin=sim_margin, diff_margin=RADIUS)

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)

    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to a biggie Stash file for training.")
    parser.add_argument("margin",
                        metavar="margin", type=float,
                        help="Margin parameter")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the .")
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
