"""write meeee"""
import argparse
import biggie
import optimus
from os import path
import marl.fileutils as futil

import dl4mir.timbre.data as D
import dl4mir.common.streams as S
from dl4mir.timbre import DRIVER_ARGS
from dl4mir.timbre import models

DRIVER_ARGS['max_iter'] = 500000
LEARNING_RATE = 0.02
BATCH_SIZE = 100


def main(args):
    trainer, predictor = models.iX_c3f2_oY(10, 3, 'large')
    time_dim = trainer.inputs['data'].shape[2]

    if args.init_param_file:
        print "Loading parameters: %s" % args.init_param_file
        trainer.load_param_values(args.init_param_file)

    print "Opening %s" % args.training_file
    stash = biggie.Stash(args.training_file, cache=True)
    stream = S.minibatch(
        D.create_pairwise_stream(stash, time_dim),
        batch_size=BATCH_SIZE)

    print "Starting '%s'" % args.trial_name
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=futil.create_directory(args.output_directory))

    hyperparams = dict(learning_rate=LEARNING_RATE, margin=float(args.margin))

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)

    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to a biggie Stash file for training.")
    # parser.add_argument("arch_size",
    #                     metavar="arch_size", type=str, default='large',
    #                     help="Size of the architecture.")
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
