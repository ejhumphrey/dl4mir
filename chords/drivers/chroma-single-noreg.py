"""write meeee"""
import argparse
import biggie
import optimus
from os import path

import dl4mir.chords.data as D
import dl4mir.common.streams as S
from dl4mir.chords import DRIVER_ARGS
from dl4mir.chords import models

DRIVER_ARGS['max_iter'] = 200000
LEARNING_RATE = 0.1
BATCH_SIZE = 100


def main(args):
    trainer, predictor = models.MODELS[args.model_name]()
    time_dim = trainer.inputs['cqt'].shape[2]

    if args.init_param_file:
        print "Loading parameters: %s" % args.init_param_file
        trainer.load_param_values(args.init_param_file)

    print "Opening %s" % args.training_file
    stash = biggie.Stash(args.training_file, cache=True)
    stream = S.minibatch(
        D.create_chroma_stream(stash, time_dim, pitch_shift=0),
        batch_size=BATCH_SIZE)

    print "Starting '%s'" % args.trial_name
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.output_directory)

    hyperparams = dict(learning_rate=LEARNING_RATE)

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)

    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to an optimus file for training.")
    parser.add_argument("model_name",
                        metavar="model_name", type=str,
                        help="Name of a pre-defined model.")
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
    parser.add_argument("--secondary_source",
                        metavar="--secondary_source", type=str, default='',
                        help="Path to a secondary stash to use for training.")
    main(parser.parse_args())
