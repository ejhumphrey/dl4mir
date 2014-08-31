"""write meeee"""
import argparse
import biggie
import optimus
from os import path
import numpy as np
import json

import dl4mir.chords.data as D
import dl4mir.chords.pipefxs as FX
import dl4mir.common.streams as S
from dl4mir.chords import DRIVER_ARGS

TIME_DIM = 20
VOCAB = 157
LEARNING_RATE = 0.02
BATCH_SIZE = 50
PITCH_DIM = 252

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, PITCH_DIM))

    target = optimus.Input(
        name='target',
        shape=(None, VOCAB))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(32, 1, 5, 19),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(64, None, 5, 15),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(128, None, 3, 15),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 1024,),
        act_type='relu')

    chord_classifier = optimus.Affine(
        name='chord_classifier',
        input_shape=layer3.output.shape,
        output_shape=(None, VOCAB,),
        act_type='sigmoid')

    all_nodes = [layer0, layer1, layer2, layer3, chord_classifier]

    # 1.1 Create Losses
    chord_mse = optimus.MeanSquaredError(
        name="chord_mse")

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input),
        (chord_classifier.output, chord_mse.prediction),
        (target, chord_mse.target)])

    update_manager = optimus.ConnectionManager([
        # (learning_rate, layer0.weights),
        # (learning_rate, layer0.bias),
        # (learning_rate, layer1.weights),
        # (learning_rate, layer1.bias),
        # (learning_rate, layer2.weights),
        # (learning_rate, layer2.bias),
        (learning_rate, layer3.weights),
        (learning_rate, layer3.bias),
        (learning_rate, chord_classifier.weights),
        (learning_rate, chord_classifier.bias)])

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_mse],
        updates=update_manager.connections)

    for n in all_nodes:
        optimus.random_init(n.weights, 0, 0.01)
        optimus.random_init(n.bias, 0, 0.01)

    if args.init_param_file:
        param_values = dict(np.load(args.init_param_file))
        keys = param_values.keys()
        for key in keys:
            if chord_classifier.name in key or layer.name in key:
                print "skipping %s" % key
                del param_values[key]
        trainer.param_values = param_values


    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input),
        (chord_classifier.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=all_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    # 3. Create Data
    print "Loading %s" % args.training_file
    stash = biggie.Stash(args.training_file)
    stream = D.create_uniform_chord_stream(
        stash, TIME_DIM, pitch_shift=0, vocab_dim=VOCAB, working_size=10)
    stream = S.minibatch(
        FX.chord_index_to_affinity_vectors(stream, vocab_dim=VOCAB),
        batch_size=BATCH_SIZE)

    print "Starting '%s'" % args.trial_name
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)

    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to an optimus file for training.")
    # Outputs
    parser.add_argument("model_directory",
                        metavar="model_directory", type=str,
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
