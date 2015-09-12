"""write meeee"""
import argparse
import biggie
import optimus
import numpy as np
from os import path

import dl4mir.chords.data as D
import dl4mir.common.streams as S
import dl4mir.common.util as util

import random
import itertools

TIME_DIM = 20
VOCAB = 157
LEARNING_RATE = 0.002

DRIVER_ARGS = dict(
    save_freq=250,
    print_freq=50)

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB

def stream_averages(mu=0, std=0.1, dropout=0.25, min_scale=0.25):
    averages = np.load("/home/ejhumphrey/Dropbox/tmp/chord_averages0.npy")
    while True:
        for chord_idx in range(157):
            cqt = averages[chord_idx]
            cqt = cqt + np.random.normal(mu, std, cqt.shape)
            cqt = cqt * np.random.binomial(1, 1.0 - dropout, size=cqt.shape)
            cqt = cqt * np.random.uniform(min_scale, 1.0)
            yield biggie.Entity(cqt=cqt, chord_idx=chord_idx)

def mux(streams, weights=None):
    if weights is None:
        weights = np.ones(len(streams), dtype=float) / len(streams)
    while True:
        idx = np.flatnonzero(np.random.multinomial(1, weights))[0]
        yield streams[idx].next()


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, 252))

    chord_idx = optimus.Input(
        name='chord_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(12, 1, 9, 19),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(16, None, 7, 15),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 6, 15),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 512,),
        act_type='relu')

    chord_classifier = optimus.Softmax(
        name='chord_classifier',
        input_shape=layer3.output.shape,
        n_out=VOCAB,
        act_type='linear')

    all_nodes = [layer0, layer1, layer2, layer3, chord_classifier]

    # 1.1 Create Losses
    chord_nll = optimus.NegativeLogLikelihood(
        name="chord_nll")

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input),
        (chord_classifier.output, chord_nll.likelihood),
        (chord_idx, chord_nll.target_idx)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, layer1.weights),
        (learning_rate, layer1.bias),
        (learning_rate, layer2.weights),
        (learning_rate, layer2.bias),
        (learning_rate, layer3.weights),
        (learning_rate, layer3.bias),
        (learning_rate, chord_classifier.weights),
        (learning_rate, chord_classifier.bias)])

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_nll],
        updates=update_manager.connections)

    optimus.random_init(chord_classifier.weights)

    validator = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_nll])

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

    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

    # 3. Create Data
    stream = S.minibatch(stream_averages(), batch_size=100)
    # driver.fit(stream, hyperparams=hyperparams, max_iter=500, **DRIVER_ARGS)

    # stream = S.minibatch(stream_averages(std=0.25, dropout=0.5, min_scale=.1),
    #                      batch_size=157)
    # driver.fit(stream, hyperparams=hyperparams, max_iter=250, **DRIVER_ARGS)

    stash = biggie.Stash(args.training_file)
    real_stream = S.minibatch(
        D.create_uniform_quality_stream(stash, TIME_DIM, vocab_dim=VOCAB),
        batch_size=100)

    streams = mux([stream, real_stream], [0.5, 0.5])
    hyperparams = {learning_rate.name: LEARNING_RATE}
    driver.fit(streams, hyperparams=hyperparams, max_iter=10000, **DRIVER_ARGS)

    # streams = mux([stream, real_stream], [0.5, 0.5])
    # driver.fit(streams, hyperparams=hyperparams, max_iter=3000, **DRIVER_ARGS)

    # streams = mux([stream, real_stream], [0.25, 0.75])
    # driver.fit(streams, hyperparams=hyperparams, max_iter=4000, **DRIVER_ARGS)

    validator_file = path.join(driver.output_directory, args.validator_file)
    optimus.save(validator, def_file=validator_file)

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)


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
    parser.add_argument("validator_file",
                        metavar="validator_file", type=str,
                        help="Name for the resulting validator graph.")
    parser.add_argument("predictor_file",
                        metavar="predictor_file", type=str,
                        help="Name for the resulting predictor graph.")
    main(parser.parse_args())
