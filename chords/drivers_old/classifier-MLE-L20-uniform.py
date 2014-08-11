"""write meeee"""
import argparse
import biggie
import optimus
from os import path

import dl4mir.chords.data as D
import dl4mir.common.streams as S

from dl4mir.chords import DRIVER_ARGS

TIME_DIM = 20
VOCAB = 157
LEARNING_RATE = 0.02

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, 252))

    chord_idx = optimus.Input(
        name='chord_idx',
        shape=(None,),
        dtype='int32')

    is_chord = optimus.Input(
        name='is_chord',
        shape=(None,))

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

    chord_estimator = optimus.Affine(
        name='chord_estimator',
        input_shape=layer3.output.shape,
        output_shape=(None, VOCAB),
        act_type='sigmoid')

    all_nodes = [layer0, layer1, layer2, layer3, chord_estimator]

    # 1.1 Create Losses
    chord_mse = optimus.SparseMeanSquaredError(
    # chord_mse = optimus.SparseCrossEntropy(
        name="chord_mse")

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_estimator.input),
        (chord_estimator.output, chord_mse.prediction),
        (chord_idx, chord_mse.index),
        (is_chord, chord_mse.target)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, layer1.weights),
        (learning_rate, layer1.bias),
        (learning_rate, layer2.weights),
        (learning_rate, layer2.bias),
        (learning_rate, layer3.weights),
        (learning_rate, layer3.bias),
        (learning_rate, chord_estimator.weights),
        (learning_rate, chord_estimator.bias)])

    print "Building trainer"
    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, is_chord, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_mse],
        updates=update_manager.connections)

    optimus.random_init(chord_estimator.weights)

    print "Building validator"
    validator = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, is_chord],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_mse])

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_estimator.input),
        (chord_estimator.output, posterior)])

    print "Building predictor"
    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=all_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    # 3. Create Data
    print "Opening Data"
    stash = biggie.Stash(args.training_file)
    stream = S.minibatch(
        D.create_contrastive_quality_stream(stash, TIME_DIM, vocab_dim=VOCAB),
        batch_size=50)

    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

    print "...aaand we're off!"
    driver.fit(stream, hyperparams=hyperparams, **DRIVER_ARGS)

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
