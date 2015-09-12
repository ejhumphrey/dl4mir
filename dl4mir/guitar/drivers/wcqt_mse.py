"""write meeee"""
import argparse
import biggie
import optimus
from os import path
import json

import dl4mir.chords.data as D
import dl4mir.chords.pipefxs as FX
import dl4mir.common.streams as S
from dl4mir.chords import DRIVER_ARGS

TIME_DIM = 20
VOCAB = 157
LEARNING_RATE = 0.1
BATCH_SIZE = 50
OCTAVE_DIM = 6
PITCH_DIM = 40

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB


def build_model():
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, OCTAVE_DIM, TIME_DIM, PITCH_DIM))

    fret_map = optimus.Input(
        name='fret_map',
        shape=(None, 6, 9))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(32, None, 5, 5),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(64, None, 5, 7),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(128, None, 3, 6),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 1024,),
        act_type='relu')

    strings = []
    for n in range(6):
        strings.append(
            optimus.Affine(
                name='string_%d' % n,
                input_shape=layer3.output.shape,
                output_shape=(None, 9),
                act_type='sigmoid'))

    param_nodes = [layer0, layer1, layer2, layer3] + strings

    # 1.1 Create Loss
    stack = optimus.Stack('stack', axes=[1, 0, 2])
    error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mse')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, strings[0].input),
        (strings[0].output, stack.input_list),
        (layer3.output, strings[1].input),
        (strings[1].output, stack.input_list),
        (layer3.output, strings[2].input),
        (strings[2].output, stack.input_list),
        (layer3.output, strings[3].input),
        (strings[3].output, stack.input_list),
        (layer3.output, strings[4].input),
        (strings[4].output, stack.input_list),
        (layer3.output, strings[5].input),
        (strings[5].output, stack.input_list)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (stack.output, error.input_a),
            (fret_map, error.input_b),
            (error.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, fret_map, learning_rate],
        nodes=param_nodes + [stack, error, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        optimus.random_init(n.weights)
        optimus.random_init(n.bias)

    fret_posterior = optimus.Output(
        name='fret_posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(stack.output, fret_posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + [stack],
        connections=predictor_edges.connections,
        outputs=[fret_posterior])
    return trainer, predictor


def main(args):
    trainer, predictor = build_model()

    if args.init_param_file:
        print "Loading parameters: %s" % args.init_param_file
        trainer.load_param_values(args.init_param_file)

    optimus.random_init(trainer.params['layer3'].weights)
    optimus.random_init(trainer.params['layer3'].bias)

    # 3. Create Data
    print "Loading %s" % args.training_file
    stash = biggie.Stash(args.training_file)
    stream = D.create_stash_stream(
        stash, TIME_DIM, pitch_shift=0, vocab_dim=VOCAB, pool_size=25)

    if args.secondary_source:
        stash2 = biggie.Stash(args.secondary_source)
        stream2 = D.create_uniform_chord_stream(
            stash2, TIME_DIM, pitch_shift=0, vocab_dim=VOCAB, working_size=5)
        stream = S.mux([stream, stream2], [0.5, 0.5])

    stream = S.minibatch(stream, batch_size=BATCH_SIZE)

    print "Starting '%s'" % args.trial_name
    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    predictor_file = path.join(driver.output_directory, args.predictor_file)
    optimus.save(predictor, def_file=predictor_file)

    hyperparams = dict(learning_rate=LEARNING_RATE)
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
    parser.add_argument("--secondary_source",
                        metavar="--secondary_source", type=str, default='',
                        help="Path to a secondary stash to use for training.")
    main(parser.parse_args())
