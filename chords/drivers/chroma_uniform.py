"""write meeee"""
import argparse
import biggie
import optimus
from os import path

import dl4mir.chords.data as D
import dl4mir.common.streams as S
from dl4mir.chords import DRIVER_ARGS

TIME_DIM = 20
LEARNING_RATE = 0.01

# Other code depends on this.
GRAPH_NAME = "chroma"


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, 252))

    target_chroma = optimus.Input(
        name='target_chroma',
        shape=(None, 12),)

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

    layer4 = optimus.Affine(
        name='layer4',
        input_shape=layer3.output.shape,
        output_shape=(None, 12,),
        act_type='sigmoid')

    all_nodes = [layer0, layer1, layer2, layer3, layer4]

    # 1.1 Create Losses
    chroma_xentropy = optimus.CrossEntropy(
        name="chroma_xentropy")

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, layer4.input),
        (layer4.output, chroma_xentropy.prediction),
        (target_chroma, chroma_xentropy.target)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, layer1.weights),
        (learning_rate, layer1.bias),
        (learning_rate, layer2.weights),
        (learning_rate, layer2.bias),
        (learning_rate, layer3.weights),
        (learning_rate, layer3.bias),
        (learning_rate, layer4.weights),
        (learning_rate, layer4.bias)])

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target_chroma, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chroma_xentropy],
        updates=update_manager.connections)

    optimus.random_init(layer0.weights)
    optimus.random_init(layer1.weights)
    optimus.random_init(layer2.weights)
    optimus.random_init(layer3.weights)
    optimus.random_init(layer4.weights)

    validator = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target_chroma],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chroma_xentropy])

    chroma_out = optimus.Output(
        name='chroma')

    predictor_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, layer4.input),
        (layer4.output, chroma_out)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=all_nodes,
        connections=predictor_edges.connections,
        outputs=[chroma_out])

    # 3. Create Data
    stash = biggie.Stash(args.training_file)
    stream = S.minibatch(
        D.uniform_quality_chroma_stream(stash, TIME_DIM),
        batch_size=50)

    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

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
