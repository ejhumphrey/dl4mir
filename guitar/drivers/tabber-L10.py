"""write meeee"""
import argparse
import optimus
from os import path
from ejhumphrey.dl4mir.guitar import transformers as T
from ejhumphrey.dl4mir.guitar import SOURCE_ARGS, DRIVER_ARGS

TIME_DIM = 10
FRET_DIM = 12
MAX_FRETS = FRET_DIM - 1
LEARNING_RATE = 0.002
GRAPH_NAME = "tabber"


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, 252))

    fret_bitmap = optimus.Input(
        name='fret_bitmap',
        shape=(None, 6, FRET_DIM))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(12, 1, 5, 19),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(16, None, 5, 15),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 2, 15),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 512,),
        act_type='relu')

    fretboard = optimus.MultiSoftmax(
        name='fretboard',
        input_shape=layer3.output.shape,
        output_shape=(None, 6, FRET_DIM),
        act_type='linear')

    all_nodes = [layer0, layer1, layer2, layer3, fretboard]

    # 1.1 Create Losses
    mse = optimus.MeanSquaredError(
        name="mean_squared_error")

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, fretboard.input),
        (fretboard.output, mse.prediction),
        (fret_bitmap, mse.target)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, layer1.weights),
        (learning_rate, layer1.bias),
        (learning_rate, layer2.weights),
        (learning_rate, layer2.bias),
        (learning_rate, layer3.weights),
        (learning_rate, layer3.bias),
        (learning_rate, fretboard.weights),
        (learning_rate, fretboard.bias)])

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, fret_bitmap, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[mse],
        updates=update_manager.connections)

    optimus.random_init(fretboard.weights)

    validator = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, fret_bitmap],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[mse])

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, fretboard.input),
        (fretboard.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=all_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    # 3. Create Data
    source = optimus.Queue(
        optimus.File(args.training_file),
        transformers=[
            T.cqt_sample(input_data.shape[2]),
            T.pitch_shift(MAX_FRETS, bins_per_pitch=3),
            T.fret_indexes_to_bitmap(FRET_DIM)],
        **SOURCE_ARGS)

    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

    driver.fit(source, hyperparams=hyperparams, **DRIVER_ARGS)

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
