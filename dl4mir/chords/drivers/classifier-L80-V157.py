"""write meeee"""
import argparse
import optimus
from os import path
from ejhumphrey.dl4mir.chords import transformers as T

VOCAB = 157
LEARNING_RATE = 0.002

DRIVER_ARGS = dict(
    max_iter=10000,
    save_freq=250,
    print_freq=5)

SOURCE_ARGS = dict(
    batch_size=50,
    refresh_prob=0.1,
    cache_size=500)


def main(args):
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 80, 252))

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
        weight_shape=(12, 1, 13, 19),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(16, None, 11, 15),
        pool_shape=(2, 1),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 9, 15),
        pool_shape=(4, 1),
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
        name='chord_classifier',
        inputs=[input_data, chord_idx, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[optimus.Graph.TOTAL_LOSS],
        losses=[chord_nll],
        updates=update_manager.connections)

    optimus.random_init(chord_classifier.weights)

    predictor_edges = optimus.ConnectionManager([
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input)])

    predictor = optimus.Graph(
        name='chord_classifier',
        inputs=[input_data],
        nodes=all_nodes,
        connections=predictor_edges.connections,
        outputs=[chord_classifier.output])

    # 3. Create Data
    source = optimus.Queue(
        optimus.File(args.training_file),
        transformers=[T.sample(input_data.shape[2]), T.map_to_index(VOCAB)],
        **SOURCE_ARGS)

    driver = optimus.Driver(
        graph=trainer,
        name=args.trial_name,
        output_directory=args.model_directory)

    hyperparams = {learning_rate.name: LEARNING_RATE}

    driver.fit(source, hyperparams=hyperparams, **DRIVER_ARGS)

    transform_file = path.join(driver.output_directory, args.transform_file)
    optimus.save(predictor, def_file=transform_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to an optimus file to predict.")
    # Outputs
    parser.add_argument("model_directory",
                        metavar="model_directory", type=str,
                        help="Path to save the training results.")
    parser.add_argument("trial_name",
                        metavar="trial_name", type=str,
                        help="Unique name for this training run.")
    parser.add_argument("transform_file",
                        metavar="transform_file", type=str,
                        help="Optimus parameter archive matching the given "
                        "graph defintion.")
    main(parser.parse_args())
