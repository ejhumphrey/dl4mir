import optimus

TIME_DIM = 20
VOCAB = 157

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB


def wcqt_nll():
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 6, TIME_DIM, 40))

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

    chord_classifier = optimus.Affine(
        name='chord_classifier',
        input_shape=layer3.output.shape,
        output_shape=(None, VOCAB),
        act_type='softmax')

    param_nodes = [layer0, layer1, layer2, layer3, chord_classifier]

    # 1.1 Create Loss
    target_values = optimus.SelectIndex(name='target_values')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_classifier.output, target_values.input),
            (chord_idx, target_values.index),
            (target_values.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + [target_values, log, neg, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_classifier.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def wcqt_sigmoid_mse(n_dim=VOCAB):
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 6, TIME_DIM, 40))

    target = optimus.Input(
        name='target',
        shape=(None, n_dim))

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

    chord_estimator = optimus.Affine(
        name='chord_estimator',
        input_shape=layer3.output.shape,
        output_shape=(None, n_dim),
        act_type='sigmoid')

    param_nodes = [layer0, layer1, layer2, layer3, chord_estimator]

    # 1.1 Create Loss
    error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mean_squared_error')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_estimator.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_estimator.output, error.input_a),
            (target, error.input_b),
            (error.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, learning_rate],
        nodes=param_nodes + [error, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_estimator.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def wcqt_likelihood(n_dim=VOCAB):
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 6, TIME_DIM, 40))

    target = optimus.Input(
        name='target',
        shape=(None, 1))

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

    chord_estimator = optimus.Affine(
        name='chord_estimator',
        input_shape=layer3.output.shape,
        output_shape=(None, n_dim),
        act_type='sigmoid')

    param_nodes = [layer0, layer1, layer2, layer3, chord_estimator]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex('select')
    dimshuffle = optimus.Dimshuffle('dimshuffle', (0, 'x'))
    error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mean_squared_error')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_estimator.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_estimator.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, dimshuffle.input),
            (dimshuffle.output, error.input_a),
            (target, error.input_b),
            (error.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, chord_idx, learning_rate],
        nodes=param_nodes + [likelihoods, dimshuffle, error, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_estimator.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


MODELS = {
    'wcqt_nll': wcqt_nll,
    'wcqt_sigmoid_mse': wcqt_sigmoid_mse,
    'wcqt_likelihood': wcqt_likelihood}
