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


def wcqt_nll_margin():
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

    margin = optimus.Input(
        name='margin',
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
        act_type='sigmoid')

    param_nodes = [layer0, layer1, layer2, layer3, chord_classifier]

    # 1.1 Create Loss
    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='neg_one0')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name="moia_values")

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Accumulate(name='summer')

    relu = optimus.RectifiedLinear(name='relu')
    loss = optimus.Mean(name='margin_loss')

    target_vals = optimus.Output(
        name='target_vals')

    moia_vals = optimus.Output(
        name='moia_vals')

    summer_vals = optimus.Output(
        name='summer_vals')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_classifier.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (margin, summer.input_list),
            (target_values.output, summer.input_list),
            (target_values.output, target_vals),
            (moia_values.output, neg_one1.input),
            (moia_values.output, moia_vals),
            (neg_one1.output, summer.input_list),
            (summer.output, relu.input),
            (summer.output, summer_vals),
            (relu.output, loss.input)])

    updates = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, margin],
        nodes=param_nodes + [log, neg_one0, target_values, moia_values,
                             neg_one1, summer, relu, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output, chord_classifier.output, target_vals, moia_vals, summer_vals],
        loss=loss.output,
        updates=updates.connections,
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

    # update_manager = optimus.ConnectionManager(
    #     map(lambda n: (learning_rate, n.weights), param_nodes) +
    #     map(lambda n: (learning_rate, n.bias), param_nodes))
    update_manager = optimus.ConnectionManager(
        [(learning_rate, chord_estimator.bias)])

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


def wcqt_likelihood_wmoia(n_dim=VOCAB):
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

    moia_weight = optimus.Input(
        name='moia_weight',
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
    main_loss = optimus.Mean(name='mean_squared_error')
    loss_nodes1 = [likelihoods, dimshuffle, error, main_loss]

    negone = optimus.Gain(name='negate')
    negone.weight.value = -1.0
    summer = optimus.Accumulate(name='moia_sum')
    flatten = optimus.Sum('flatten', axis=1)
    dimshuffle2 = optimus.Dimshuffle('dimshuffle2', (0, 'x'))
    margin = optimus.RectifiedLinear(name='margin')
    weight = optimus.Multiply(name="margin_weight")
    margin_loss = optimus.Mean(name='margin_loss', axis=None)

    loss_nodes2 = [negone, summer, margin, flatten,
                   dimshuffle2, margin_loss, weight]
    total_loss = optimus.Accumulate("total_loss")

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
            (error.output, main_loss.input),
            # Margin loss
            (dimshuffle.output, negone.input),
            (negone.output, summer.input_list),
            (chord_estimator.output, summer.input_list),
            (summer.output, margin.input),
            (margin.output, flatten.input),
            (flatten.output, dimshuffle2.input),
            (dimshuffle2.output, weight.input_a),
            (target, weight.input_b),
            (weight.output, margin_loss.input),
            (margin_loss.output, total_loss.input_list),
            (main_loss.output, total_loss.input_list)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    all_nodes = param_nodes + loss_nodes1 + loss_nodes2 + [total_loss]
    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, chord_idx, learning_rate],
        nodes=all_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss.output],
        loss=total_loss.output,
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
    'wcqt_likelihood': wcqt_likelihood,
    'wcqt_likelihood_wmoia': wcqt_likelihood_wmoia}
