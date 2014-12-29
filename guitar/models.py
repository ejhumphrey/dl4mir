import optimus

TIME_DIM = 20
NUM_FRETS = 9
GRAPH_NAME = "guitar"


def classifier_init(nodes):
    for n in nodes:
        for p in n.params.values():
            if 'classifier' in n.name and 'bias' in p.name:
                continue
            optimus.random_init(p)


def iXc3_nll(n_in, size='large', use_dropout=False):
    k0, k1, k2 = dict(
        small=(10, 20, 40),
        med=(12, 24, 48),
        large=(16, 32, 64),
        xlarge=(20, 40, 80),
        xxlarge=(24, 48, 96))[size]

    n0, n1, n2 = {
        1: (1, 1, 1),
        4: (3, 2, 1),
        8: (5, 3, 2),
        10: (3, 3, 1),
        20: (5, 5, 1)}[n_in]

    p0, p1, p2 = {
        1: (1, 1, 1),
        4: (1, 1, 1),
        8: (1, 1, 1),
        10: (2, 2, 1),
        12: (2, 2, 1),
        20: (2, 2, 2)}[n_in]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, n_in, 252))

    indexes = []
    for name in 'EADGBe':
        indexes.append(optimus.Input(
            name='{0}_index'.format(name),
            shape=(None,),
            dtype='int32'))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, learning_rate] + indexes

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, n0, 13),
        pool_shape=(p0, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, n1, 37),
        pool_shape=(p1, 1),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, n2, 33),
        pool_shape=(p2, 1),
        act_type='relu')

    dropout_edges = []
    if use_dropout:
        layer0.enable_dropout()
        layer1.enable_dropout()
        layer2.enable_dropout()
        inputs += [dropout]
        dropout_edges += [(dropout, layer0.dropout),
                          (dropout, layer1.dropout),
                          (dropout, layer2.dropout)]

    predictors = []
    softmaxes = []
    for name in 'EADGBe':
        predictors.append(optimus.Affine(
            name='{0}_predictor'.format(name),
            input_shape=layer2.output.shape,
            output_shape=(None, NUM_FRETS),
            act_type='linear'))
        softmaxes.append(optimus.Softmax('{0}_softmax'.format(name)))

    stack = optimus.Stack('stacker', num_inputs=6, axes=(1, 0, 2))

    param_nodes = [layer0, layer1, layer2] + predictors
    misc_nodes = [stack] + softmaxes

    # 1.1 Create Loss
    likelihoods = []
    logs = []
    neg_ones = []
    for name in 'EADGBe':
        likelihoods.append(
            optimus.SelectIndex(name='{0}_likelihood'.format(name)))

        logs.append(optimus.Log(name='{0}_log'.format(name)))
        neg_ones.append(optimus.Multiply(name='{0}_gain'.format(name),
                                         weight_shape=None))
        neg_ones[-1].weight.value = -1.0

    loss_sum = optimus.Add(name='loss_sum', num_inputs=6)
    ave_loss = optimus.Mean(name='ave_loss')
    loss_nodes = likelihoods + logs + neg_ones + [loss_sum, ave_loss]
    total_loss = optimus.Output(name='total_loss')

    fretboard = optimus.Output(name='fretboard')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input)]

    for p, smax in zip(predictors, softmaxes):
        base_edges += [
            (layer2.output, p.input),
            (p.output, smax.input),
        ]
    base_edges += [
        (softmaxes[0].output, stack.input_0),
        (softmaxes[1].output, stack.input_1),
        (softmaxes[2].output, stack.input_2),
        (softmaxes[3].output, stack.input_3),
        (softmaxes[4].output, stack.input_4),
        (softmaxes[5].output, stack.input_5),
        (stack.output, fretboard)
    ]

    trainer_edges = []
    for n, name in enumerate('EADGBe'):
        trainer_edges += [
            (softmaxes[n].output, likelihoods[n].input),
            (indexes[n], likelihoods[n].index),
            (likelihoods[n].output, logs[n].input),
            (logs[n].output, neg_ones[n].input)
        ]
    trainer_edges += [
        (neg_ones[0].output, loss_sum.input_0),
        (neg_ones[1].output, loss_sum.input_1),
        (neg_ones[2].output, loss_sum.input_2),
        (neg_ones[3].output, loss_sum.input_3),
        (neg_ones[4].output, loss_sum.input_4),
        (neg_ones[5].output, loss_sum.input_5),
        (loss_sum.output, ave_loss.input),
        (ave_loss.output, total_loss)
    ]

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=optimus.ConnectionManager(
            base_edges + trainer_edges).connections,
        outputs=[total_loss, fretboard],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    if use_dropout:
        layer0.disable_dropout()
        layer1.disable_dropout()
        layer2.disable_dropout()

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[fretboard],
        verbose=True)

    return trainer, predictor


MODELS = {
    'L': lambda: iXc3_nll(20, 'large'),
    'XL': lambda: iXc3_nll(20, 'xlarge'),
    'XXL': lambda: iXc3_nll(20, 'xxlarge'),
    'XXL_dropout': lambda: iXc3_nll(20, 'xxlarge', True),
    'XL_dropout': lambda: iXc3_nll(20, 'xlarge', True),
    'L_dropout': lambda: iXc3_nll(20, 'large', True)}
