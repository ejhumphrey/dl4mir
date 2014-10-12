import optimus
import dl4mir.chords.labels as L
from sklearn.decomposition import PCA
import numpy as np

TIME_DIM = 20
VOCAB = 157

# Other code depends on this.
GRAPH_NAME = "classifier-V%03d" % VOCAB


def classifier_init(nodes):
    for n in nodes:
        for p in n.params.values():
            if 'classifier' in n.name and 'bias' in p.name:
                continue
            optimus.random_init(p)


def i8c4b10_nll_dropout(size='large'):
    k0, k1, k2 = dict(
        large=(24, 48, 64))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 8, 252))

    chord_idx = optimus.Input(
        name='chord_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, 3, 13),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 3, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 3, 33),
        act_type='relu')

    layer3 = optimus.Conv3D(
        name='layer3',
        input_shape=layer2.output.shape,
        weight_shape=(10, None, 2, 1),
        act_type='relu')

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer3.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer3.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, layer3,
                   null_classifier, chord_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    layer0.enable_dropout()
    layer1.enable_dropout()
    layer2.enable_dropout()

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_classifier.input),
        (layer3.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (dropout, layer0.dropout),
            (dropout, layer1.dropout),
            (dropout, layer2.dropout),
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes[:-1]) +
        map(lambda n: (learning_rate, n.bias), param_nodes[:-1]))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, dropout],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes[:-1])

    semitones = L.semitone_matrix(157)[:13, 2:]
    chord_classifier.weights.value = semitones.reshape(13, 10, 1, 1)

    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])

    layer0.disable_dropout()
    layer1.disable_dropout()
    layer2.disable_dropout()

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


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
        name='data',
        shape=(None, 1, n_in, 252))

    chord_idx = optimus.Input(
        name='class_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, chord_idx, learning_rate]

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

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')
    prior = optimus.Multiply("prior", weight_shape=(1, 157), broadcast=[0])
    prior.weight.value = np.ones([1, 157])

    param_nodes = [layer0, layer1, layer2, null_classifier, chord_classifier]
    misc_nodes = [flatten, cat, softmax, prior]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Multiply(name='gain', weight_shape=None)
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    posterior = optimus.Output(name='posterior')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer2.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input),
        (softmax.output, prior.input),
        (prior.output, posterior)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + dropout_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
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
        outputs=[posterior],
        verbose=True)

    return trainer, predictor


def iXc3_nll2(n_in, size='large', use_dropout=False):
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
        name='data',
        shape=(None, 1, n_in, 252))

    chord_idx = optimus.Input(
        name='class_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, chord_idx, learning_rate]

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

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer0.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')
    prior = optimus.Multiply("prior", weight_shape=(1, 157), broadcast=[0])
    prior.weight.value = np.ones([1, 157])

    param_nodes = [layer0, layer1, layer2, null_classifier, chord_classifier]
    misc_nodes = [flatten, cat, softmax, prior]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Multiply(name='gain', weight_shape=None)
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    posterior = optimus.Output(name='posterior')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer0.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input),
        (softmax.output, prior.input),
        (prior.output, posterior)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + dropout_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
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
        outputs=[posterior],
        verbose=True)

    return trainer, predictor


def i8x1c3T_nll2(size, use_dropout=False):
    k0, k1, k2 = dict(
        small=(10, 20, 40),
        med=(12, 24, 48),
        large=(16, 32, 64),
        xlarge=(20, 40, 80),
        xxlarge=(24, 48, 96))[size]

    n0, n1, n2 = (3, 3, 4)
    p0, p1, p2 = (1, 1, 1)

    input_data = optimus.Input(
        name='data',
        shape=(None, 8, 1, 252))

    chord_idx = optimus.Input(
        name='class_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, chord_idx, learning_rate]

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    # 1.2 Create Nodes
    transpose = optimus.Dimshuffle('rearrange', axes=(0, 2, 1, 3))

    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=(None, 1, 8, 252),
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

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer0.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')
    prior = optimus.Multiply("prior", weight_shape=(1, 157), broadcast=[0])
    prior.weight.value = np.ones([1, 157])

    param_nodes = [layer0, layer1, layer2, null_classifier, chord_classifier]
    misc_nodes = [transpose, flatten, cat, softmax, prior]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Multiply(name='gain', weight_shape=None)
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    posterior = optimus.Output(name='posterior')

    # 2. Define Edges
    base_edges = [
        (input_data, transpose.input),
        (transpose.output, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer0.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input),
        (softmax.output, prior.input),
        (prior.output, posterior)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + dropout_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
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
        outputs=[posterior],
        verbose=True)

    return trainer, predictor


def i8x1a3T_nll2(size, use_dropout=False):
    k0, k1, k2 = dict(
        large=(2048, 2048, 40),)[size]

    input_data = optimus.Input(
        name='data',
        shape=(None, 8, 1, 252))

    chord_idx = optimus.Input(
        name='class_idx',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    inputs = [input_data, chord_idx, learning_rate]

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Affine(
        name='layer0',
        input_shape=input_data.shape,
        output_shape=(None, k0),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        output_shape=(None, k1),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        output_shape=(None, k2, 1, 12),
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

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer0.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')
    prior = optimus.Multiply("prior", weight_shape=(1, 157), broadcast=[0])
    prior.weight.value = np.ones([1, 157])

    param_nodes = [layer0, layer1, layer2, null_classifier, chord_classifier]
    misc_nodes = [flatten, cat, softmax, prior]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Multiply(name='gain', weight_shape=None)
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    posterior = optimus.Output(name='posterior')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer0.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input),
        (softmax.output, prior.input),
        (prior.output, posterior)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + dropout_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
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
        outputs=[posterior],
        verbose=True)

    return trainer, predictor


def i8c3_pwmse(size='large'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 8, 252))

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
        weight_shape=(k0, None, 3, 13),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 3, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 3, 33),
        act_type='relu')

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 2, 1),
        act_type='sigmoid')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='sigmoid')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')
    dimshuffle = optimus.Dimshuffle('dimshuffle', (0, 'x'))
    squared_error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mean_squared_error')

    loss_nodes = [likelihoods, dimshuffle, squared_error, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer2.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (cat.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, dimshuffle.input),
            (dimshuffle.output, squared_error.input_a),
            (target, squared_error.input_b),
            (squared_error.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    posterior = optimus.Output(name='posterior')
    predictor_edges = optimus.ConnectionManager(
        base_edges + [(cat.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
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
    summer = optimus.Add(name='moia_sum')
    flatten = optimus.Sum('flatten', axis=1)
    dimshuffle2 = optimus.Dimshuffle('dimshuffle2', (0, 'x'))
    margin = optimus.RectifiedLinear(name='margin')
    weight = optimus.Multiply(name="margin_weight")
    margin_loss = optimus.Mean(name='margin_loss', axis=None)

    loss_nodes2 = [negone, summer, margin, flatten,
                   dimshuffle2, margin_loss, weight]
    total_loss = optimus.Add("total_loss")

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


def i20c3_mse12(size='large'):
    k0, k1, k2 = dict(
        small=(10, 20, 40),
        med=(12, 24, 48),
        large=(16, 32, 64))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 20, 252))

    target = optimus.Input(
        name='target',
        shape=(None, 12))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 5, 37),
        pool_shape=(2, 1),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 1, 33),
        pool_shape=(2, 1),
        act_type='relu')

    chroma_estimator = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(1, None, 1, 1),
        act_type='sigmoid')

    flatten = optimus.Flatten('flatten', 2)

    param_nodes = [layer0, layer1, layer2, chroma_estimator]
    misc_nodes = [flatten]

    # 1.1 Create Loss
    error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mean_squared_error')
    loss_nodes = [error, loss]

    chroma = optimus.Output(name='chroma')
    total_loss = optimus.Output(name='total_loss')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chroma_estimator.input),
        (chroma_estimator.output, flatten.input),
        (flatten.output, chroma)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (flatten.output, error.input_a),
            (target, error.input_b),
            (error.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    classifier_init(param_nodes)

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, target, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, chroma],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[chroma])

    return trainer, predictor


MODELS = {
    'i1c3_nll_L': lambda: iXc3_nll(1, 'large'),
    'i1c3_nll_M': lambda: iXc3_nll(1, 'med'),
    'i1c3_nll_S': lambda: iXc3_nll(1, 'small'),
    'i20c3_nll_L': lambda: iXc3_nll(20, 'large'),
    'i20c3_nll_XL': lambda: iXc3_nll(20, 'xlarge'),
    'i20c3_nll_XXL': lambda: iXc3_nll(20, 'xxlarge'),
    'i1c3_nll_dropout_L': lambda: iXc3_nll(1, 'large', True),
    'i1c3_nll_dropout_M': lambda: iXc3_nll(1, 'med', True),
    'i1c3_nll_dropout_S': lambda: iXc3_nll(1, 'small', True),
    'i4c3_nll_dropout_L': lambda: iXc3_nll(4, 'large', True),
    'i4c3_nll_dropout_M': lambda: iXc3_nll(4, 'med', True),
    'i4c3_nll_dropout_S': lambda: iXc3_nll(4, 'small', True),
    'i10c3_nll_dropout_L': lambda: iXc3_nll(10, 'large', True),
    'i10c3_nll_dropout_M': lambda: iXc3_nll(10, 'med', True),
    'i10c3_nll_dropout_S': lambda: iXc3_nll(10, 'small', True),
    'i20c3_nll_dropout_XXL': lambda: iXc3_nll(20, 'xxlarge', True),
    'i20c3_nll_dropout_XL': lambda: iXc3_nll(20, 'xlarge', True),
    'i20c3_nll_dropout_L': lambda: iXc3_nll(20, 'large', True),
    'i20c3_nll_dropout_M': lambda: iXc3_nll(20, 'med', True),
    'i20c3_nll_dropout_S': lambda: iXc3_nll(20, 'small', True),
    'i8x1c3T_nll2_L': lambda: i8x1c3T_nll2('large', False),
    'i8x1c3T_nll2_dropout_L': lambda: i8x1c3T_nll2('large', True)}
