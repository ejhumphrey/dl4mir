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
        weight_shape=(16, None, 5, 5),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(20, None, 5, 7),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(24, None, 3, 6),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 512,),
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

    classifier_init(param_nodes)

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


def wcqt_nll2():
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 5, TIME_DIM, 80))

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
        weight_shape=(32, None, 5, 9),
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
        weight_shape=(128, None, 4, 7),
        act_type='relu')

    layer3 = optimus.Conv3D(
        name='layer3',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    reorder = optimus.Flatten('reorder', 2)

    # no_chord = optimus.Affine(
    #     name='no_chord',
    #     input_shape=(None, 128*12*2),
    #     output_shape=(None, 1),
    #     act_type='linear')

    # cat = optimus.Concatenate('concatenate', axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, layer3]
    misc_nodes = [reorder, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, reorder.input),
        (reorder.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    posterior = optimus.Output(
        name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def allconv_nll(size='small'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

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
        weight_shape=(k0, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 5, 37),
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
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

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
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])  # , out0, out1, out2])

    return trainer, predictor


def bs_conv4_pcabasis_nll(size='small'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

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
        weight_shape=(8, None, 2, 1),
        act_type='linear')

    l2norm = optimus.NormalizeDim(
        name='l2norm', axis=1, mode='l2')

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
    misc_nodes = [l2norm, flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]
    total_loss = optimus.Output(name='total_loss')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, l2norm.input),
        (l2norm.output, chord_classifier.input),
        (layer3.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
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
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes[:-1])

    semitones = L.semitone_matrix(157)[:-1]
    bases = PCA(n_components=8).fit_transform(semitones)
    bases /= np.sqrt(np.power(bases, 2.0).sum(axis=1)).reshape(-1, 1)
    chord_classifier.weights.value = bases.reshape(13, 8, 1, 1)

    # l2out = optimus.Output(name='l2out')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (l2norm.output, l2out)])
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


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


def i1c3_nll(size='large'):
    k0, k1, k2 = dict(
        small=(10, 20, 40),
        med=(12, 24, 48),
        large=(16, 32, 64))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 1, 252))

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
        weight_shape=(k0, None, 1, 13),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 1, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 1, 33),
        act_type='relu')

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
        base_edges + [
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
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[posterior])

    return trainer, predictor


def i1c3_nll_dropout(size='large'):
    k0, k1, k2 = dict(
        small=(10, 20, 40),
        med=(12, 24, 48),
        large=(16, 32, 64))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 1, 252))

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
        weight_shape=(k0, None, 1, 13),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 1, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 1, 33),
        act_type='relu')

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

    layer0.enable_dropout()
    layer1.enable_dropout()
    layer2.enable_dropout()

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
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, dropout],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    layer0.disable_dropout()
    layer1.disable_dropout()
    layer2.disable_dropout()

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[posterior])

    return trainer, predictor


def i1c6_nll_dropout(size='large'):
    k0, k1, k2, k3, k4, k5 = dict(
        large=(16, 32, 64, 256, 256, 256))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, 1, 252))

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
        weight_shape=(k0, None, 1, 13),
        pool_shape=(1, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 1, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 1, 33),
        act_type='relu')

    layer3 = optimus.Conv3D(
        name='layer3',
        input_shape=layer2.output.shape,
        weight_shape=(k3, None, 1, 1),
        act_type='relu')

    layer4 = optimus.Conv3D(
        name='layer4',
        input_shape=layer3.output.shape,
        weight_shape=(k4, None, 1, 1),
        act_type='relu')

    layer5 = optimus.Conv3D(
        name='layer5',
        input_shape=layer4.output.shape,
        weight_shape=(k5, None, 1, 1),
        act_type='relu')

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer5.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer5.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')
    prior = optimus.Multiply("prior", weight_shape=(1, 157), broadcast=[0])
    prior.weight.value = np.ones([1, 157])

    layer3.enable_dropout()
    layer4.enable_dropout()
    layer5.enable_dropout()

    param_nodes = [layer0, layer1, layer2, layer3, layer4, layer5,
                   null_classifier, chord_classifier]
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
        (layer2.output, layer3.input),
        (layer3.output, layer4.input),
        (layer4.output, layer5.input),
        (layer5.output, chord_classifier.input),
        (layer5.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input),
        (softmax.output, prior.input),
        (prior.output, posterior)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (dropout, layer3.dropout),
            (dropout, layer4.dropout),
            (dropout, layer5.dropout),
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, dropout],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, posterior],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    layer3.disable_dropout()
    layer4.disable_dropout()
    layer5.disable_dropout()

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[posterior])

    return trainer, predictor


def bs_conv3_nll(size='large'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

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
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

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
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])  # , out0, out1, out2])

    return trainer, predictor


def bs_conv3_bottleneck_nll(size='large'):
    k0, k1, k2 = dict(
        large=(16, 32, 48))[size]

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

    bottleneck = optimus.Conv3D(
        name='bottleneck',
        input_shape=layer1.output.shape,
        weight_shape=(8, None, 2, 1),
        act_type='linear')

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

    param_nodes = [layer0, layer1, layer2, bottleneck,
                   chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, bottleneck.input),
        (bottleneck.output, chord_classifier.input),
        (bottleneck.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    posterior = optimus.Output(name='posterior')
    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def bs_conv3_nll_dropout(size='large'):
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

    layer0.enable_dropout()
    layer1.enable_dropout()
    layer2.enable_dropout()

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 2, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

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
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, dropout],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

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


def i8c3_pwmse_dropout(size='large'):
    k0, k1, k2 = dict(
        large=(24, 48, 64))[size]

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

    layer0.enable_dropout()
    layer1.enable_dropout()
    layer2.enable_dropout()

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
            (dropout, layer0.dropout),
            (dropout, layer1.dropout),
            (dropout, layer2.dropout),
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
        inputs=[input_data, target, chord_idx, learning_rate, dropout],
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


def bs_conv3_l2normed_nll(size='large'):
    k0, k1, k2 = dict(
        large=(16, 32, 48))[size]

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

    l2norm = optimus.NormalizeDim(
        name='l2norm', axis=3, mode='l2')

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
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [l2norm, flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, l2norm.input),
        (l2norm.output, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input),
        (layer2.output, null_classifier.input),
        (chord_classifier.output, flatten.input),
        (flatten.output, cat.input_0),
        (null_classifier.output, cat.input_1),
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    classifier_init(param_nodes)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (l2norm.output, out0)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def bs_conv3_mce(size='large'):
    k0, k1, k2 = dict(
        large=(16, 32, 48))[size]

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

    decay = optimus.Input(
        name='decay',
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

    # 1.1 Create Losses
    weight_loss0 = optimus.WeightDecay("weight_loss0")
    weight_loss1 = optimus.WeightDecay("weight_loss1")
    weight_loss2 = optimus.WeightDecay("weight_loss2")

    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='gain')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name='moia_values')

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Add(name='summer', num_inputs=2)

    soft_step = optimus.Sigmoid(name='soft_step')
    mce_loss = optimus.Mean(name='mce_loss')
    total_loss = optimus.Add('total_loss', num_inputs=4)
    loss_nodes = [weight_loss0, weight_loss1, weight_loss2,
                  log, neg_one0, target_values, moia_values,
                  neg_one1, summer, soft_step, mce_loss, total_loss]

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
            (layer0.weights, weight_loss0.input),
            (decay, weight_loss0.weight),
            (layer1.weights, weight_loss1.input),
            (decay, weight_loss1.weight),
            (layer2.weights, weight_loss2.input),
            (decay, weight_loss2.weight),
            (cat.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (target_values.output, summer.input_0),
            (moia_values.output, neg_one1.input),
            (neg_one1.output, summer.input_1),
            (summer.output, soft_step.input),
            (soft_step.output, mce_loss.input),
            (mce_loss.output, total_loss.input_0),
            (weight_loss0.output, total_loss.input_1),
            (weight_loss1.output, total_loss.input_2),
            (weight_loss2.output, total_loss.input_3)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, decay],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss.output],
        loss=total_loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            if 'classifier' in n.name and 'bias' in p.name:
                continue
            optimus.random_init(p)

    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(cat.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])  # , out0, out1, out2])

    return trainer, predictor


def bs_conv3_cnll(size='large'):
    k0, k1, k2 = dict(
        large=(16, 32, 48))[size]

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

    margin = optimus.Input(
        name='margin',
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
    log = optimus.Log(name='log')
    neg_one = optimus.Gain(name='gain')
    neg_one.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    target_matrix = optimus.Dimshuffle('target_matrix', (0, 'x'))
    margin_sum = optimus.Add(name='margin_sum', num_inputs=3)
    relu = optimus.RectifiedLinear(name='relu')
    margin_loss = optimus.Mean('margin_loss', axis=None)
    target_loss = optimus.Mean('target_loss', axis=None)

    total_loss = optimus.Add(name='total_loss', num_inputs=2)
    loss_nodes = [log, neg_one, target_values, target_matrix,
                  margin_sum, relu, margin_loss, target_loss, total_loss]

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
            (cat.output, log.input),
            (log.output, neg_one.input),
            (neg_one.output, target_values.input),
            (chord_idx, target_values.index),
            (target_values.output, target_loss.input),
            (target_loss.output, total_loss.input_0),
            (margin, margin_sum.input_0),
            (target_values.output, target_matrix.input),
            (target_matrix.output, margin_sum.input_1),
            (log.output, margin_sum.input_2),
            (margin_sum.output, relu.input),
            (relu.output, margin_loss.input),
            (margin_loss.output, total_loss.input_1)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, margin],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss.output],
        loss=total_loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            if 'classifier' in n.name and 'bias' in p.name:
                continue
            optimus.random_init(p)

    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(cat.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])  # , out0, out1, out2])

    return trainer, predictor


def bs_conv3_margin(size='large'):
    k0, k1, k2 = dict(
        large=(16, 32, 48))[size]

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

    margin = optimus.Input(
        name='margin',
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
    log = optimus.Log(name='log')
    neg_one = optimus.Gain(name='gain')
    neg_one.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MaxNotIndex(name='moia_values')

    margin_sum = optimus.Add(name='margin_sum', num_inputs=3)
    relu = optimus.RectifiedLinear(name='relu')
    margin_loss = optimus.Mean('margin_loss', axis=None)
    # target_loss = optimus.Mean('target_loss', axis=None)

    # total_loss = optimus.Add(name='total_loss', num_inputs=2)
    loss_nodes = [log, neg_one, target_values, moia_values,
                  margin_sum, relu, margin_loss]  # , target_loss, total_loss]

    posterior = optimus.Output(name='posterior')
    target_out = optimus.Output(name='target_out')
    moia_out = optimus.Output(name='moia_out')
    loss = optimus.Output(name='loss')

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
            (cat.output, log.input),
            (cat.output, posterior),
            (log.output, neg_one.input),
            (neg_one.output, target_values.input),
            (chord_idx, target_values.index),
            (margin, margin_sum.input_0),
            (target_values.output, margin_sum.input_1),
            (log.output, moia_values.input),
            (chord_idx, moia_values.index),
            (moia_values.output, margin_sum.input_2),
            (moia_values.output, moia_out),
            (margin_sum.output, relu.input),
            (relu.output, margin_loss.input),
            (target_values.output, target_out),
            (margin_loss.output, loss)])
            # (target_values.output, target_loss.input),

            # (target_loss.output, total_loss.input_0),
            # (margin_loss.output, total_loss.input_1),
            # (total_loss.output, loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, margin],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss, posterior, target_out, moia_out],
        loss=loss,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p, 0.0, 0.1)

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(cat.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def allconv_margin(size='small'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

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

    margin = optimus.Input(
        name='margin',
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
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='neg_one0')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name="moia_values")

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Add(name='summer', num_inputs=3)

    max_w0 = optimus.RectifiedLinear(name='max_w0')
    loss = optimus.Mean(name='margin_loss')

    loss_nodes = [log, neg_one0, target_values, moia_values,
                  neg_one1, summer, max_w0, loss]

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
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (target_values.output, summer.input_0),
            (moia_values.output, neg_one1.input),
            (neg_one1.output, summer.input_1),
            (margin, summer.input_2),
            (summer.output, max_w0.input),
            (max_w0.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, margin],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    out0 = optimus.Output(name='out0')
    out1 = optimus.Output(name='out1')
    out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior),
                      (layer0.output, out0),
                      (layer1.output, out1),
                      (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior, out0, out1, out2])

    return trainer, predictor


def wcqt_allconv_nll(size='small'):
    k0, k1, k2 = dict(
        small=(8, 16, 20),
        med=(12, 24, 32),
        large=(16, 32, 48))[size]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 5, TIME_DIM, 80))

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
        weight_shape=(k0, None, 5, 9),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, 5, 7),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, 3, 7),
        act_type='relu')

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 2, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

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
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])  # , out0, out1, out2])

    return trainer, predictor


def cqt_3layer_allconv_smax_dropout():
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

    dropout = optimus.Input(
        name='dropout',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(12, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(40, None, 5, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(40, None, 4, 33),
        act_type='relu')

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

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')
    loss_nodes = [likelihoods, log, neg, loss]

    layer0.enable_dropout()
    layer1.enable_dropout()
    layer2.enable_dropout()

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
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate, dropout],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    # for n in param_nodes:
    #     for p in n.params.values():
    #         optimus.random_init(p)

    layer0.disable_dropout()
    layer1.disable_dropout()
    layer2.disable_dropout()

    out0 = optimus.Output(name='out0')
    out1 = optimus.Output(name='out1')
    out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior),
                      (layer0.output, out0),
                      (layer1.output, out1),
                      (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior, out0, out1, out2])

    return trainer, predictor


def cqt_3layer_convclassifier_smax_smce():
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
        weight_shape=(8, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(20, None, 5, 37),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 3, 33),
        act_type='relu')

    chord_classifier = optimus.Conv3D(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 2, 1),
        act_type='linear')

    flatten = optimus.Flatten('flatten', 2)

    null_classifier = optimus.Affine(
        name='null_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, 1),
        act_type='linear')

    cat = optimus.Concatenate('concatenate', num_inputs=2, axis=1)
    softmax = optimus.Softmax('softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier, null_classifier]
    misc_nodes = [flatten, cat, softmax]

    # 1.1 Create Loss
    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='neg_one0')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name="moia_values")

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Add(name='summer', num_inputs=2)

    soft_step = optimus.SoftRectifiedLinear(name='soft_step', knee=5.0)
    loss = optimus.Mean(name='mce_loss')

    loss_nodes = [log, neg_one0, target_values, moia_values,
                  neg_one1, summer, soft_step, loss]

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
        (cat.output, softmax.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (target_values.output, summer.input_0),
            (moia_values.output, neg_one1.input),
            (neg_one1.output, summer.input_1),
            (summer.output, soft_step.input),
            (soft_step.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    # for n in param_nodes:
    #     for p in n.params.values():
    #         optimus.random_init(p)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(softmax.output, posterior)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + misc_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def cqt_nll_margin():
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

    margin = optimus.Input(
        name='margin',
        shape=None)

    # 1.2 Create Nodes
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(6, None, 5, 7),  # or 13
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(20, None, 5, 15),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 3, 15),
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

    param_nodes = [layer0, layer1, layer2, layer3, chord_estimator]

    # 1.1 Create Loss
    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='neg_one0')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name="moia_values")

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Add(name='summer')

    relu = optimus.RectifiedLinear(name='relu')
    loss = optimus.Mean(name='margin_loss')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, chord_estimator.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_estimator.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (margin, summer.input_list),
            (target_values.output, summer.input_list),
            (moia_values.output, neg_one1.input),
            (neg_one1.output, summer.input_list),
            (summer.output, relu.input),
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
        outputs=[loss.output],
        loss=loss.output,
        updates=updates.connections,
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
        name='chord_estimator',
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
    summer = optimus.Add(name='summer')

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
        outputs=[loss.output],
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


def cqt_smax_3layer(n_dim=VOCAB):
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
        weight_shape=(8, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(24, None, 5, 37),
        act_type='relu')

    layer2 = optimus.Affine(
        name='layer2',
        input_shape=layer1.output.shape,
        output_shape=(None, 512,),
        act_type='relu')

    chord_classifier = optimus.Affine(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, n_dim),
        act_type='softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex(name='likelihoods')

    log = optimus.Log(name='log')
    neg = optimus.Gain(name='gain')
    neg.weight.value = -1.0

    loss = optimus.Mean(name='negative_log_likelihood')

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_classifier.output, likelihoods.input),
            (chord_idx, likelihoods.index),
            (likelihoods.output, log.input),
            (log.output, neg.input),
            (neg.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + [likelihoods, neg, log, loss],
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    # out0 = optimus.Output(name='out0')
    # out1 = optimus.Output(name='out1')
    # out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_classifier.output, posterior)])
                      # (layer0.output, out0),
                      # (layer1.output, out1),
                      # (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def cqt_smax_3layer_mce(n_dim=VOCAB):
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
        weight_shape=(8, None, 5, 13),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(24, None, 5, 37),
        act_type='relu')

    layer2 = optimus.Affine(
        name='layer2',
        input_shape=layer1.output.shape,
        output_shape=(None, 512,),
        act_type='relu')

    chord_classifier = optimus.Affine(
        name='chord_classifier',
        input_shape=layer2.output.shape,
        output_shape=(None, n_dim),
        act_type='softmax')

    param_nodes = [layer0, layer1, layer2, chord_classifier]

    # 1.1 Create Loss
    log = optimus.Log(name='log')
    neg_one0 = optimus.Gain(name='neg_one0')
    neg_one0.weight.value = -1.0

    target_values = optimus.SelectIndex(name='target_values')
    moia_values = optimus.MinNotIndex(name="moia_values")

    neg_one1 = optimus.Gain(name='neg_one1')
    neg_one1.weight.value = -1.0
    summer = optimus.Add(name='summer', num_inputs=2)

    soft_step = optimus.Sigmoid(name='soft_step')
    loss = optimus.Mean(name='mce_loss')

    loss_nodes = [log, neg_one0, target_values, moia_values,
                  neg_one1, summer, soft_step, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, chord_classifier.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (chord_classifier.output, log.input),
            (log.output, neg_one0.input),
            (neg_one0.output, target_values.input),
            (chord_idx, target_values.index),
            (neg_one0.output, moia_values.input),
            (chord_idx, moia_values.index),
            (target_values.output, summer.input_0),
            (moia_values.output, neg_one1.input),
            (neg_one1.output, summer.input_1),
            (summer.output, soft_step.input),
            (soft_step.output, loss.input)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data, chord_idx, learning_rate],
        nodes=param_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    out0 = optimus.Output(name='out0')
    out1 = optimus.Output(name='out1')
    out2 = optimus.Output(name='out2')
    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_classifier.output, posterior),
                      (layer0.output, out0),
                      (layer1.output, out1),
                      (layer2.output, out2)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[out0, out1, out2, posterior])

    return trainer, predictor


def cqt_likelihood(n_dim=VOCAB):
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, TIME_DIM, 252))

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
        weight_shape=(6, None, 5, 7),  # or 13
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(20, None, 5, 15),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 3, 15),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 512,),
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

    posterior = optimus.Output(name='posterior')
    out0 = optimus.Output(name='out0')
    out1 = optimus.Output(name='out1')
    out2 = optimus.Output(name='out2')
    out3 = optimus.Output(name='out3')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_estimator.output, posterior),
                      (layer0.output, out0),
                      (layer1.output, out1),
                      (layer2.output, out2),
                      (layer3.output, out3)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[out0, out1, out2, out3, posterior])

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
        weight_shape=(12, None, 5, 5),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(16, None, 5, 7),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 3, 6),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, 512,),
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

    for n in layer0, layer1, layer2:
        for p in n.params.values():
            optimus.random_init(p, 0.01, 0.01)

    for p in layer3.params.values():
        optimus.random_init(p, 0.01, 0.001)

    for p in chord_estimator.params.values():
        optimus.random_init(p, 0.0, 0.001)

    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(chord_estimator.output, posterior)])

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes,
        connections=predictor_edges.connections,
        outputs=[posterior])

    return trainer, predictor


def wcqt_likelihood2():
    input_data = optimus.Input(
        name='cqt',
        shape=(None, 5, TIME_DIM, 80))

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
        weight_shape=(12, None, 5, 9),
        pool_shape=(2, 3),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(16, None, 5, 7),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(20, None, 4, 7),
        act_type='relu')

    layer3 = optimus.Conv3D(
        name='layer3',
        input_shape=layer2.output.shape,
        weight_shape=(13, None, 1, 1),
        act_type='sigmoid')

    reorder = optimus.Flatten('reorder', 2)

    # no_chord = optimus.Affine(
    #     name='no_chord',
    #     input_shape=(None, 128*12),
    #     output_shape=(None, 1),
    #     act_type='sigmoid')

    # cat = optimus.Concatenate('concatenate', axis=1)

    param_nodes = [layer0, layer1, layer2, layer3]
    misc_nodes = [reorder]

    # 1.1 Create Loss
    likelihoods = optimus.SelectIndex('select')
    dimshuffle = optimus.Dimshuffle('dimshuffle', (0, 'x'))
    error = optimus.SquaredEuclidean(name='squared_error')
    loss = optimus.Mean(name='mean_squared_error')

    loss_nodes = [likelihoods, dimshuffle, error, loss]

    # 2. Define Edges
    base_edges = [
        (input_data, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, reorder.input)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (reorder.output, likelihoods.input),
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
        inputs=[input_data, chord_idx, target, learning_rate],
        nodes=param_nodes + misc_nodes + loss_nodes,
        connections=trainer_edges.connections,
        outputs=[loss.output],
        loss=loss.output,
        updates=update_manager.connections,
        verbose=True)

    for n in param_nodes:
        for p in n.params.values():
            optimus.random_init(p)

    posterior = optimus.Output(name='posterior')

    predictor_edges = optimus.ConnectionManager(
        base_edges + [(reorder.output, posterior)])

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


MODELS = {
    'bs_conv3_bottleneck_nll_large': lambda: bs_conv3_bottleneck_nll('large'),
    'bs_conv3_nll_dropout_large': lambda: bs_conv3_nll_dropout('large'),
    'bs_conv3_l2normed_nll_large': lambda: bs_conv3_l2normed_nll('large'),
    'bs_conv4_pcabasis_nll_small': lambda: bs_conv4_pcabasis_nll('small'),
    'bs_conv4_pcabasis_nll_med': lambda: bs_conv4_pcabasis_nll('med'),
    'bs_conv4_pcabasis_nll_large': lambda: bs_conv4_pcabasis_nll('large'),
    'i8c4b10_nll_dropout_L': lambda: i8c4b10_nll_dropout('large'),
    'i8c3_pwmse_dropout_L': lambda: i8c3_pwmse_dropout('large'),
    'i8c3_pwmse_L': lambda: i8c3_pwmse('large'),
    'i1c3_nll_L': lambda: i1c3_nll('large'),
    'i1c3_nll_M': lambda: i1c3_nll('med'),
    'i1c3_nll_S': lambda: i1c3_nll('small'),
    'i1c3_nll_dropout_L': lambda: i1c3_nll_dropout('large'),
    'i1c3_nll_dropout_M': lambda: i1c3_nll_dropout('med'),
    'i1c3_nll_dropout_S': lambda: i1c3_nll_dropout('small'),
    'i1c6_nll_dropout_L': lambda: i1c6_nll_dropout('large'),
    'bs_conv3_nll_large': lambda: bs_conv3_nll('large'),
    'bs_conv3_nll_small': lambda: bs_conv3_nll('small'),
    'bs_conv3_nll_med': lambda: bs_conv3_nll('med'),
    'bs_conv3_mce_large': lambda: bs_conv3_mce('large'),
    'bs_conv3_cnll_large': lambda: bs_conv3_cnll('large'),
    'bs_conv3_margin_large': lambda: bs_conv3_margin('large'),
    'cqt_allconv_nll_small': lambda: allconv_nll('small'),
    'cqt_allconv_nll_med': lambda: allconv_nll('med'),
    'cqt_allconv_nll_large': lambda: allconv_nll('large'),
    'wcqt_allconv_nll_small': lambda: wcqt_allconv_nll('small'),
    'wcqt_allconv_nll_med': lambda: wcqt_allconv_nll('med'),
    'wcqt_allconv_nll_large': lambda: wcqt_allconv_nll('large')}
