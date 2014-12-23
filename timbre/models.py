import optimus

TIME_DIM = 20
OUT_DIM = 3
GRAPH_NAME = "nlse"


def param_init(nodes, skip_biases=True):
    for n in nodes:
        for k, p in n.params.items():
            if 'bias' in k and skip_biases:
                continue
            optimus.random_init(p, 0, 0.01)


def iX_c3f2_oY(n_in, n_out, size='large'):
    # Kernel shapes
    k0, k1, k2, k3 = dict(
        small=(10, 20, 40, 96),
        med=(12, 24, 48, 128),
        large=(16, 32, 64, 192),
        xlarge=(20, 40, 80, 256),
        xxlarge=(24, 48, 96, 512))[size]

    # Input dimensions
    n0, n1, n2 = {
        1: (1, 1, 1),
        4: (3, 2, 1),
        8: (5, 3, 2),
        10: (3, 3, 1),
        20: (5, 5, 1)}[n_in]

    # Pool shapes
    p0, p1, p2 = {
        1: (1, 1, 1),
        4: (1, 1, 1),
        8: (1, 1, 1),
        10: (2, 2, 1),
        12: (2, 2, 1),
        20: (2, 2, 2)}[n_in]

    input_data = optimus.Input(
        name='cqt',
        shape=(None, 1, n_in, 192))

    input_data_2 = optimus.Input(
        name='cqt_2',
        shape=(None, 1, n_in, 192))

    score = optimus.Input(
        name='score',
        shape=(None,))

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    sim_margin = optimus.Input(
        name='sim_margin',
        shape=None)

    diff_margin = optimus.Input(
        name='diff_margin',
        shape=None)

    inputs = [input_data, input_data_2, score, learning_rate,
              sim_margin, diff_margin]

    # 1.2 Create Nodes
    logscale = optimus.Log("logscale", 1.0)
    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=input_data.shape,
        weight_shape=(k0, None, n0, 13),
        pool_shape=(p0, 2),
        act_type='tanh')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, n1, 11),
        pool_shape=(p1, 2),
        act_type='tanh')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, n2, 9),
        pool_shape=(p2, 2),
        act_type='tanh')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, k3),
        act_type='tanh')

    layer4 = optimus.Affine(
        name='layer4',
        input_shape=layer3.output.shape,
        output_shape=(None, n_out),
        act_type='linear')

    param_nodes = [layer0, layer1, layer2, layer3, layer4]

    # 1.1 Create cloned nodes
    logscale_2 = logscale.clone("logscale_2")
    layer0_2 = layer0.clone('layer0_2')
    layer1_2 = layer1.clone('layer1_2')
    layer2_2 = layer2.clone('layer2_2')
    layer3_2 = layer3.clone('layer3_2')
    layer4_2 = layer4.clone('layer4_2')

    param_nodes_2 = [layer0_2, layer1_2, layer2_2, layer3_2, layer4_2]

    # 1.2 Create Loss
    # ---------------
    #  sim_cost = y*hwr(D - sim_margin)^2
    #  diff_cost = (1 - y) * hwr(diff_margin - D)^2
    #  total = ave(sim_cost + diff_cost)
    sqdistance = optimus.SquaredEuclidean(name='euclidean')
    distance = optimus.Sqrt(name='sqrt')

    # Sim terms
    sim_margin_sum = optimus.Add(name="sim_margin_sum", num_inputs=2)
    sim_hwr = optimus.RectifiedLinear(name="sim_hwr")
    sim_sqhwr = optimus.Power(name='sim_sqhwr', exponent=2.0)
    sim_cost = optimus.Product(name="sim_cost")

    # Diff terms
    neg_distance = optimus.Multiply(name='neg_distance', weight_shape=None)
    neg_distance.weight.value = -1.0
    diff_margin_sum = optimus.Add(name="diff_margin_sum", num_inputs=2)
    diff_hwr = optimus.RectifiedLinear(name="diff_hwr")
    diff_sqhwr = optimus.Power(name='diff_sqhwr', exponent=2.0)

    pos_one = optimus.Constant(name='pos_one', shape=None)
    pos_one.data.value = 1.0
    neg_score = optimus.Multiply(name='neg_score', weight_shape=None)
    neg_score.weight.value = -1.0

    diff_selector = optimus.Add("diff_selector", num_inputs=2)
    diff_cost = optimus.Product(name='diff_cost')

    total_cost = optimus.Add('total_cost', num_inputs=2)
    loss = optimus.Mean(name='loss')

    loss_nodes = [sqdistance, distance,
                  sim_margin_sum, sim_hwr, sim_sqhwr, sim_cost,
                  neg_distance, diff_margin_sum, diff_hwr, diff_sqhwr,
                  pos_one, neg_score, diff_selector, diff_cost,
                  total_cost, loss]

    # Graph outputs
    total_loss = optimus.Output(name='total_loss')
    embedding = optimus.Output(name='embedding')
    embedding_2 = optimus.Output(name='embedding_2')

    # 2. Define Edges
    base_edges = [
        (input_data, logscale.input),
        (logscale.output, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, layer4.input),
        (layer4.output, embedding)]

    base_edges_2 = [
        (input_data_2, logscale_2.input),
        (logscale_2.output, layer0_2.input),
        (layer0_2.output, layer1_2.input),
        (layer1_2.output, layer2_2.input),
        (layer2_2.output, layer3_2.input),
        (layer3_2.output, layer4_2.input),
        (layer4_2.output, embedding_2)]

    cost_edges = [
        (layer4.output, sqdistance.input_a),
        (layer4_2.output, sqdistance.input_b),
        (sqdistance.output, distance.input),
        # Sim terms
        (score, sim_cost.input_a),
        (distance.output, sim_margin_sum.input_0),
        (sim_margin, sim_margin_sum.input_1),
        (sim_margin_sum.output, sim_hwr.input),
        (sim_hwr.output, sim_sqhwr.input),
        (sim_sqhwr.output, sim_cost.input_b),
        (sim_cost.output, total_cost.input_0),
        # Diff terms
        # - margin term
        (distance.output, neg_distance.input),
        (diff_margin, diff_margin_sum.input_0),
        (neg_distance.output, diff_margin_sum.input_1),
        (diff_margin_sum.output, diff_hwr.input),
        (diff_hwr.output, diff_sqhwr.input),
        # - score selector
        (score, neg_score.input),
        (pos_one.output, diff_selector.input_0),
        (neg_score.output, diff_selector.input_1),
        # - product
        (diff_selector.output, diff_cost.input_a),
        (diff_sqhwr.output, diff_cost.input_b),
        (diff_cost.output, total_cost.input_1)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + base_edges_2 + cost_edges + [
            # Combined
            (total_cost.output, loss.input),
            (loss.output, total_loss)])

    update_manager = optimus.ConnectionManager(
        map(lambda n: (learning_rate, n.weights), param_nodes) +
        map(lambda n: (learning_rate, n.bias), param_nodes))

    param_init(param_nodes)

    misc_nodes = [logscale, logscale_2]

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + param_nodes_2 + loss_nodes + misc_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, embedding, embedding_2],
        loss=total_loss,
        updates=update_manager.connections,
        verbose=True)

    pw_cost = optimus.Output(name='pw_cost')
    zerofilt_edges = optimus.ConnectionManager(
        base_edges + base_edges_2 + cost_edges + [
            # Combined
            (total_cost.output, pw_cost)])

    zerofilter = optimus.Graph(
        name=GRAPH_NAME + "_zerofilter",
        inputs=[input_data, input_data_2, score, sim_margin, diff_margin],
        nodes=param_nodes + param_nodes_2 + loss_nodes[:-1] + misc_nodes,
        connections=zerofilt_edges.connections,
        outputs=[pw_cost, embedding, embedding_2],
        verbose=True)

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[input_data],
        nodes=param_nodes + [logscale],
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[embedding],
        verbose=True)

    return trainer, predictor, zerofilter


def test_pairwise(n_in):
    input_data = optimus.Input(
        name='cqt',
        shape=(None, n_in))

    input_data_2 = optimus.Input(
        name='cqt_2',
        shape=(None, n_in))

    score = optimus.Input(
        name='score',
        shape=(None,))

    margin = optimus.Input(
        name='margin',
        shape=None)

    inputs = [input_data, input_data_2, score, margin]

    # 1.2 Create Loss
    # ---------------
    #  same = y*D
    #  diff = (1 - y) * hwr(margin - D)
    #  total = ave(same_cost + diff_cost)

    distance = optimus.SquaredEuclidean(name='euclidean')
    cost_sim = optimus.Product(name="cost_sim")

    neg_distance = optimus.Multiply(
        name='neg_distance',
        weight_shape=None)
    neg_distance.weight.value = -1.0
    margin_sum = optimus.Add(name="margin_sum", num_inputs=2)
    hwr = optimus.RectifiedLinear(name="hwr")

    pos_one = optimus.Constant(
        name='pos_one',
        shape=None)
    pos_one.data.value = 1.0

    neg_score = optimus.Multiply(
        name='neg_score',
        weight_shape=None)
    neg_score.weight.value = -1.0

    diff_selector = optimus.Add("diff_selector", num_inputs=2)
    cost_diff = optimus.Product(name='cost_diff')

    total_cost = optimus.Add('total_cost', num_inputs=2)
    loss = optimus.Mean(name='loss')

    loss_nodes = [distance, cost_sim, neg_distance, margin_sum, hwr,
                  pos_one, neg_score, diff_selector, cost_diff,
                  total_cost, loss]

    # Graph outputs
    total_loss = optimus.Output(name='total_loss')
    dist_sim = optimus.Output(name='dist_sim')
    dist_diff = optimus.Output(name='dist_diff')
    dist = optimus.Output(name='dist')

    # 2. Define Edges
    trainer_edges = optimus.ConnectionManager([
        (input_data, distance.input_a),
        (input_data_2, distance.input_b),
        (distance.output, dist),
        # Sim terms
        (score, cost_sim.input_a),
        (distance.output, cost_sim.input_b),
        (cost_sim.output, dist_sim),
        (cost_sim.output, total_cost.input_0),
        # Diff terms
        (distance.output, neg_distance.input),
        (margin, margin_sum.input_0),
        (neg_distance.output, margin_sum.input_1),
        (margin_sum.output, hwr.input),
        (score, neg_score.input),
        (pos_one.output, diff_selector.input_0),
        (neg_score.output, diff_selector.input_1),
        (diff_selector.output, cost_diff.input_a),
        (hwr.output, cost_diff.input_b),
        (cost_diff.output, dist_diff),
        (cost_diff.output, total_cost.input_1),
        # Combined
        (total_cost.output, loss.input),
        (loss.output, total_loss)])

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=loss_nodes,
        connections=trainer_edges.connections,
        outputs=[total_loss, dist, dist_sim, dist_diff],
        verbose=True)

    return trainer
