from ejhumphrey import deeznets
import theano

# Define nodes.
# - - - - - - -
conv0 = deeznets.Conv3D(
    name='conv0',
    input_shape=(1, 28, 28),
    weight_shape=(15, 9, 9),
    pool_shape=(2, 2),
    activation='relu')

affine0 = deeznets.Affine(
    name='affine0',
    input_shape=conv0.output_shapes.values()[0],
    output_shape=(512,),
    activation='relu')

softmax0 = deeznets.Softmax(
    name='softmax0',
    input_shape=(512,),
    output_shape=(2,),
    activation='linear')

# Store all nodes in a list.
# - - - - - - - - - - - - - -
nodes = [conv0, affine0, softmax0]

# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
edges = [("$:x", conv0.inputs[0]),
         (conv0.outputs[0], affine0.inputs[0]),
         (affine0.outputs[0], softmax0.inputs[0]),
         (affine0.outputs[0], "=:affine0_out"),
         (softmax0.outputs[0], "=:posterior")]

# Create the graph, specifying the name in-line.
# - - - - - - - - - - - - - - - - - - - - - - - -
graph = deeznets.Graph(nodes=nodes, edges=edges)

# Define losses over the network.
# - - - - - - - - - - - - - - - -
nll = deeznets.NegativeLogLikelihood(
    posterior='posterior',
    target_idx='class_idx')

conv0_decay = deeznets.L2Norm(
    variable="conv0.weights",
    weight='conv0-decay')

affine0_sparsity = deeznets.L1Norm(
    variable="affine0_out",
    weight='affine0-sparsity')

losses = [nll, conv0_decay, affine0_sparsity]
accumulator = deeznets.Accumulator(losses, graph.variables)

param_names = graph.params.keys()
sgd = deeznets.SGD(param_names, accumulator.total, graph.params)

# conv_norm = deeznets.UnitL2Norm('conv0.weights')
# constraints = deeznets.Constraints([conv_norm], graph.params)
classify_inputs = graph.inputs.values()
fx_classify = theano.function(inputs=classify_inputs,
                              outputs=graph.outputs['posterior'],
                              allow_input_downcast=True)

loss_inputs = classify_inputs + accumulator.inputs.values()
fx_loss = theano.function(inputs=loss_inputs,
                          outputs=accumulator.total,
                          allow_input_downcast=True)

update_inputs = loss_inputs + sgd.inputs.values()
fx_update = theano.function(inputs=update_inputs,
                            outputs=accumulator.total,
                            updates=sgd.updates,
                            allow_input_downcast=True)
