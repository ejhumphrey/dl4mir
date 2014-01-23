from ejhumphrey import deeznets
from ejhumphrey.shufflr import sources
from ejhumphrey.shufflr import selectors

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
    output_shape=(10,),
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

# Define losses over the network.
# - - - - - - - - - - - - - - - -
nll = deeznets.NegativeLogLikelihood(
    posterior='posterior',
    target_idx='class_idx')

conv0_decay = deeznets.L2Norm(
    variable="conv0.weights",
    weight='weight_decay:conv0.weights')

affine0_sparsity = deeznets.L1Norm(
    variable="affine0_out",
    weight='sparsity:affine0_out')

losses = [nll, conv0_decay, affine0_sparsity]

# conv_norm = deeznets.UnitL2Norm('conv0.weights')
# constraints = deeznets.Constraints([conv_norm], graph.params)

driver = deeznets.Driver(nodes=nodes, edges=edges, losses=losses)

fh = sources.File("/Users/ejhumphrey/Desktop/mnist.shf")
cache = sources.Cache(
    source=selectors.Permutation(fh),
    cache_size=1000,
    refresh_prob=0)

batch = sources.LabelBatch(
    source=selectors.UniformLabel(cache),
    batch_size=50,
    label_key='0',
    value_shape=(1, 28, 28))

update_sources = {
    'x': batch.values,
    'class_idx': batch.labels,
    'affine0.dropout': deeznets.Constant(0),
    'conv0.dropout': deeznets.Constant(0),
    'weight_decay:conv0.weights': deeznets.Constant(0),
    'sparsity:affine0_out': deeznets.Constant(0),
    'learning_rate:conv0.bias': deeznets.Constant(0.01),
    'learning_rate:affine0.bias': deeznets.Constant(0.01),
    'learning_rate:affine0.weights': deeznets.Constant(0.01),
    'learning_rate:softmax0.bias': deeznets.Constant(0.01),
    'learning_rate:conv0.weights': deeznets.Constant(0.01),
    'learning_rate:softmax0.weights': deeznets.Constant(0.01)}

inputs = deeznets.DataServer(update_sources, updates=[batch.refresh])

driver.train(inputs, 5000, 50)
