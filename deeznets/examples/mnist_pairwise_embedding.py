
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

affine1 = deeznets.Affine(
    name='affine1',
    input_shape=(512,),
    output_shape=(2,),
    activation='tanh')

conv0_A = conv0.copy('conv0_A')
affine0_A = affine0.copy('affine0_A')
affine1_A = affine1.copy('affine1_A')

distance = deeznets.LpDistance(
    name='distance',
    p=2.0)

# Store all nodes in a list.
# - - - - - - - - - - - - - -
nodes = [conv0, affine0, affine1,
         conv0_A, affine0_A, affine1_A,
         distance]

# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
edges = [("$:x_A", conv0.inputs[0]),
         (conv0.outputs[0], affine0.inputs[0]),
         (affine0.outputs[0], affine1.inputs[0]),
         (affine1.outputs[0], distance.inputs[0]),
         (affine1.outputs[0], "=:affine0_out"),
         ("$:x_B", conv0_A.inputs[0]),
         (conv0_A.outputs[0], affine0_A.inputs[0]),
         (affine0_A.outputs[0], affine1_A.inputs[0]),
         (affine1_A.outputs[0], distance.inputs[1]),
         (distance.outputs[0], "=:distance")]


# Define losses over the network.
# - - - - - - - - - - - - - - - -
cdloss = deeznets.ContrastiveDivergence(
    distance='distance',
    score='similarity',
    margin='margin')

conv0_decay = deeznets.L2Norm(
    variable="conv0.weights",
    weight='weight_decay:conv0.weights')

affine0_sparsity = deeznets.L1Norm(
    variable="affine0_out",
    weight='sparsity:affine0_out')

losses = [cdloss, conv0_decay, affine0_sparsity]

# conv_norm = deeznets.UnitL2Norm('conv0.weights')
# constraints = deeznets.Constraints([conv_norm], graph.params)

driver = deeznets.Driver(nodes=nodes, edges=edges, losses=losses)

fh = sources.File("/Users/ejhumphrey/Desktop/mnist.shf")
cache = sources.Cache(
    source=selectors.Permutation(fh),
    cache_size=5000,
    refresh_prob=0)

batch = sources.PairedBatch(
    source=selectors.UniformLabel(cache),
    batch_size=50,
    label_key='0',
    value_shape=(1, 28, 28))

update_sources = {
    'x_A': batch.values_A,
    'x_B': batch.values_B,
    'similarity': batch.equals,
    'margin': deeznets.Constant(0.25),
    'conv0.dropout': deeznets.Constant(0),
    'conv0_A.dropout': deeznets.Constant(0),
    'affine0.dropout': deeznets.Constant(0),
    'affine0_A.dropout': deeznets.Constant(0),
    'affine1.dropout': deeznets.Constant(0),
    'affine1_A.dropout': deeznets.Constant(0),
    'weight_decay:conv0.weights': deeznets.Constant(0),
    'sparsity:affine0_out': deeznets.Constant(0),
    'learning_rate:conv0.weights': deeznets.Constant(0.01),
    'learning_rate:conv0.bias': deeznets.Constant(0.01),
    'learning_rate:affine0.weights': deeznets.Constant(0.01),
    'learning_rate:affine0.bias': deeznets.Constant(0.01),
    'learning_rate:affine1.weights': deeznets.Constant(0.01),
    'learning_rate:affine1.bias': deeznets.Constant(0.01)}

inputs = deeznets.DataServer(update_sources, updates=[batch.refresh])

driver.train(inputs, 5000, 50)
