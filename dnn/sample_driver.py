from ejhumphrey import dnn


# Define nodes.
# - - - - - - -
conv0 = dnn.Conv3D(
    name='conv0',
    input_shape=(1, 28, 28),
    weight_shape=(15, 9, 9),
    pool_shape=(2, 2),
    activation='relu')

affine0 = dnn.Affine(
    name='affine0',
    input_shape=conv0.output_shapes.values()[0],
    output_shape=(512,),
    activation='relu')

softmax0 = dnn.Softmax(
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
         (softmax0.outputs[0], "=:posterior")]

# Create the graph, specifying the name in-line.
# - - - - - - - - - - - - - - - - - - - - - - - -
graph = dnn.Graph(
    name='classifier',
    nodes=nodes,
    edges=edges)
