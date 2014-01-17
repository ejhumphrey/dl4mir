
import numpy as np
from ejhumphrey.dnn.core import nodes as N
from ejhumphrey.dnn.core import graphs as G
from ejhumphrey.dnn.core import losses as L
from ejhumphrey.dnn.core import updates as U

from ejhumphrey.dnn import driver as D

# Use convenience functions to create nodes.
# - - - - - - - - - - - - - - - - - - - - - -
node0 = N.Conv3DArgs(
    name='conv0',
    input_shape=(1, 28, 28),
    weight_shape=(15, 9, 9),
    pool_shape=(2, 2),
    activation='relu').Node()

node1 = N.AffineArgs(
    name='affine0',
    weight_shape=(np.prod(node0.output_shapes.values()), 512),
    activation='relu').Node()

node2 = N.SoftmaxArgs(
    name='classifier',
    input_dim=512,
    output_dim=10).Node()

# Store all nodes in a dictionary.
# - - - - - - - - - - - - - - - - -
nodes = {node0.name: node0,
         node1.name: node1,
         node2.name: node2}

# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
edges = [(node0.outputs[0], node1.inputs[0]),
         (node1.outputs[0], node2.inputs[0])]

# Create a graph, specifying the name and input in-line.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
graph = G.Network(
    name='mnist_classifier',
    input_names=node0.inputs,
    nodes=nodes,
    edges=edges)

# Define losses over the network.
# - - - - - - - - - - - - - - - -
nll = L.NegativeLogLikelihood(
    posterior='mnist_classifier/classifier.output',
    target_idx='class_labels')

# Configure the parameters to update.
# - - - - - - - - - - - - - - - - - -
updates = U.SGD(param_urls=graph.params.keys())

driver = D.Driver(
    graphs={graph.name: graph},
    losses=[nll],
    updates=updates,
    constraints=None)
