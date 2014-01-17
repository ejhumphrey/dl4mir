
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

node2 = N.AffineArgs(
    name='affine1',
    weight_shape=(512, 2),
    activation='tanh').Node()

# Store all nodes in a dictionary.
# - - - - - - - - - - - - - - - - -
nodes = {node0.name: node0,
         node1.name: node1,
         node2.name: node2}

# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
edges = [(node0.outputs[0], node1.inputs[0]),
         (node1.outputs[0], node2.inputs[0])]

# Create the graphs, specifying the name and input in-line.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
net_1 = G.Network(
    name='net_1',
    inputs=dict(x_in=node0.inputs[0]),
    nodes=nodes,
    edges=edges)

net_2 = G.Network(
    name='net_2',
    inputs=dict(x_in=node0.inputs[0]),
    nodes=nodes,
    edges=edges)

# Define losses over the network.
# - - - - - - - - - - - - - - - -
cdloss = L.ContrastiveDistance(
    input_A='net_1/affine1.output',
    input_B='net_2/affine1.output',
    target_idx='similarity')

# Configure the parameters to update.
# - - - - - - - - - - - - - - - - - -
updates = U.SGD(
    param_urls=net_1.params.keys() + net_2.params.keys())

driver = D.Driver(
    graphs={net_1.name: net_1, net_2.name: net_2},
    losses=[cdloss],
    updates=updates,
    constraints=None)
