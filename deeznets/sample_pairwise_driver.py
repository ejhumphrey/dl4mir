
import numpy as np

from ejhumphrey.dnn import driver as D

from ejhumphrey.dnn.core import nodes as N
from ejhumphrey.dnn.core import graphs as G
from ejhumphrey.dnn.core import losses as L
from ejhumphrey.dnn.core import updates as U
from ejhumphrey.dnn.core import types as T

# Define nodes.
# - - - - - - -
node0 = N.Conv3D(
    name='conv0',
    input_shape=(1, 28, 28),
    weight_shape=(15, 9, 9),
    pool_shape=(2, 2),
    activation='relu')

node1 = N.Affine(
    name='affine0',
    weight_shape=(np.prod(node0.output_shapes.values()), 512),
    activation='relu')

node2 = N.Affine(
    name='affine1',
    weight_shape=(512, 2),
    activation='tanh')

# Store all nodes in a dictionary.
# - - - - - - - - - - - - - - - - -
nodes = {node0.name: node0,
         node1.name: node1,
         node2.name: node2}

# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
path1 = [("$:x1_in", node0.inputs[0]),
         (node0.outputs[0], node1.inputs[0]),
         (node1.outputs[0], node2.inputs[0]),
         (node2.outputs[0], "$:x1_out")]

path2 = [("$:x2_in", node0.inputs[0]),
         (node0.outputs[0], node1.inputs[0]),
         (node1.outputs[0], node2.inputs[0]),
         (node2.outputs[0], "$:x2_out")]

# Define losses over the network.
# - - - - - - - - - - - - - - - -
cdloss = L.ContrastiveDistance(
    input_A='x1_out',
    input_B='x2_out',
    target_idx='similarity')

# Configure the parameters to update.
# - - - - - - - - - - - - - - - - - -
updates = U.SGD(param_urls=net.params.keys())

driver = D.Driver(
    graphs={net.name: net},
    losses=[cdloss],
    updates=updates,
    constraints=None)
