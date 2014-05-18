"""
"""

import numpy as np

from ejhumphrey import deeznets
from ejhumphrey.shufflr import batches
from ejhumphrey.deeznets import Constant, Variable
from ejhumphrey.shufflr.mnist_example import load_mnist

# Constants
MNIST_PKL = "/Users/ejhumphrey/Desktop/mnist.pkl"

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

# Training Data Presentation
train = load_mnist(MNIST_PKL)['train']
batch = batches.SimpleBatch(values=train['values'][:, np.newaxis, :, :],
                            labels=train['labels'],
                            batch_size=50)

data_sources = [Variable('x', batch.values, batch.refresh),
                Variable('class_idx', batch.labels),
                Constant('affine0.dropout', 0),
                Constant('conv0.dropout', 0),
                Constant('weight_decay:conv0.weights', 0),
                Constant('sparsity:affine0_out', 0),
                Constant('learning_rate:conv0.bias', 0.01),
                Constant('learning_rate:affine0.bias', 0.01),
                Constant('learning_rate:affine0.weights', 0.01),
                Constant('learning_rate:softmax0.bias', 0.01),
                Constant('learning_rate:conv0.weights', 0.01),
                Constant('learning_rate:softmax0.weights', 0.01)]

inputs = deeznets.DataServer(data_sources)

driver.train(inputs, 5000, 50)
