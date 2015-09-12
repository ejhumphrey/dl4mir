"""
"""

import unittest

import numpy as np
import optimus
import biggie

import dl4mir.common.convolve_graph_with_dset as C


class TransformStashTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convolve(self):
        input_data = optimus.Input(name='x_in', shape=(None, 1, 1, 1))
        flatten = optimus.Flatten(name='flatten', ndim=1)
        output_data = optimus.Output(name='x_out')
        edges = optimus.ConnectionManager([
            (input_data, flatten.input),
            (flatten.output, output_data)])

        transform = optimus.Graph(
            name='test',
            inputs=[input_data],
            nodes=[flatten],
            connections=edges.connections,
            outputs=[output_data])

        x = np.arange(10).reshape(1, 10, 1)
        y = np.array(['a', 'b'])
        entity = biggie.Entity(x_in=x, y=y)
        z = C.convolve(entity, transform, 'x_in')

        np.testing.assert_equal(z.x_out, np.arange(10))
        np.testing.assert_equal(z.y, y)

if __name__ == "__main__":
    unittest.main()
