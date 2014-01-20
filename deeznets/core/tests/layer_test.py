'''
Created on Sep 23, 2013

@author: ejhumphrey
'''
import unittest

from ..layers import Conv3DArgs
from ..layers import AffineArgs
from ..layers import SoftmaxArgs
from ..layers import Layer

layer_0 = Layer(Conv3DArgs(name='convlayer0',
                           input_shape=(1, 28, 28),
                           weight_shape=(30, 5, 5),
                           pool_shape=(2, 2),
                           activation='tanh'))

layer_1 = Layer(Conv3DArgs(name='convlayer1',
                           input_shape=layer_0.output_shape,
                           weight_shape=(50, 7, 7),
                           pool_shape=(2, 2),
                           activation='tanh'))

layer_2 = Layer(AffineArgs(name='affine2',
                           input_shape=layer_1.output_shape,
                           output_shape=(128,),
                           activation='tanh'))

classifier = Layer(SoftmaxArgs(name='classifier',
                               input_dim=layer_2.output_shape[0],
                               output_dim=10))

network_def = [layer_0, layer_1, layer_2, classifier]

class Test(unittest.TestCase):


    def testName(self):
        pass


if __name__ == "__main__":
    unittest.main()
