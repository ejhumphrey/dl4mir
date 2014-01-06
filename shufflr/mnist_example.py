'''
Created on Sep 25, 2013

@author: ejhumphrey
'''

import cPickle

from marl.hewey.core import DataPoint
from marl.hewey.core import DataSequence
from marl.hewey.file import DataPointFile
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen

mnist_pkl_file = "/Users/ejhumphrey/Desktop/mnist.pkl"
hewey_filebase = '/Volumes/speedy/mnist'

def create_datapoint_file(filename):
    mnist = cPickle.load(open(mnist_pkl_file))
    file_handle = DataPointFile(filename)
    keygen = uniform_keygen(2)
    # Training set is the first item.
    for x, y in zip(mnist[0][0], mnist[0][1]):
        dpoint = DataPoint(value=x.reshape(28, 28), label="%d" % y)
        file_handle.write(keygen.next(), dpoint)

    file_handle.create_tables()

def create_datasequence_file(filename):
    mnist = cPickle.load(open(mnist_pkl_file))
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)
    # Training set is the first item.
    for x, y in zip(mnist[0][0], mnist[0][1]):
        dseq = DataSequence(value=x.reshape(28, 28), label=["%d" % y] * 28)
        file_handle.write(keygen.next(), dseq)

    file_handle.create_tables()


def main():
    create_datapoint_file(hewey_filebase + ".dpf")
    create_datasequence_file(hewey_filebase + ".dsf")

if __name__ == '__main__':
    pass
