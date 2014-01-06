"""
"""

import os
import tempfile
import unittest

import h5py
import numpy as np

from .. import ReservedKeys
from ..  import core

def safe_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)

class Test(unittest.TestCase):
    tmpfile = tempfile.mktemp(suffix=".hdf5", dir=tempfile.tempdir)

    def setUp(self):
        safe_remove(Test.tmpfile)
        self.file = h5py.File(Test.tmpfile)
        self.value = np.random.uniform(size=(5, 5))
        self.label = "abcdef"
        self.target = 56
        self.metadata = dict(filename=Test.tmpfile)
        self.key = "00/00/13"
        dset = self.file.create_dataset(name=self.key, data=self.value)
        [dset.attrs.create(name=k, data=v) for k, v in self.metadata.iteritems()]
        dset.attrs[ReservedKeys.LABEL] = self.label
        dset.attrs[ReservedKeys.TARGET] = self.target

    def tearDown(self):
        safe_remove(Test.tmpfile)


    def test_datapoint_parsing(self):
        datapoint = core.DataPoint.from_file(self.file.get(self.key))
        self.assertEqual(
            datapoint.label(), self.label, "Labels do not match.")
        self.assertEqual(
            datapoint.target(), self.target, "Targets do not match.")
        np.testing.assert_array_equal(
            datapoint.value(), self.value, "Value arrays do not match.")
        self.assertEqual(
            datapoint.metadata(), self.metadata, "Metadata does not match.")
        self.assertEqual(
            datapoint.name(), self.key, "Names do not match.")

    def test_datasequence(self):
        x = np.arange(10)
        datapoint = core.DataSequence(value=x)
        np.testing.assert_array_equal(
            datapoint.value(), x, "Full value arrays do not match.")
        np.testing.assert_array_equal(
            datapoint.value(4), x[4:5], "Single indexing does not match.")
        np.testing.assert_array_equal(
            datapoint.value(4, 2, 2), x[2:7], "In-range slice does not match.")
        np.testing.assert_array_equal(
            datapoint.value(0, 2, 2),
            np.concatenate([np.zeros(2), x[:3]]),
            "Front out-of-bounds slice does not match.")
        np.testing.assert_array_equal(
            datapoint.value(9, 2, 2),
            np.concatenate([x[7:], np.zeros(2)]),
            "Back out-of-bounds slice does not match.")



if __name__ == "__main__":
    unittest.main()
