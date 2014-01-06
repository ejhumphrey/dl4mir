"""
TODO(ejhumphrey): Missing DataSequenceFile tests!
"""

import os
import tempfile
import unittest

import numpy as np

from .. import keyutils
from ..file import DataPointFile
# from ..file import DataSequenceFile

from ..utils import max_depth
from ..utils import safe_remove
from ..utils import populate_datapointfile
# from ..utils import populate_datasequencefile


class Test(unittest.TestCase):
    tmpfile = tempfile.mktemp(suffix=".hdf5", dir=tempfile.tempdir)

    def setUp(self):
        safe_remove(Test.tmpfile)
        self.dpfile = DataPointFile(Test.tmpfile)

        # Fake data params.
        num_items = 25
        shape = (5,)
        label_max = 10
        depth = max_depth(num_items)
        populate_datapointfile(
            self.dpfile, num_items, shape, label_max=label_max)

        # Expected outputs.
        self.index_table = np.array(
            [np.arange(num_items), np.arange(num_items) % label_max]).T
        self.label_enum = dict([("%d" % n, n) for n in range(label_max)])
        self.keys = [keyutils.index_to_key(n, depth) for n in range(num_items)]

    def tearDown(self):
        safe_remove(Test.tmpfile)

    def test_datapointfile(self):

        self.assertEqual(self.keys, self.dpfile.keys(), "Keys mismatch.")
        self.assertEqual(self.label_enum,
                         self.dpfile.label_enum(),
                         "Label enumeration mismatch.")
        np.testing.assert_array_equal(self.index_table,
                                      self.dpfile.index_table(),
                                      "Index table mismatch.")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
