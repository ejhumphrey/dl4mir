import unittest
import numpy as np
import numpy.testing as nptest
import dl4mir.chords.data as D

class DataTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_extract_tile(self):
        x_in = np.arange(10)[:, np.newaxis].astype(int)

        nptest.assert_array_equal(
            D.extract_tile(x_in, 0, 4),
            np.array([0, 0, 0, 1])[:, np.newaxis])

        nptest.assert_array_equal(
            D.extract_tile(x_in, 0, 5),
            np.array([0, 0, 0, 1, 2])[:, np.newaxis])

        nptest.assert_array_equal(
            D.extract_tile(x_in, 4, 4),
            np.array([2, 3, 4, 5])[:, np.newaxis])

        nptest.assert_array_equal(
            D.extract_tile(x_in, 9, 4),
            np.array([7, 8, 9, 0])[:, np.newaxis])

        nptest.assert_array_equal(
            D.extract_tile(x_in, 9, 5),
            np.array([7, 8, 9, 0, 0])[:, np.newaxis])


if __name__ == "__main__":
    unittest.main()
