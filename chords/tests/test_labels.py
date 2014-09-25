"""
"""

import unittest
import numpy as np
import dl4mir.chords.labels as L


class LabelsTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_subtract_mod12(self):
        self.assertEqual(L.subtract_mod(0, 0, 12), 0)
        self.assertEqual(L.subtract_mod(0, 10, 12), 10)
        self.assertEqual(L.subtract_mod(10, 0, 12), 2)
        self.assertEqual(L.subtract_mod(10, 9, 12), 11)
        self.assertEqual(L.subtract_mod(25, 24, 12), 35)
        self.assertEqual(L.subtract_mod(25, None, 12), None)
        self.assertEqual(L.subtract_mod(None, 24, 12), None)
        self.assertEqual(L.subtract_mod(3, 0, 12), 9)
        self.assertEqual(L.subtract_mod(3, 10, 12), 7)
        self.assertEqual(L.subtract_mod(15, 12, 12), 21)

    def test_add_mod12(self):
        self.assertEqual(L.add_mod(0, 0, 12), 0)
        self.assertEqual(L.add_mod(0, 10, 12), 10)
        self.assertEqual(L.add_mod(10, 0, 12), 10)
        self.assertEqual(L.add_mod(10, 9, 12), 7)
        self.assertEqual(L.add_mod(25, 11, 12), 24)
        self.assertEqual(L.add_mod(25, None, 12), None)
        self.assertEqual(L.add_mod(None, 24, 12), None)
        self.assertEqual(L.add_mod(3, 10, 12), 1)
        self.assertEqual(L.add_mod(15, 12, 12), 15)

    def test_compress_labeled_intervals(self):
        intervals = np.array([[0, 1.0], [1.0, 2.0], [2.0, 4.0]])
        labels = ['a', 'a', 'b']
        res = L.compress_labeled_intervals(intervals, labels)
        np.testing.assert_array_equal(res[0], np.array([[0, 2.0], [2.0, 4.0]]))
        self.assertEqual(res[1], ['a', 'b'])

if __name__ == "__main__":
    unittest.main()
