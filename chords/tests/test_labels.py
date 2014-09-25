"""
"""

import unittest
import dl4mir.chords.labels as L


class LabelsTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_subtract_mod12(self):
        self.assertEqual(L.subtract_mod12(0, 0), 0)
        self.assertEqual(L.subtract_mod12(0, 10), 10)
        self.assertEqual(L.subtract_mod12(10, 0), 2)
        self.assertEqual(L.subtract_mod12(10, 9), 11)
        self.assertEqual(L.subtract_mod12(25, 24), 35)


if __name__ == "__main__":
    unittest.main()
