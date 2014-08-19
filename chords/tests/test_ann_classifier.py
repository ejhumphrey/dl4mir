import unittest
import numpy as np
from sklearn import datasets

from dl4mir.chords.nearest_neighbors import ANNClassifier


class ANNTests(unittest.TestCase):

    def setUp(self):
        self.centers = np.array([[-1, 1],[1, -1]])
        self.cluster_std = (.25, .25)
        self.X, self.y = datasets.make_blobs(
            1000, 2, centers=self.centers, cluster_std=self.cluster_std)

    def tearDown(self):
        pass

    def test_1nn(self):
        ann = ANNClassifier(1)
        ann.fit(self.X, self.y)

        X, y = datasets.make_blobs(
            1000, 2, centers=self.centers, cluster_std=self.cluster_std)

        y_pred = ann.predict(X)
        self.assertEqual(np.equal(y, y_pred).mean(), 1.0)

    def test_knn(self):
        ann = ANNClassifier(5)
        ann.fit(self.X, self.y)

        X, y = datasets.make_blobs(
            10, 2, centers=self.centers, cluster_std=self.cluster_std)

        y_pred = ann.predict(X)
        self.assertEqual(np.equal(y, y_pred).mean(), 1.0)

if __name__ == "__main__":
    unittest.main()
