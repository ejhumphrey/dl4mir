import annoy
import argparse
import numpy as np
import biggie
import joblib
import json
import time
from sklearn.decomposition import PCA

import marl.fileutils as futil
from optimus.util import array_stepper
import dl4mir.chords.labels as L

NUM_CPUS = 8


class ANNClassifier(object):
    def __init__(self, n_neighbors=1, trees=50, index_file=None):
        self.n_neighbors = n_neighbors
        self.index_ = None
        self._index_file = index_file
        self._class_labels = list()
        self.n_dim = None
        self.trees = 50

    def load(self, index_file):
        pass

    def save(self, index_file):
        pass

    def fit(self, x, y):
        assert x.ndim == 2
        assert len(x) == len(y)
        self._class_labels = np.array(y)

        n_obs, self.n_dim = x.shape
        self.index_ = annoy.AnnoyIndex(self.n_dim)
        for idx, xi in enumerate(x):
            self.index_.add_item(idx, xi.tolist())
        self.index_.build(self.trees)

    def predict(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        assert x.ndim == 2
        assert x.shape[1] == self.n_dim

        y_idx = np.array([self.index_.get_nns_by_vector(xi.tolist(),
                                                        self.n_neighbors)
                          for xi in x], dtype=int)
        y_pred = self._class_labels[y_idx]
        if self.n_neighbors > 1:
            for idx in range(len(y_pred)):
                counts = np.bincount(y_pred[idx])
                winning_labels = np.arange(len(counts))[counts == counts.max()]
                if len(winning_labels) > 1:
                    # Tie-breaker
                    for y in y_pred[idx]:
                        if y in winning_labels:
                            break
                else:
                    y = winning_labels[0]
                y_pred[idx] = y
        return y_pred[:, 0]

    def __del__(self):
        pass


def build_model(centers, class_idxs, n_components=400, n_neighbors=3,
                verbose=True):
    X = np.concatenate([c.reshape(len(c), np.prod(c.shape[1:]))
                        for c in centers], axis=0)
    y = np.concatenate([np.zeros(len(c), dtype=int) + i
                        for c, i in zip(centers, class_idxs)])

    if verbose:
        print "Fitting PCA..."
    pca = PCA(n_components).fit(X)
    X = pca.transform(X)

    ann = ANNClassifier(n_neighbors=n_neighbors)
    ann.fit(X, y)
    return pca, ann


def predict(cqt, pca, ann, win_length=20):
    X = np.array([x.flatten() for x in array_stepper(cqt, win_length,
                                                     axis=1, mode='same')])
    return ann.predict(pca.transform(X))


def predict_all(stash, pca, ann):
    predictions = dict()
    for n, key in enumerate(stash.keys()):
        x = stash.get(key)
        y_true = np.array(
            L.chord_label_to_class_index(x.chord_labels, 157))
        y_pred = predict(x.cqt, pca, ann)
        predictions[key] = y_pred.tolist()
        valid_idx = np.not_equal(y_true, None)
        if valid_idx.sum() > 0:
            score = np.equal(y_true[valid_idx], y_pred[valid_idx]).mean()
        else:
            score = 0
        print "[%s] %4d: %s (%0.4f)" % (time.asctime(), n, key, score)
    return predictions


def main(args):
    class_data = np.load(args.centers_file)
    pca, ann = build_model(
        centers=class_data['centers'], class_idxs=class_data['chord_idx'],
        n_neighbors=args.n_neighbors, n_components=args.n_components)
    pool = joblib.Parallel(n_jobs=NUM_CPUS)
    stash = biggie.Stash(args.stash_file)
    predictions = predict_all(stash, pca, ann)
    with open(args.output_file, 'w') as fp:
        json.dump(predictions, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("stash_file",
                        metavar="stash_file", type=str,
                        help="Stash file.")
    parser.add_argument("centers_file",
                        metavar="centers_file", type=str,
                        help="Class centers in a npz archive.")
    parser.add_argument("n_neighbors",
                        metavar="n_neighbors", type=int,
                        help="Number of neighbors to consider.")
    parser.add_argument("n_components",
                        metavar="n_components", type=int,
                        help="Number of components to keep for the PCA redux.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="JSON Output.")

    main(parser.parse_args())
