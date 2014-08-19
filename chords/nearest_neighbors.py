from sklearn.decomposition import PCA
import annoy
import numpy as np
import biggie
import joblib


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
                print y
                y_pred[idx] = y
        return y_pred[:, 0]

    def __del__(self):
        pass


def build_knn_classifier(centers, class_idxs, counts, n_components=400,
                         ann_trees=50, mode='sorted', verbose=True):
    x_true = np.concatenate([c.reshape(len(c), np.prod(c.shape[1:]))
                             for c in centers], axis=0)
    y_true = np.concatenate([np.zeros(len(c), dtype=int) + i
                             for c, i in zip(centers, class_idxs)])
    if verbose:
        print "Fitting PCA..."
    pca = PCA(n_components).fit(x_true)
    x_true = pca.transform(x_true)





def main(args):
    class_data = np.load(args.centers_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("stash_file",
                        metavar="stash_file", type=str,
                        help="Stash file.")
    parser.add_argument("centers_file",
                        metavar="centers_file", type=str,
                        help="Class centers in a npz archive.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="JSON Output.")

    main(parser.parse_args())
