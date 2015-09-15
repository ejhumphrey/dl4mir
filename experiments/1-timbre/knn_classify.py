"""Train a kNN classifier over a biggie stash."""

from __future__ import print_function
import argparse
import biggie
import json
import os
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

import dl4mir.timbre.data as D
import dl4mir.common.fileutil as futil


def classify(train, valid, test, num_neighbors=[1, 5, 9, 15, 21, 35, 51, 75],
             num_train=25000, num_valid=2500, num_test=25000):
    x_train, y_train = D.sample_embedding_stash(train, num_train)[:2]
    x_valid, y_valid = D.sample_embedding_stash(valid, num_valid)[:2]
    x_test, y_test = D.sample_embedding_stash(test, num_test)[:2]

    best_knn = None
    best_score = -1
    for k in num_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
        score = knn.score(x_valid, y_valid)
        if score > best_score:
            best_knn = knn
            best_score = score
            print("[{0}] New Best @ k={1}: {2:.4}"
                  "".format(time.asctime(), k, best_score))

    proba_test = knn.predict_proba(x_test)
    Y_test = label_binarize(y_test, classes=best_knn.classes_)

    # Compute micro-average ROC curve and ROC area
    precision_micro, recall_micro, _ = precision_recall_curve(
        Y_test.ravel(), proba_test.ravel())

    average_precision_micro = average_precision_score(
        Y_test, proba_test, average="micro")
    conf_mat = confusion_matrix(y_test, best_knn.predict(x_test))
    return dict(
        average_precision_micro=average_precision_micro,
        precision_micro=precision_micro.tolist(),
        recall_micro=recall_micro.tolist(),
        confusion_matrix=conf_mat.tolist(),
        class_labels=best_knn.classes_.tolist(),
        train_score=best_knn.score(x_train, y_train),
        valid_score=best_knn.score(x_valid, y_valid),
        test_score=best_knn.score(x_test, y_test))


def main(args):
    fpath = os.path.join(args.data_directory, "{0}.hdf5")
    train = biggie.Stash(fpath.format('train'), cache=True)
    valid = biggie.Stash(fpath.format('valid'), cache=True)
    test = biggie.Stash(fpath.format('test'), cache=True)
    results = classify(train, valid, test,
                       num_train=50000, num_valid=10000, num_test=25000)

    for k in 'train', 'valid', 'test':
        print("{0}: {1:.4}".format(k, results['{0}_score'.format(k)]))

    output_dir = os.path.split(args.stats_file)[0]
    futil.create_directory(output_dir)
    with open(args.stats_file, 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("data_directory",
                        metavar="training_file", type=str,
                        help="Path to a biggie Stash file for training.")
    # Outputs
    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="Name for the resulting predictor graph.")
    main(parser.parse_args())
