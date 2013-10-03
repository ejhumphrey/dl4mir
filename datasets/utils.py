"""
"""

import numpy as np
import os
import fnmatch
import time
from collections import OrderedDict
import json

def collect_nested_files(base_dir, extensions):
    """Collect a list of filepaths matching a given extension recursively.

    Parameters
    ----------
    base_dir : str
        Top-level directory to walk.
    extensions : list of strings
        List of file extensions to match.

    Returns
    -------
    matched_files : list of strings
        All filepaths that match the set of extensions.
    """
    matched_files = []
    for root, _dirs, files in os.walk(base_dir):
        # Iterate over valid extensions.
        for ext in extensions:
            # Files that match the current extension.
            these_files = fnmatch.filter(files, "*%s" % ext)
            # Expand the full filepath of all matched files.
            matched_files += [os.path.join(root, f) for f in these_files]

    return matched_files


def join_on_filebase(filepaths):
    """Given a set of files, group into lists by filebase (minus the extension).

    Parameters
    ----------
    filepaths : list
        Collection of string filepaths.

    Returns
    -------
    path_collections : list of lists
    """
    results = dict()
    for filepath in filepaths:
        filebase = os.path.splitext(filepath)[0]
        if not filebase in results:
            results[filebase] = []
        results[filebase].append(filepath)

    return results.values()


def stratify_GA(X,
                pop_size,
                n_iter,
                num_folds=5,
                fitness_penalty=4.0,
                mutation_rate=0.005,
                VERBOSE=False):
    """
    """

    def softmax(x):
        y = np.exp(x)
        return y / y.sum()

    def fitness(X, p_set):
        fit = np.zeros(len(p_set))
        for p in range(len(p_set)):
            Z = np.dot(p_set[p], X.transpose()[:, :, np.newaxis]).squeeze()
            Z /= Z.sum(axis=0)[np.newaxis, :]
            fit[p] = Z.std()

        return fit

    def fitness2(X, p_set):
        fit = np.zeros(len(p_set))
        for p in range(len(p_set)):
#            print p_set[p].shape,X[:,:,:,np.newaxis].shape
            Z = np.dot(p_set[p], X[:, :, :, np.newaxis]).squeeze().reshape(num_folds, 625)
            s_norm = Z.sum(axis=0)[np.newaxis, :]
            s_norm[s_norm == 0] = 1.0
            Z /= s_norm
            fit[p] = Z.std()

        return fit

    N = np.max(X.shape)
    n = 0
    DONE = False
    pop_set = []

    best_fit = np.inf
    best_W = None

    # Init population
    for _p in range(pop_size):
        W = np.zeros([5, 1, N])
        W[np.random.randint(0, 5, (N,)), 0, np.arange(N, dtype=int)] = 1.0
        pop_set += [W]

    # ...and run!
    while not DONE:
        # Evaluate Set
        fit = fitness2(X, pop_set)

        if fit.min() < best_fit:
            best_fit = fit.min()
            best_W = pop_set[fit.argmin()]
            if VERBOSE:
                print "%6d\tNew best: %0.5f" % (n, best_fit)

        # Fitness -> Probability
        # - normalize
        fit -= fit.min()
        scale = fit.max()
        if scale == 0:
            DONE = True
            break
        fit /= scale

        # - penalize, flip and cdf
        cdf = np.zeros(pop_size + 1)
        cdf[1:] = softmax(-fitness_penalty * fit)
        cdf = np.cumsum(cdf)

        # Pair and merge survivors
        children = []
        for p in range(pop_size):
            idx1 = (cdf <= np.random.rand()).argmin() - 1
            idx2 = idx1
            while idx2 == idx1:
                idx2 = (cdf <= np.random.rand()).argmin() - 1

            W1 = pop_set[idx1]
            W2 = pop_set[idx2]
            W3 = W1.copy()
            gene_idx = np.random.randint(0, 2, (N,)).astype(bool)
            W3[:, 0, gene_idx] = W2[:, 0, gene_idx]
            children += [W3]

        n += 1

        pop_set = children
        if n > n_iter:
            DONE = True

        #if all Wi==Wj, done
#        return pop_set, fitness(X,pop_set)
    return best_W, best_fit

def split_folds(fold_map, train, valid, test):
    """
    Parameters
    ----------
    fold_map : dict
    train : list of ints
    valid : list of ints
    test : list of ints

    Returns
    -------
    splits : dict of lists
        Has the keys 'train', 'valid', and 'test'
    """
    fold_indexes = {'train':train, 'valid':valid, 'test':test}
    splits = {'train':[], 'valid':[], 'test':[]}
    for key, fold in fold_map.iteritems():
        for name, fold_set in fold_indexes.iteritems():
            if fold in fold_set:
                splits[name].append(key)
                break

    return splits


def filebase(filepath):
    """For a full filepath like '/path/to/some/filename.xyz', return 'filename'.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    filebase : str
    """
    return os.path.splitext(os.path.split(filepath)[-1])[0]

def expand_filebase(filebase, output_dir, ext):
    """

    Parameters
    ----------
    filebase : str
    output_dir : str
    ext : str

    Returns
    -------
    filepath : str
    """
    ext = ext.strip(".")
    return os.path.join(output_dir, "%s.%s" % (filebase, ext))


def build_label_map(unique_labels):
    """
    genres : set or list
        Items must be unique.
    """
    return dict([(label, enum) for enum, label in enumerate(unique_labels)])

def stratify_labels(labels, num_folds):
    """
    """
    label_map = build_label_map(set(labels))
    num_labels = len(label_map)
    label_enum = np.array([label_map.get(l) for l in labels], dtype=int)
    num_points = len(label_enum)
    fold_assignment = np.zeros(label_enum.shape, dtype=int)
    all_idx = np.arange(num_points)
    for enum in range(num_labels):
        is_enum = label_enum == enum
        num_label = np.sum(is_enum)
        enum_idx = all_idx[is_enum] #The positions of this enum
        fold_assignment[enum_idx] = np.arange(num_label) % num_folds

    return fold_assignment

def timestamp():
    """Returns a string representation of the time, like 'YYYYMMDD'.
    """
    TIME_FMT = "%Y%m%d"
    return time.strftime(TIME_FMT)

def load_label_enum_map(filepath):
    """JSON refuses to store integer zeros, so they are written as strings and
    interpreted as integers on load.
    """
    return OrderedDict([(k, int(v)) for k, v in json.load(open(filepath)).iteritems()])

def save_label_enum_map(label_map, filepath):
    """JSON refuses to store integer zeros, so they are written as strings and
    interpreted as integers on load.
    """
    json.dump(dict([(k, "%d" % v) for k, v in label_map.iteritems()]),
              open(filepath, 'w'),
              indent=2)

