"""

"""

import numpy as np
import os

from ejhumphrey.datasets import utils
from scipy.io.matlab.mio import loadmat
from marl.hewey.core import DataSequence
import time

def filename_to_genre(filename):
    """Extract the LMD genre from a filename.

    Expects a filename like:
         '/Volumes/Audio/LMD/LMD240x10_mid/LMD-Axe-001.mat'
    and returns:
        'Axe'
    """
    return os.path.split(filename)[-1].split('-')[1]


def stratify_matfiles(mat_files, num_folds=10):
    """Create a mapping from file to fold assignment.
    """
    labels = [filename_to_genre(f) for f in mat_files]
    fold_assignments = utils.stratify_labels(labels, num_folds)
    return dict([(f, fold) for f, fold in zip(mat_files, fold_assignments)])


def generate_all_splits(mat_files, output_dir, num_folds):
    fold_map = stratify_matfiles(mat_files, num_folds)
    for n in range(num_folds):
        test_fold = [n]
        valid_fold = [(n + 1) % num_folds]
        train_folds = range(num_folds)
        train_folds.remove(test_fold[0])
        train_folds.remove(valid_fold[0])
        splits = utils.split_folds(
            fold_map, train=train_folds, valid=valid_fold, test=test_fold)
        for split, file_list in splits.iteritems():
            filepath = os.path.join(output_dir,
                                    "%s%02d_%s.txt" % (split,
                                                       n,
                                                       utils.timestamp()))
            file_handle = open(filepath, "w")
            [file_handle.write("%s\n" % f) for f in file_list]
            file_handle.close()


def matfile_to_datasequence(mat_file, stdev_params):
    """
    Parameters
    ----------
    mat_file : strings
        Path to a matfile.
    stdev_params : np.ndarray
        The first dimension is mean, the second standard deviation.

    Returns
    -------
    dseq : hewey.core.DataSequence
        Populated DataSequence object.
    """
    data = loadmat(mat_file)
    op_matrix = data.get("features")
    shp = op_matrix.shape
    op_matrix = np.reshape(op_matrix, newshape=(np.prod(shp[:2]), shp[-1])).T
    op_matrix = (op_matrix - stdev_params[0]) / stdev_params[1]
    labels = [str(data.get('genre')[0])] * len(op_matrix)
    metadata = {"timestamp": time.asctime(),
                "filesource": mat_file}
    return DataSequence(value=op_matrix, label=labels, metadata=metadata)

