import numpy as np

import dl4mir.common.util as U


def test_viterbi():
    idx = np.array([0, 0, 1, 1, 2])
    posterior = np.zeros([5, 3])
    posterior[np.arange(len(posterior)), idx] = 1.0
    posterior[3, :2] = np.array([0.8, 0.2])
    trans_mat = np.ones([3, 3])

    max_idx = np.array([0, 0, 1, 0, 2])
    np.testing.assert_equal(posterior.argmax(axis=1), max_idx)

    # Make sure that the self-transition penalty survives the gap.
    vit_idx = U.viterbi(posterior, trans_mat, penalty=-10)
    np.testing.assert_equal(vit_idx, idx)


def test_stratify():
    num_items = 100
    items = range(num_items)
    num_folds = 5
    valid_ratio = 0.25
    num_test = len(items) / num_folds
    num_valid = int(valid_ratio * (num_items - num_test))
    num_train = num_items - num_valid - num_test
    sizes = dict(train=num_train, valid=num_valid, test=num_test)
    folds = U.stratify(items, num_folds, valid_ratio)
    for fidx, split in folds.items():
        # Check the sizes
        for name in sizes:
            assert len(split[name]) == sizes[name]
        # Ensure disjoint sets
        for a, x in split.items():
            for b, y in split.items():
                if a != b:
                    assert not set(x).intersection(set(y))
