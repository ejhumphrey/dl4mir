"""write me.
"""

import numpy as np
from collections import OrderedDict


class LabelBatch(OrderedDict):
    """
    """
    def __init__(self, source, batch_size, label_key, value_shape):
        OrderedDict.__init__(self)
        self.source = source
        self._batch_size = batch_size
        self._label_key = label_key
        self._value_shape = value_shape

    def refresh(self):
        self.clear()
        self.load(num_items=self._batch_size)

    def values(self):
        return np.array([np.reshape(self[k].value, self._value_shape)
                         for k in self])

    def labels(self):
        return np.array([self[k].labels[self._label_key] for k in self])

    def load(self, num_items=1):
        """Load the next 'num_items'."""
        # while num_items > 0:
            # k, v = self.source.next()
            # self[k] = v
            # num_items -= 1
        for n in xrange(num_items):
            self.update(dict([self.source.next()]))


class PairedBatch(LabelBatch):
    """
    """
    def values_A(self):
        return self.values()[self._idx_A]

    def values_B(self):
        return self.values()[self._idx_B]

    def equals(self):
        return np.equal(self.labels()[self._idx_A],
                        self.labels()[self._idx_B]).astype(float)

    def refresh(self):
        LabelBatch.refresh(self)
        self._pair()

    def _pair(self):
        self._idx_A, self._idx_B = [], []
        N = len(self)
        labels = self.labels()
        for n in range(self._batch_size/2):
            y_idx = np.random.randint(N)
            possible_idx = list(np.arange(N)[labels == labels[y_idx]])
            possible_idx.remove(y_idx)
            self._idx_A.append(y_idx)
            self._idx_B.append(choice(possible_idx))

        for n in range(self._batch_size/2):
            y_idx = np.random.randint(N)
            possible_idx = list(np.arange(N)[labels != labels[y_idx]])
            self._idx_A.append(y_idx)
            self._idx_B.append(choice(possible_idx))

        M = min([len(self._idx_A), len(self._idx_B)])
        self._idx_A = np.asarray(self._idx_A)[:M]
        self._idx_B = np.asarray(self._idx_B)[:M]


class SimpleBatch(list):
    """writeme"""
    def __init__(self, values, labels, batch_size):
        list.__init__(self)
        self._values = np.asarray(values)
        self._labels = np.asarray(labels)
        assert len(self._values) == len(self._labels)
        self._batch_size = batch_size
        self._order = np.random.permutation(len(self))
        self._idx = 0
        self.reset()
        self.refresh()

    def __len__(self):
        return self._labels.shape[0]

    def reset(self):
        """writeme"""
        np.random.shuffle(self._order)
        self._idx = 0

    def refresh(self):
        """Not sure I like the disctinction between loading and refreshing the
        batch. Additionally, perhaps this should be update instead of refresh?
        """
        list.__init__(self)
        self.load(num_items=self._batch_size)

    def load(self, num_items=1):
        """Load the next 'num_items'."""
        # while num_items > 0:
            # k, v = self.source.next()
            # self[k] = v
            # num_items -= 1
        while num_items:
            idx = self._order[self._idx]
            self.append((self._values[idx], self._labels[idx]))

            num_items -= 1
            self._idx += 1
            if self._idx > len(self._order):
                self.reset()

    def values(self):
        """writeme"""
        return np.array([item[0] for item in self])

    def labels(self):
        """writeme"""
        return np.array([item[1] for item in self])
