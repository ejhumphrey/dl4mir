"""Classes for selecting data from sources.
"""

import numpy as np
from . import core
from . import utils


class BaseSelector(object):
    def __init__(self, source):
        self._source = source
        self._keys = self._source.keys()
        self.num_items = len(self._keys)
        self.reset()

    def reset(self):
        self._idx = 0

    def next(self):
        key = self._keys[self._idx]
        self._idx += 1
        if self._idx > len(self._keys):
            self.reset()
        return key, self._source.get(key)


class Permutation(BaseSelector):
    def reset(self):
        self._idx = 0
        np.random.shuffle(self._keys)


class Random(BaseSelector):
    def next(self):
        key = self._keys[np.random.randint(len(self._keys))]
        return key, self._source.get(key)


class RandomSequenceSample(Random):
    def __init__(self, source, left, right):
        Random.__init__(self, source)
        self.left = left
        self.right = right

    def next(self):
        k, v = Random.next(self)
        idx = np.random.randint(len(v))
        value = utils.context_slice(v.value, idx, self.left, self.right, 0.0)
        labels = dict([(k, l[idx]) for k, l in v.labels.iteritems()])
        targets = dict([(k, t[idx]) for k, t in v.targets.iteritems()])
        metadata = v.metadata.copy()
        metadata.update(idx=idx)
        return k, core.Sample(name=k,
                              value=value,
                              labels=labels,
                              targets=targets,
                              metadata=metadata)
