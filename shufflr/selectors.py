"""Classes for selecting data from sources.
"""

import numpy as np

from random import choice

from ejhumphrey.shufflr import keyutils


def split_index_table(index_table):
    split_table = dict()
    indexes, label_enum = index_table[:, :-1], index_table[:, -1]
    for i in np.unique(label_enum):
        split_table[i] = indexes[label_enum == i]
    return split_table


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


class UniformLabel(BaseSelector):

    def __init__(self, source, equivalence_map=None):
        self._source = source
        self._keys = source.keys()
        self.depth = self._keys[0].count('/') + 1
        label_enum = source.label_enum()
        index_table = source.index_table()
        if equivalence_map:
            # Create mutable enumeration data structure
            new_enum = np.zeros(len(index_table), dtype=index_table.dtype) - 1
            # Translate the equiv map into enumeration values.
            enum_equiv = dict([(label_enum.get(l, -1), i)
                              for l, i in equivalence_map.iteritems()])
            for i, j in enum_equiv.iteritems():
                new_enum[index_table[:, -1] == i] = j
            # Replace old enumeration with new equivalence-based one.
            index_table[:, -1] = new_enum

        self._index_table = split_index_table(index_table)
        self._index_table.pop(-1, None)
        self._label_keys = self._index_table.keys()
        self.reset()

    def reset(self):
        self._idx = 0
        np.random.shuffle(self._label_keys)

    def next(self):
        label_key = self._label_keys[self._idx]
        self._idx += 1
        if self._idx >= len(self._label_keys):
            self.reset()
        key = keyutils.index_to_key(
            choice(self._index_table[label_key]), self.depth)
        return key, self._source.get(key)

"""
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
"""