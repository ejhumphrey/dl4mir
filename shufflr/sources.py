"""
"""

from random import choice

import numpy as np

from .core import Batch
from . import keyutils
# from . import core


class Deck(dict):
    """Provides an in-memory data interface.

    Maintains a consistent interface with a Sequence/SampleFile.
    """

    def __init__(self, selector, refresh_prob=0.25, cache_size=1000):
        """
        Parameters
        ----------
        selector : Subclass of BaseSelector
            An instantiated selector.
        refresh_prob : float, in [0, 1]
            Probability a cached item may be dropped and replaced.
        cache_size : int
            Number of items to maintain in the cache.
        """
        self._refresh_prob = refresh_prob
        self.selector = selector
        # If the dataset can fit in the cache, do it, and disable replacement.
        if self.selector.num_items < cache_size:
            cache_size = self.selector.num_items
            refresh_prob = 0

        self.load(num_items=cache_size)

    def create_tables(self):
        raise NotImplementedError("No table methods yet")

    def refresh_rand(self, p=None):
        """Randomly select a key to drop with probability 'p'."""
        # Fall back to default probability in the absence of an arg.
        self.refresh(key=choice(self.keys()), p=p)

    def refresh(self, key, p=None):
        """Swap an existing key-value pair with a new one."""
        if p is None:
            p = self._refresh_prob
        # Refresh on success.
        if np.random.binomial(1, p=p):
            self.remove(key)
            self.load(1)

    def load(self, num_items=1):
        """Load the next 'num_items'."""
        while num_items > 0:
            k, v = self.selector.next()
            self[k] = v
            num_items -= 1

    def remove(self, key):
        del self[key]
        # Clean up indexes; or, at least delete tables.


class RandomSource(Deck):
    """DataSource that randomly selects entire values from the cache.
    """
    def drop(self, key):
        Deck.remove(self, key)
        self.selector.remove(key)

    def load(self, num_items=1):
        Deck.load(self, num_items=num_items)
        self.selector = keyutils.KeySelector(self.cache.keys())

    def next(self):
        key = self.selector.next()
        value = self.cache.get(key)
        self.refresh_key(key)
        return value

    def next_batch(self, batch_size):
        """
        Need to unpack here, values and labels
        """
        batch = Batch()
        for _n in range(batch_size):
            dpoint = self.next()
            x, y = dpoint.value(), dpoint.label()
            if self._value_shape:
                x = np.reshape(x, newshape=self._value_shape)
            batch.add_value(x)
            default_y = y
            if self._label_map:
                default_y = -1
            batch.add_label(self._label_map.get(y, default_y))
        return batch

class SequenceSampler(RandomSource):
    def __init__(self, dataset, left, right,
                 refresh_prob=0.25, cache_size=1000):
        RandomSource.__init__(self, dataset, refresh_prob=refresh_prob, cache_size=cache_size)
        self.value_left, self.value_right = left, right
        def noop(x):
            return x
        self._batch_transformer = noop

    def set_transformer(self, transformer):
        """Apply some pre-processing to each batch.
        """
        self._batch_transformer = transformer

    def next(self):
        datasequence = RandomSource.next(self)
        index = np.random.randint(low=0, high=len(datasequence.value()))
        return DataPoint(
                value=datasequence.value(index,
                                     left=self.value_left,
                                     right=self.value_right),
                label=datasequence.label(index, left=0, right=0)[0])

    def next_batch(self, batch_size):
        return self._batch_transformer(
            RandomSource.next_batch(self, batch_size))

class UniformSampler(RandomSource):
    def __init__(self, dataset, left, right, label_map, equivalence_map,
                 refresh_prob=0.0, cache_size=1000, MAX_LOCAL_INDEX=1000000):
        self._equivalence_map = equivalence_map
        self._MAX_LOCAL_INDEX = MAX_LOCAL_INDEX
        self._local_index_table = np.zeros([self._MAX_LOCAL_INDEX, 3], dtype=np.int16) - 1
        self.cache_key_enum = dict()
        self._key_enums = range(cache_size)
        self._cache_size = cache_size
        self._key_enum_idx = 0
        RandomSource.__init__(self, dataset, refresh_prob=refresh_prob, cache_size=cache_size)
        self.set_label_map(label_map)

        self.value_left, self.value_right = left, right
        def noop(x):
            return x
        self._batch_transformer = noop

    def update_index_table(self):
        idx = 0
        open_indexes = np.arange(len(self._local_index_table))[self._local_index_table[:, -1] == -1]
        for key, enum in self.cache_key_enum.iteritems():
            if enum in self._local_index_table[:, 0]:
                continue

            for n, label in enumerate(self.cache.get(key).label()):
                self._local_index_table[open_indexes[idx], :] = (enum, n, self._equivalence_map.get(label, -1))
                idx += 1
        self.item_selector = uniform_table_sampler(self._local_index_table)

    def load(self, num_items=1):
        RandomSource.load(self, num_items=num_items)
        assert len(self.cache) <= self._cache_size, \
            "Undefined behavior if cache is bigger than intended."
        for k in self.cache:
            if not k in self.cache_key_enum:
                self.cache_key_enum[k] = self._key_enums[self._key_enum_idx]
                self._key_enum_idx = (self._key_enum_idx + 1) % len(self.cache)

        self.cache_key_enum_reverse = dict([(v, k) for k, v in self.cache_key_enum.iteritems()])
        self.update_index_table()

    def drop(self, key):
        enum = self.cache_key_enum[key]
        self._local_index_table[[self._local_index_table[:, 0] == enum]] = -1
        del self.cache_key_enum[key]
        del self.cache_key_enum_reverse[enum]
        RandomSource.drop(self, key)


    def set_transformer(self, transformer):
        """Apply some pre-processing to each batch.
        """
        self._batch_transformer = transformer

    def next(self):
        enum, index, _label = self.item_selector.next()
        datasequence = self.cache.get(self.cache_key_enum_reverse[enum])
        x = DataPoint(
                value=datasequence.value(index,
                                     left=self.value_left,
                                     right=self.value_right),
                label=datasequence.label(index, left=0, right=0)[0])
        return x

    def next_batch(self, batch_size):
        self.refresh_rand()
        return self._batch_transformer(
            RandomSource.next_batch(self, batch_size))

class WeightedSampler(UniformSampler):

    def __init__(self, dataset, left, right, label_map, equivalence_map, weights,
        refresh_prob=0.0, cache_size=1000, MAX_LOCAL_INDEX=1000000):
        self.weights = weights
        UniformSampler.__init__(self, dataset, left, right, label_map, equivalence_map, refresh_prob=refresh_prob, cache_size=cache_size, MAX_LOCAL_INDEX=MAX_LOCAL_INDEX)

    def update_index_table(self):
        UniformSampler.update_index_table(self)
        self.item_selector = weighted_table_sampler(self._local_index_table, self.weights)
