"""
"""

import numpy as np
from random import choice


def permute(keys):
    """writeme"""
    order = np.arange(len(keys))
    np.random.shuffle(order)
    idx = 0
    while True:
        yield keys[order[idx]]
        idx += 1
        if idx >= len(keys):
            idx = 0
            np.random.shuffle(order)


class Cache(dict):
    """Provides an in-memory sampling interface.

    """
    def __init__(self, source, batch_size, transformers=None,
                 cache_size=1000, refresh_prob=0.25):
        """
        Parameters
        ----------
        source : dict-like, with .keys() and .get(key) methods
            A data source with which to populate the deck.
        batch_size: int
            Number of datapoints to store in the queue.
        transformers : List of data transformers
            Consume an object, return an object
        refresh_prob : float, in [0, 1]
            Probability a cached item may be dropped and replaced.
        cache_size : int
            Number of items to maintain in the cache.
        """

        self._source = source
        self.batch_size = batch_size
        if transformers is None:
            transformers = list()
        self.transformers = transformers

        # If the dataset can fit in the cache, do it, and disable replacement.
        if len(self._source) < cache_size:
            cache_size = len(self._source)
            refresh_prob = 0
        self._cache_size = cache_size
        self._refresh_prob = refresh_prob

        # Post init, load things.
        self._source_selector = permute(source.keys())
        self._key_selector = None
        self.load()

    def _refresh_rand(self, p=None):
        """Randomly select a key to drop with probability 'p'."""
        # Fall back to default probability in the absence of an arg.
        self._refresh(key=choice(self.keys()), p=p)

    def _refresh(self, key, p=None):
        """Swap an existing key-value pair with a new one."""
        if p is None:
            p = self._refresh_prob
        # Refresh on success.
        if np.random.binomial(1, p=p):
            self.remove(key)
            self.load()

    def load(self):
        """(re)Fill the cache."""
        for key in self._source_selector:
            if len(self) >= self._cache_size:
                break
            self[key] = self._source.get(key)
        # The cache has been modified; re-init selector.
        self._key_selector = permute(self.keys())

    def remove(self, key):
        """like pop, but without a return"""
        del self[key]
        # The cache has been modified; re-init selector.
        self._key_selector = permute(self.keys())

    def refresh_queue(self):
        """writeme"""
        self.clear_queue()
        for n, key in enumerate(self._key_selector):
            if n == self.batch_size:
                break
            obj = self.get(key)
            for fx in self.transformers:
                obj = fx(obj)
            self.add_to_queue(obj)
            self._refresh(key, self._refresh_prob)

    def add_to_queue(self, obj):
        """asdf"""
        raise NotImplementedError("Subclass me to parse the object %s" % obj)

    def clear_queue(self):
        """asdf"""
        raise NotImplementedError("Subclass me to clear the batch queue")


class Test(Cache):
    """asdf"""
    def __init__(self, source, batch_size, transformers=None,
                 cache_size=1000, refresh_prob=0.25):
        Cache.__init__(self, source, batch_size, transformers=None,
                       cache_size=1000, refresh_prob=0.25)
        self._chroma = []
        self._mfccs = []

    def clear_queue(self):
        """asdf"""
        self._chroma = []
        self._mfccs = []

    def add_to_queue(self, obj):
        """asdf"""
        self._chroma.append(choice(obj.chroma.value))
        self._mfccs.append(choice(obj.mfcc.value))

    def chroma(self):
        """asdf"""
        return np.asarray(self._chroma)

    def mfccs(self):
        """asdf"""
        return np.asarray(self._mfccs)
