"""Various data sources, both on-disk and in-memory.

"""

import h5py
import numpy as np

from collections import OrderedDict
from random import choice

from . import ReservedKeys
from . import core
from . import keyutils
from . import utils


class SampleFile(h5py.File):
    """Object for efficiently reading and writing data on-disk.

    This object should cache the following:
      - label_index : Integer array of (index, label_enum) pairs.
      - key_manifest : List of all keys in this file.
      - label_enum : Ordered array of strings.

    Any time the file is modified (write/remove), both local and persistent
    data should be cleared.
    """

    def __init__(self, filepath):
        """
        filepath : str
            Path to file.
        """
        h5py.File.__init__(self, filepath, mode=None, driver=None, libver=None)
        self._filepath = filepath
        self._clear_local_tables()

    def __len__(self):
        return len(self.keys())

    def _clear_local_tables(self):
        """Clear local table data."""
        # Collection of keys in this file.
        self._keys = list()
        # Look-up table of labels corresponding integers.
        self._label_enum = dict()
        # Index-label_enum pairs.
        self._index_table = None

    def _clear_persistent_tables(self):
        """Clear all table data."""
        self._clear_local_tables()
        for k in [ReservedKeys.KEY_MANIFEST,
                  ReservedKeys.LABEL_ENUM,
                  ReservedKeys.INDEX_TABLE]:
            if k in self:
                del self[k]

    def get(self, key):
        """Retrieve the datapoint for the given key; fails if not found."""
        assert key in self, "Does not contain an item at '%s'." % key
        return core.Sample.from_file(h5py.File.get(self, name=key))

    def add(self, key, data, overwrite=False):
        """Add data sample under the given key.

        Parameters
        ----------
        key : string
            String under which to write the data to file.
        data : Any type that conforms to the interface of a Dataset.
            Initialized data to write; note that the data will be renamed
            with the given key.
        overwrite : bool, default=False
            Overwrite any existing data under the given key, if it exists.
        """
        key = keyutils.cleanse(key)
        assert keyutils.is_keylike(key), "Improperly formatted key: '%s'" % key
        # Persistent data will be inconsistent; delete everything.
        self._clear_persistent_tables()

        # Create the object on-disk.
        data.name = key
        dataset = self.create_dataset(name=data.name,
                                      data=data.value)
        attrs = utils.partition_attrs(data.attrs)
        # Copy the attrs dictionary of the datapoint.
        for k, v in attrs.iteritems():
            # Be sure to write any numerical representations that may evaluate
            # to false, i.e. zero. Separate statements to handle np.ndarrays.
            if isinstance(v, (int, float, np.ndarray)):
                dataset.attrs[k] = v
            elif v:
                dataset.attrs[k] = v

    def remove(self, key):
        """Remove the key and corresponding datapoint from the filesystem.

        Parameters
        ----------
        key : string, key-like
            DataPoint to drop. Must exist, or will fail loudly.
        """
        assert key in self, "Key does not exists in filesystem."
        del self[key]
        self._clear_persistent_tables()

    def keys(self):
        """All keys corresponding to datapoints in this file.

        Note: The returned list will contain no ReservedKeys.
        """
        if not ReservedKeys.KEY_MANIFEST in self:
            self.create_tables(write=True)

        assert ReservedKeys.KEY_MANIFEST in self, \
            "Could not find a persistent key manifest!"
        if not self._keys:
            self._keys = list(self[ReservedKeys.KEY_MANIFEST].value)

        return list(self._keys)

    def label_enum(self):
        """Unique labels and enumeration values."""
        if not ReservedKeys.LABEL_ENUM in self:
            self.create_tables(write=True)

        assert ReservedKeys.LABEL_ENUM in self, \
            "Could not find a persistent label enumeration map!"
        if not self._label_enum:
            for k, v in dict(self[ReservedKeys.LABEL_ENUM].value).iteritems():
                self._label_enum[k] = int(v)

        return dict(self._label_enum)

    def index_table(self):
        """Integer keys and label enumeration values.

        Returns
        -------
        index_table : np.ndarray
            Integer keys and label enumeration values.
        """
        if not ReservedKeys.INDEX_TABLE in self:
            self.create_tables(write=True)

        assert ReservedKeys.INDEX_TABLE in self, \
            "Could not find a persistent index table!"
        if self._index_table is None:
            self._index_table = self[ReservedKeys.INDEX_TABLE].value

        return self._index_table

    def create_tables(self, write=True):
        """Iterate over all items, find keyed paths (conforming to is_keylike),
        write the indexing tables to file, and cache them locally.

        write : bool
            Write the indexing tables to file.
        """
        self._clear_persistent_tables()
        index_list = []
        # self._key_list = []

        def cache_data(key, obj):
            """Callback function for h5py's 'visititems' method."""
            if isinstance(obj, h5py.Dataset) and keyutils.is_keylike(key):
                # Add new keys to the manifest.
                self._keys.append(keyutils.cleanse(key))
                # Enumerate labels on the fly as they're visited.
                for label in core.Dataset(obj).labels.values():
                    if not label in self._label_enum:
                        self._label_enum[label] = len(self._label_enum)
                    # Populate index-enum tuples.
                    index_list.append((keyutils.key_to_index(key),
                                       self._label_enum[label]))

        self.visititems(cache_data)
        self.create_dataset(name=ReservedKeys.INDEX_TABLE,
                            data=np.asarray(index_list, dtype=int))
        self.create_dataset(name=ReservedKeys.KEY_MANIFEST,
                            data=self._keys)
        label_enum = [(k, v) for k, v in self._label_enum.iteritems()]
        self.create_dataset(name=ReservedKeys.LABEL_ENUM, data=label_enum)


class SequenceFile(SampleFile):
    """Write me.

    Sequences and whatnot ... time-aligned, blah blah.
    """

    def get(self, key):
        """Retrieve the datapoint for the given key; fails if not found."""
        assert key in self, "Does not contain an item at '%s'." % key
        return core.Sequence.from_file(h5py.File.get(self, name=key))

    def create_tables(self, write=True):
        """Iterate over all items, find keyed paths (conforming to is_keylike),
        write the indexing tables to file, and cache them locally.

        write : bool
            Write the indexing tables to file.
        """
        self._clear_persistent_tables()
        index_list = []
        # self._key_list = []

        def cache_data(key, obj):
            """Callback function for h5py's 'visititems' method."""
            if isinstance(obj, h5py.Dataset) and keyutils.is_keylike(key):
                # Add new keys to the manifest.
                self._keys.append(keyutils.cleanse(key))
                # Enumerate labels on the fly as they're visited.
                for label_seq in core.Dataset(obj).labels.values():
                    for subindex, label in enumerate(label_seq):
                        if not label in self._label_enum:
                            self._label_enum[label] = len(self._label_enum)
                        # Populate index-subindex-enum tuples.
                        index_list.append((keyutils.key_to_index(key),
                                           subindex,
                                           self._label_enum[label]))

        self.visititems(cache_data)
        self.create_dataset(name=ReservedKeys.INDEX_TABLE,
                            data=np.asarray(index_list, dtype=int))
        self.create_dataset(name=ReservedKeys.KEY_MANIFEST,
                            data=self._keys)
        label_enum = [(k, v) for k, v in self._label_enum.iteritems()]
        self.create_dataset(name=ReservedKeys.LABEL_ENUM, data=label_enum)


class Cache(dict):
    """Provides an in-memory data interface.

    Maintains a consistent interface with a Sample/SequenceFiles.
    """

    def __init__(self, source, cache_size=1000, refresh_prob=0.25):
        """
        Parameters
        ----------
        source : Any object with a 'next()' method.
            A data source with which to populate the deck.
        refresh_prob : float, in [0, 1]
            Probability a cached item may be dropped and replaced.
        cache_size : int
            Number of items to maintain in the cache.
        """
        self._refresh_prob = refresh_prob
        self.source = source
        # If the dataset can fit in the cache, do it, and disable replacement.
        if self.source.num_items < cache_size:
            cache_size = self.source.num_items
            refresh_prob = 0

        self.load(num_items=cache_size)

    def _clear_local_tables(self):
        """Clear local table data."""
        # Collection of keys in this file.
        self._keys = list()
        # Look-up table of labels corresponding integers.
        self._label_enum = dict()
        # Index-label_enum pairs.
        self._index_table = None

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
            k, v = self.source.next()
            self[k] = v
            num_items -= 1
        # self.create_tables()

    def remove(self, key):
        del self[key]
        # Clean up indexes; or, at least delete tables.
        self._clear_local_tables()


class Batch(OrderedDict):
    """
    """
    def __init__(self, source, batch_size):
        OrderedDict.__init__(self)
        self.source = source
        self.batch_size = batch_size
        self.next()

    def next(self):
        self.clear()
        self.load(num_items=self.batch_size)

    def load(self, num_items=1):
        """Load the next 'num_items'."""
        while num_items > 0:
            k, v = self.source.next()
            self[k] = v
            num_items -= 1

    def values(self):
        return [self[k].value for k in self]

    def labels(self, name):
        return [self[k].labels[name] for k in self]
