"""Core data types and classes."""

import numpy as np

from . import ReservedKeys
from . import Config
from . import keyutils
from . import utils


class Dataset(object):
    """Shufflr wrapper around h5py datasets.

    This object exists to provide a unified interface to h5py datasets, while
    lazily loading information as necessary, i.e. only parsing a value as a
    np.ndarray when requested. The efficiency gains from this are noticeable.
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._name = dataset.name
        self._value = None
        self._labels = dict()
        self._targets = dict()
        self.__parse_attrs__(dataset.attrs)

    def __str__(self):
        """Render the datapoint as a string."""
        msg = []
        for n in ["name", "value", "labels", "targets", "metadata"]:
            value = eval("self.%s" % n)
            msg += ["'%s' : %s " % (n, value)]
        return "{ %s}" % "".join(msg)

    def __parse_attrs__(self, attrs):
        """Consume an h5py AttributeManager and store data locally.
        """
        self._metadata = utils.decode_attrs(dict(attrs))

        label_keys = [k for k in self.metadata.iterkeys()
                      if k.startswith(ReservedKeys.LABEL_PREFIX)]
        self._labels = dict([(k.strip(ReservedKeys.LABEL_PREFIX),
                              self.metadata.pop(k)) for k in label_keys])

        target_keys = [k for k in self.metadata.iterkeys()
                       if k.startswith(ReservedKeys.TARGET_PREFIX)]
        self._targets = dict([(k.strip(ReservedKeys.TARGET_PREFIX),
                               self.metadata.pop(k)) for k in target_keys])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if keyutils.is_keylike(name):
            name = keyutils.cleanse(name)
        self._name = name

    @property
    def value(self):
        return self._dataset.value

    @property
    def labels(self):
        return self._labels

    @property
    def targets(self):
        return self._targets

    @property
    def metadata(self):
        return self._metadata


class Sample(Dataset):
    """Data structure for a single, instantaneous data point."""

    def __init__(self, value, name=None, labels=None, targets=None,
                 metadata=None, dtype=np.float32):
        """Create a data Sample.

        value : np.ndarray [Required]
            Data payload.
        name : str
            Identifier for the data; may be anonymous. Any leading '/'s
            are stripped from the string.
        labels : list or dict
            Semantic descriptors of the data, i.e. tags. If the input
            is a list, the labels will be assigned keys; otherwise, dict inputs
            will preserve the labels' given keys.
        targets : list or dict
            Numeric representation for the data, i.e. regression values. If the
            input is a list, the targets will be assigned keys; otherwise, dict
            inputs will preserve the targets' given keys.
        metadata : dict, default={}
            Miscellaneous information to associate with the data.
        dtype : data_type
            Default is float32.
        """
        self._file_data = None
        self._dtype = dtype
        self._value = np.asarray(value, dtype=dtype)

        if isinstance(labels, list):
            labels = dict([("%d" % n, v) for n, v in enumerate(labels)])
        elif not labels:
            labels = dict()
        self._labels = labels

        if isinstance(targets, list):
            targets = dict([("%d" % n, v) for n, v in enumerate(targets)])
        elif not targets:
            targets = dict()
        self._targets = targets

        if not metadata:
            metadata = dict()
        self._metadata = metadata.copy()
        self._name = name

    @classmethod
    def from_file(self, dataset, dtype=np.float32):
        """Create a Sample from an h5py dataset.

        This implementation serves as an alternate constructor, since typed
        overloading doesn't totally jive with Python.
        """
        dset = Dataset(dataset)
        return Sample(name=dset.name,
                      value=dset.value,
                      labels=dset.labels,
                      targets=dset.targets,
                      metadata=dset.metadata,
                      dtype=dtype)

    @property
    def attrs(self):
        """Bundle necessary information as an attribute dictionary, i.e.
        everything that is not 'value'.

        This is conceptually analogous to an h5py.Dataset's attrs variable.
        """
        attrs = self.metadata.copy()
        for k, v in self.targets.iteritems():
            assert np.asarray(v).nbytes < Config.MAX_ATTRS_SIZE, \
                "Warning: Target value must be smaller than 64kB."
            attrs[ReservedKeys.TARGET_PREFIX + k] = v

        for k, v in self.labels.iteritems():
            assert np.asarray(v).nbytes < Config.MAX_ATTRS_SIZE, \
                "Warning: Label value must be smaller than 64kB."
            attrs[ReservedKeys.LABEL_PREFIX + k] = v

        return attrs

    @property
    def value(self):
        return self._value


class Sequence(Sample):
    """A Sequence is an array ordered along the first dimension.
    """

    def __init__(self, value, name=None, labels=None, targets=None,
                 metadata=None, dtype=np.float32):
        """Create a Sequence.

        value : np.ndarray [Required]
            Data payload.
        name : str
            Identifier for the datapoint; may be anonymous. Any leading '/'s
            are stripped from the string.
        labels : list or dict of sequences, default=None
            Semantic descriptor of the data, i.e. class, tag, etc. For use in
            classification problems; Each label sequence must be the same
            length (first dimension) as the provided value.
        targets : list or dict of sequences, default=None
            A desired numerical representation for the data, i.e. embedding,
            bit vector, etc. For use in regression problems; note that defining
            as the string 'self' returns value, useful for reconstruction
            training.
            Note: It is a known issue that targets are limited to 64kB of data.
        metadata : dict, default={}
            Miscellaneous information to associate with the data.
        dtype : data_type
            Default is float32.
        """
        # Need to de-serialize labels / targets here.
        Sample.__init__(self, value, name, labels, targets, metadata, dtype)

    @classmethod
    def from_file(self, dataset, dtype=None):
        """Create a DataPoint from an h5py Dataset.

        This implementation serves as an alternate constructor, since typed
        overloading doesn't totally jive with Python.
        """
        data = Dataset(dataset)
        return Sequence(name=data.name,
                        value=data.value,
                        labels=data.labels,
                        targets=data.targets,
                        metadata=data.metadata)

    def __len__(self):
        return len(self.value)

    @property
    def attrs(self):
        """Bundle necessary information as an attribute dictionary, i.e.
        everything that is not 'value', i.e. non-value data serialization.

        This is functionally analogous to h5py.Dataset's attrs.
        """
        attrs = self.metadata.copy()
        for k, v in self.targets.iteritems():
            attrs[ReservedKeys.TARGET_PREFIX + k] = v

        for k, v in self.labels.iteritems():
            attrs[ReservedKeys.LABEL_PREFIX + k] = v

        return attrs

'''
Not sure any of this should happen here; sub-Sequence should probably be a
filter. Consume a sequence and parameters, return another (smaller) Sequence.

    def subsequence(self, index, left, right,
                    label_fill_values=None, target_fill_values=None):
        """Return a sub-Sequence.
        """
        assert left >= 0, "Left context cannot be negative."
        assert right >= 0, "Right context cannot be negative."
        value = utils.context_slice(
            self.value, index, left, right, fill_value=0.0)
        labels = dict([(k, )])
        return Sequence(name=self.name,
                        value=value,
                        labels=data.labels,
                        targets=data.targets,
                        metadata=data.metadata)


    def slice_value(self, index, left=0, right=0):
        """Return a slice, or all, of the data sequence.

        Parameters
        ----------
        index : int, or None
            Index of the data to slice. If None, this behaves like a data point
            and the full value is returned.
        left : int
            Number of datapoints to prepend to the result; will zero-pad when
            out-of-range.
        right : int
            Number of datapoints to append to the result; will zero-pad when
            out-of-range.

        Returns
        -------
        observation : np.ndarray
            Sliced or full array; ndims is always equal.
        """
        assert left >= 0, "Left context cannot be negative."
        assert right >= 0, "Right context cannot be negative."
        return utils.context_slice(
            self.value, index, left, right, fill_value=0.0)

    def slice_label(self, name, index, left=0, right=0, fill_value=None):
        """Return a slice of a label sequence.

        Parameters
        ----------
        name : str
            Name of the label to slice.
        index : int
            Index to center theslice.
        left : int
            Number of points to prepend to the result; will zero-pad when
            out-of-range.
        right : int
            Number of points to append to the result; will zero-pad when
            out-of-range.

        Returns
        -------
        label : array_like
            Sliced label sequence.
        """
        assert left >= 0, "Left range cannot be negative."
        assert right >= 0, "Right context cannot be negative."
        assert name in self.labels, \
            "Sequence does not contain a label named '%s'." % name
        full_label = self.labels[name]
        return utils.context_slice(
            full_label, index, left, right, fill_value=fill_value)

    def slice_target(self, name, index, left=0, right=0, fill_value=None):
        """Return a slice of a target sequence.

        Parameters
        ----------
        name : str
            Name of the label to slice.
        index : int
            Index to center theslice.
        left : int
            Number of points to prepend to the result; will zero-pad when
            out-of-range.
        right : int
            Number of points to append to the result; will zero-pad when
            out-of-range.

        Returns
        -------
        target : array_like
            Sliced target sequence.
        """
        assert left >= 0, "Left range cannot be negative."
        assert right >= 0, "Right context cannot be negative."
        assert name in self.targets, \
            "Sequence does not contain a label named '%s'." % name
        full_target = self.target[name]
        return utils.context_slice(
            full_target, index, left, right, fill_value=fill_value)
'''

'''
Not sure this goes here either.

class Batch(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._values = list()
        self._labels = list()

    def add_value(self, x):
        self._values.append(x)

    def add_label(self, y):
        self._labels.append(y)

    @property
    def values(self):
        return np.asarray(self._values)

    @property
    def labels(self):
        return np.asarray(self._labels)
'''