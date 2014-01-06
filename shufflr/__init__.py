"""Shufflr: A data shuffling / presentation library for online learning.

Shufflr is a library for randomly accessing datapoints in arbitrarily large
datasets for training online learning algorithms.

There are two kinds of supported Sources:
 - Points
 - Sequences

"""


class Config(object):
    """Module-wide configuration parameters."""
    # MB limit to keep in memory.
    MEMORY_LIMIT = 8192

    INDEX_SIZE = 3
    INDEX_INTKEY = 0
    INDEX_POS = 1
    INDEX_LABEL = 2

    MAX_DEPTH = 3
    ITEMS_PER_NODE = 256
    MAX_ATTRS_SIZE = (2 ** 16) - 16
    SAFE_ATTRS_SIZE = MAX_ATTRS_SIZE * 0.8


class ReservedKeys(object):
    """Data structure for reserved keys in the HDF5 filesystem."""
    # Like in SQL-esque databases, we build an index table that keeps track of
    # what data lives where in the source for efficient access. Currently, this
    # table has two integer columns:
    #
    #   [ index, label_enum ]
    #
    # index : A numeric representation of the full HDF5 path to a dataset. To
    #    access the corresponding item, it must be parsed into a hexadecimal
    #    string representation, with every two characters separated by a '/'.
    # label_enum : An enumeration value corresponding to a semantically
    #    meaningful label in LABEL_ENUM, which is populated as data is added to
    #    the source.
    # - Future? -
    # position : Relative position in the dataset corresponding to the label.
    # If the label applies to the entire dataset, position < 0.
    INDEX_TABLE = 'index_table'

    # As data is added to the object, a key-value store is created mapping
    # semantically meaningful (i.e. makes sense to a human) labels to an
    # integer representation.
    LABEL_ENUM = "label_enum"

    #
    KEY_MANIFEST = 'key_manifest'

    DEPTH = 'depth'
    MAX_ITEMS = 'max_item_count'

    # 'attrs' dictionary keys
    LABEL = 'label'
    LABEL_PREFIX = "label:"
    METADATA = 'metadata'
    TARGET = 'target'
    TARGET_PREFIX = "target:"
    DATA = 'data'
    PARTITION_EXT = ".part-"
    PARTITION_FMT = "%%s%s%%d" % PARTITION_EXT
    TYPE = "datatype"