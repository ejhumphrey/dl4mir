"""Utilities for managing hex-keys, i.e. string representations of hexadecimal
numbers, for efficiently indexing and organizing HDF5 files.

Key design is a necessary, and sometimes non-trivial, component of using an
HDF5 based filesystem. In the interest of evenly distributing data
hierarchically (in a tree), we can create hexadecimal keys that can be used as
slash-separated strings, or efficiently converted to integers for manipulating
in numpy arrays.

Therefore, there are conceptually two ways to refer to data in an HDF5 file:

  keys : Forward-slash separated strings of hexadecimal characters. Each key
    has both a width and depth parameter. In terms of a tree, width defines the
    number of leaves at a node, and the depth defines the number of levels in
    the tree. Note that defining a fixed width, the number of end-nodes in the
    tree increases exponentially with depth. Empirically, width should seldom,
    if ever, exceed 2**8 = 256. Here, a depth of 3 provides a large key space.
    Such a key might look like "02/a5/f1"; note that every terminal node can be
    accessed directly in this manner.
  indexes : An index is a scalar representation of a key. Note that because a
    key is merely a string representation of a number, they can be converted
    interchangeably with minimal side information.

TODO(ejhumphrey): These could use some tests.
"""

from random import shuffle


def is_hex(val):
    """Test a string for hexadecimal-ness."""
    if val.startswith("0x"):
        val = val[2:]
    h = ["%d" % i for i in range(10)] + list('abcdef')
    for c in val.lower():
        if not c in h:
            return False
    return True


def cleanse(key):
    """Parse a key into a common, clean format."""
    return str(key).strip('/').lower()


def is_keylike(key):
    """Test for key-ness, defined by three criteria:
    - Is a string.
    - Equal characters per slash-split. ['00', '4f', '25']
    - Each portion passes is_hex().

    If each of these are True, the input 'key' can be considered as such.
    """
    key = cleanse(key)
    parts = key.split("/")
    equal_spacing = [len(p) == len(parts[0]) for p in parts]
    hex_parts = [is_hex(p) for p in parts]
    return all(equal_spacing) and all(hex_parts)


def expand_hex(hexval, width):
    """Zero-pad a hexadecimal representation out to a given number of places.

    Example:

    Parameters
    ----------
    hexval : str
        Hexadecimal representation, produced by hex().
    width : int
        Number of hexadecimal places to expand.

    Returns
    -------
    padded_hexval : str
        Zero-extended hexadecimal representation.

    Note: An error is raised if width is less than the number of hexadecimal
    digits required to represent the number.
    """
    chars = hexval[2:]
    assert width >= len(chars), \
        "Received: %s. Width (%d) must be >= %d." % (hexval, width, len(chars))
    y = list('0x' + '0' * width)
    y[-len(chars):] = list(chars)
    return "".join(y)


def index_to_key(index, depth):
    """Convert an integer to a hex-key representation.

    Example: index_to_key(843, 2) -> '03/4b'

    Parameters
    ----------
    index : int
        Integer index representation.
    depth : int
        Number of levels in the key (number of slashes plus one).

    Returns
    -------
    key : str
        Slash-separated hex-key.
    """
    hx = expand_hex(hex(int(index)), depth * 2)
    tmp = ''.join(
        [''.join(d) for d in zip(hx[::2], hx[1::2], '/' * (len(hx) / 2))])
    return tmp[3:-1]


def key_to_index(key):
    """Convert a hex-key representation to an integer.

    Example: key_to_index('03/4b') -> 843

    Parameters
    ----------
    key : str
        A hexadecimal key.

    Returns
    -------
    index : int
        Integer representation of the hexkey.
    """
    assert is_keylike(key), "The provided key '%s' is not key-like." % key
    key = cleanse(key)
    return int("0x" + "".join(key.split('/')), base=0)


def uniform_keygen(depth, width=256):
    """Generator to produce uniformly distributed keys at a given depth.

    Deterministic and consistent, equivalent to a strided xrange() that yields
    strings like '04/1b/22' for depth=3, width=256.

    Parameters
    ----------
    depth : int
        Number of nodes in a single branch. See docstring in keyutil.py for
        more information.
    width : int
        Child nodes per parent. See docstring in keyutil.py for more
        information.

    Returns
    -------
    key : str
        Hexadecimal key path.
    """
    max_index = width ** depth
    index = 0
    for index in xrange(max_index):
        v = expand_hex(hex(index), depth * 2)
        hexval = "0x" + "".join([a + b for a, b in zip(v[-2:1:-2], v[:1:-2])])
        yield index_to_key(int(hexval, 16), depth)
    raise ValueError("Unique keys exhausted.")

'''
TODO(ejhumphrey): Move to selector or delete entirely.

class KeySelector(list):
    """Random item selector, infinite generator.

    Inherits from list, and can therefore be initialized with an iterable
    collection. Alternatively, items can be added via append().

    Example:
    >>> ks = KeySelector('abcdefghijkf')
    >>> print [ks.next() for n in range(3)]
    ['c', 'g', 'h']

    Subclasses should also implement __iter__() to achieve more interesting
    behavior as to how this class can generate a sequence of its items.
    """
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.__reset__()

    def __reset__(self):
        shuffle(self)
        self.__index__ = 0

    def __iter__(self):
        while True:
            yield self.next()

    def __add__(self, *args, **kwargs):
        res = list.__add__(self, *args, **kwargs)
        self.__reset__()
        return res

    def next(self):
        next_key = self[self.__index__]
        self.__index__ += 1
        if self.__index__ >= len(self):
            self.__reset__()
        return next_key
'''
