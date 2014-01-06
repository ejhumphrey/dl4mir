"""
"""

import os
import numpy as np

from . import Config
from . import ReservedKeys


def safe_remove(filename):
    """Delete a file iff it exists."""
    if os.path.exists(filename):
        os.remove(filename)


def max_depth(num_items):
    """Return the number of levels given the max capacity."""
    return int(np.ceil(np.log(num_items) / np.log(Config.ITEMS_PER_NODE)))


def is_sequence(obj):
    """Determine if an object is a sequence.

    Specifically, 'sequences' are only lists or arrays.
    """
    if isinstance(obj, np.ndarray):
        return True
    else:
        return isinstance(obj, list)


def context_slice(value, index, left, right, fill_value):
    """Slice a sequence with context, padding values out of range.

    Parameters
    ----------
    value : np.ndarray
        Multidimensianal array to slice.
    index : int
        Position along the first axis to center the slice.
    left : int
        Number of previous points to return.
    right : int
        Number of subsequent points to return.
    fill_value : scalar
        Value to use for out-of-range regions.

    Returns
    -------
    region : np.ndarray
        Slice of length left + right + 1; all other dims are equal to the
        input.
    """
    idx_left = max([index - left, 0])
    idx_right = min([index + right + 1, len(value)])
    observation = value[idx_left:idx_right]
    if isinstance(value, np.ndarray):
        other_dims = list(value.shape[1:])
        result = np.empty([left + right + 1] + other_dims,
                          dtype=value.dtype)
        result[:] = fill_value
    else:
        result = [fill_value] * (left + right + 1)
    idx_out = idx_left - (index - left)
    result[idx_out:idx_out + len(observation)] = observation
    return result


def partition_attrs(attrs):
    """Modify in-place.
    """
    attrs = attrs.copy()
    keys = attrs.keys()
    for k in keys:
        attr_size = np.asarray(attrs[k]).nbytes
        if attr_size >= Config.MAX_ATTRS_SIZE:
            # Break up the sequence into mulitiple chunks.
            v = attrs.pop(k)
            num_splits = attr_size / float(Config.SAFE_ATTRS_SIZE)
            split_len = int(len(v) / num_splits)
            start_idx, count = 0, 0
            v_sub = v[start_idx:start_idx + split_len]
            while v_sub:
                attrs[ReservedKeys.PARTITION_FMT % (k, count)] = v_sub
                count += 1
                start_idx += split_len
                v_sub = v[start_idx:start_idx + split_len]
    return attrs


def decode_attrs(attrs):
    """Modify in-place.
    """
    attrs = attrs.copy()
    # First, collect keys to concatenate.
    grouped_keys = {}
    for key in attrs:
        if ReservedKeys.PARTITION_EXT in key:
            base, part_idx = key.split(ReservedKeys.PARTITION_EXT)
            if not base in grouped_keys:
                grouped_keys[base] = []
            grouped_keys[base].append(int(part_idx))

    # Second, concatenate parts.
    for key in grouped_keys:
        assert not key in attrs, "Key conflict! Already contains %s." % key
        attrs[key] = []
        grouped_keys[key].sort()
        for idx in grouped_keys[key]:
            part_key = ReservedKeys.PARTITION_FMT % (key, idx)
            attrs[key].append(attrs.pop(part_key))
        attrs[key] = np.concatenate(attrs[key])
    return attrs
