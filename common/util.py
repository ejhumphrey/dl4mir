import numpy as np


def inarray(ar1, ar2):
    """Test whether each element of an array is present in a second array.

    Returns a boolean array the same shape as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.

    Returns
    -------
    out : ndarray, bool
        The values of `ar1` that are in `ar2`.
    """
    ar1 = np.asarray(ar1)
    out = np.zeros(ar1.shape, dtype=bool)
    for val in np.asarray(ar2).flatten():
        out |= np.equal(ar1, val)
    return out


def partition(obj, mapper, *args, **kwargs):
    """Label the partitions of `obj` based on the function `mapper`.

    Parameters
    ----------
    obj : dict_like
        Data collection to partition.
    mapper : function
        A partition labeling function.
    *args, **kwargs
          Additional positional arguments or keyword arguments to pass
          through to ``generator()``

    Returns
    -------
    labels : dict
        Partition labels, under the same keys in `obj`.
    """
    return dict([(key, mapper(obj.get(key), *args, **kwargs))
                 for key in obj.keys()])


def index_partition_arrays(partition_labels, label_set):
    """Index a dict of partition label arrays filtered by a set of labels.

    Parameters
    ----------
    partition_labels : dict_like
        A labeled partition object.
    label_set: list
        Set of labels for restricting the given partition.

    Returns
    -------
    subset_index : dict
        The indexes into `partition_labels` that match `label_set`, under the
        same keys.
    """
    index = dict()
    for key in partition_labels.keys():
        partition_array = partition_labels.get(key)
        in_array = inarray(partition_array, label_set)
        if in_array.sum():
            index[key] = np.arange(len(in_array), dtype=int)[in_array]
    return index


def boundary_pool(x_in, index_edges, pool_func='mean'):
    """Pool the values of an array, bounded by a set of edges.

    Parameters
    ----------
    x_in : np.ndarray, shape=(n_points, ...)
        Array to pool.
    index_edges : array_like, shape=(n_edges,)
        Boundary indices for pooling the array.
    pool_func : str
        Name of pooling function to use; one of {`mean`, `median`, `max`}.

    Returns
    -------
    z_out : np.ndarray, shape=(n_edges-1, ...)
        Pooled output array.
    """
    fxs = dict(mean=np.mean, max=np.max, median=np.median)
    assert pool_func in fxs, \
        "Function '%s' unsupported. Expected one of {%s}" % (pool_func,
                                                             fxs.keys())
    pool = fxs[pool_func]
    num_points = len(index_edges) - 1
    z_out = np.zeros([num_points, x_in.shape[1]])
    for idx, delta in enumerate(np.diff(index_edges)):
        if delta > 0:
            z = pool(x_in[index_edges[idx]:index_edges[idx + 1]], axis=0)
        elif delta == 0:
            z = x_in[index_edges[idx]]
        else:
            raise ValueError("`index_edges` must be monotonically increasing.")
        z_out[idx] = z
    return z_out
