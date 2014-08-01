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


def normalize(x, axis=None):
    """Normalize the values of an ndarray to sum to 1 along the given axis.

    Parameters
    ----------
    x : np.ndarray
        Input multidimensional array to normalize.
    axis : int, default=None
        Axis to normalize along, otherwise performed over the full array.

    Returns
    -------
    z : np.ndarray, shape=x.shape
        Normalized array.
    scalar : float
        Scale factor to normalize the input.
    """
    if not axis is None:
        shape = list(x.shape)
        shape[axis] = 1
        scalar = x.astype(float).sum(axis=axis).reshape(shape)
        scalar[scalar == 0] = 1.0
    else:
        scalar = x.sum()
        scalar = 1 if scalar == 0 else scalar
    return x / scalar, scalar


def viterbi(posterior, transition_matrix, prior=None, penalty=0, scaled=True):
    """Find the optimal Viterbi path through a posteriorgram.

    Ported closely from Tae Min Cho's MATLAB implementation.

    Parameters
    ----------
    posterior: np.ndarray, shape=(num_obs, num_states)
        Matrix of observations (events, time steps, etc) by the number of
        states (classes, categories, etc).
    transition_matrix: np.ndarray
        Transition matrix for the viterbi algorithm.
    prior: np.ndarray, default=None (uniform)
        Probability distribution over the states.
    penalty: scalar, default=0
        Scalar penalty to down-weight off-diagonal states.
    scaled : bool, default=True
        Scale transition probabilities between steps in the algorithm.
        Note: Hard-coded to True in TMC's implementation; it's probably a bad
        idea to change this.

    Returns
    -------
    path: np.ndarray, shape=(num_obs,)
        Optimal state indices through the posterior.
    """
    def log(x):
        """Logarithm with built-in epsilon offset."""
        return np.log(x + np.power(2.0, -10.0))

    num_obs, num_states = posterior.shape

    # Normalize the posterior.
    posterior = normalize(posterior, axis=1)[0]

    # Apply the off-axis penalty.
    offset = np.ones([num_states]*2, dtype=float)
    offset -= np.eye(num_states, dtype=np.float)
    penalty = offset * np.exp(penalty) + np.eye(num_states, dtype=np.float)
    transition_matrix = penalty * transition_matrix

    # Create a uniform prior if one isn't provided.
    prior = np.ones(num_states) / float(num_states) if prior is None else prior

    # Algorithm initialization
    delta = np.zeros_like(posterior)
    psi = np.zeros_like(posterior)
    path = np.zeros(num_obs, dtype=int)

    idx = 0
    delta[idx, :] = prior * posterior[idx, :]
    if scaled:
        delta[idx, :] = normalize(delta[idx, :])[0]

    for idx in range(1, num_obs):
        for state in range(num_states):
            res = delta[idx - 1, :] * transition_matrix[state, :]
            delta[idx, state], psi[idx, state] = np.max(res), np.argmax(res)
            delta[idx, state] *= posterior[idx, state]
        if scaled:
            delta[idx, :] = normalize(delta[idx, :])[0]

    path[-1] = np.argmax(delta[-1, :])
    for idx in range(num_obs - 2, -1, -1):
        path[idx] = psi[idx + 1, path[idx + 1]]
    return path
