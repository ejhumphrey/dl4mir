from __future__ import print_function
import biggie
from itertools import groupby
import json
import numpy as np
import optimus
import os
import pyjams
import scipy.stats
import shutil
from sklearn.cross_validation import KFold
import time


def hwr(x):
    return x * (x > 0.0)


def mode(*args, **kwargs):
    return scipy.stats.mode(*args, **kwargs)[0]


def mode2(x_in, axis):
    value_to_idx = dict()
    idx_to_value = dict()
    for x in x_in:
        obj = buffer(x)
        if obj not in value_to_idx:
            idx = len(value_to_idx)
            value_to_idx[obj] = idx
            idx_to_value[idx] = x
    counts = np.bincount([value_to_idx[buffer(x)] for x in x_in])
    return idx_to_value[counts.argmax()]


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
    unique1 = np.unique(ar1)
    unique2 = np.unique(ar2)
    if len(unique1) > len(unique2):
        for val in unique2:
            out |= np.equal(ar1, val)
    else:
        for val in unique1:
            out |= np.equal(ar1, val) * (val in unique2)
    return out


def partition(obj, mapper, *args, **kwargs):
    """Label the partitions of `obj` based on the function `mapper`.

    Parameters
    ----------
    obj : dict_like
        Data collection to partition.
    mapper : function
        A partition labeling function; consumes entities, returns integers.
    *args, **kwargs
          Additional positional arguments or keyword arguments to pass
          through to ``mapper()``

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


def boundary_pool(x_in, index_edges, axis=0, pool_func='mean'):
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
    fxs = dict(mean=np.mean, max=np.max, median=np.median, mode=mode2)
    assert pool_func in fxs, \
        "Function '%s' unsupported. Expected one of {%s}" % (pool_func,
                                                             fxs.keys())
    pool = fxs[pool_func]
    num_points = len(index_edges) - 1
    axes_order = range(x_in.ndim)
    axes_order.insert(0, axes_order.pop(axis))
    axes_reorder = np.array(axes_order).argsort()
    x_in = x_in.transpose(axes_order)

    z_out = np.empty([num_points] + list(x_in.shape[1:]), dtype=x_in.dtype)
    for idx, delta in enumerate(np.diff(index_edges)):
        if delta > 0:
            z = pool(x_in[index_edges[idx]:index_edges[idx + 1]], axis=0)
        elif delta == 0:
            z = x_in[index_edges[idx]]
        else:
            raise ValueError("`index_edges` must be monotonically increasing.")
        z_out[idx, ...] = z
    return z_out.transpose(axes_reorder)


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
    """
    if axis is not None:
        shape = list(x.shape)
        shape[axis] = 1
        scalar = x.astype(float).sum(axis=axis).reshape(shape)
        scalar[scalar == 0] = 1.0
    else:
        scalar = x.sum()
        scalar = 1 if scalar == 0 else scalar
    return x / scalar


def lp_scale(x, p=2.0, axis=None):
    """Scale the values of `x` by the lp-norm of each vector.

    Parameters
    ----------
    x : np.ndarray
        Input multidimensional array to normalize.
    p : scalar
        Lp space for scaling.
    axis : int, default=None
        Axis to normalize along, otherwise performed over the full array.

    Returns
    -------
    z : np.ndarray, shape=x.shape
        Normalized array.
    """
    if axis is not None:
        shape = list(x.shape)
        shape[axis] = 1
        scalar = np.power(np.power(np.abs(x.astype(float)), p).sum(axis=axis),
                          1.0/p).reshape(shape)
        scalar[scalar == 0] = 1.0
    else:
        scalar = np.power(np.power(np.abs(x.astype(float)), p).sum(), 1.0/p)
        scalar = 1 if scalar == 0 else scalar
    return x / scalar


def viterbi(posterior, transition_matrix=None, prior=None, penalty=0,
            scaled=True):
    """Find the optimal Viterbi path through a posteriorgram.

    Ported closely from Tae Min Cho's MATLAB implementation.

    Parameters
    ----------
    posterior: np.ndarray, shape=(num_obs, num_states)
        Matrix of observations (events, time steps, etc) by the number of
        states (classes, categories, etc), e.g.
          posterior[t, i] = Pr(y(t) | Q(t) = i)
    transition_matrix: np.ndarray, shape=(num_states, num_states)
        Transition matrix for the viterbi algorithm. For clarity, each row
        corresponds to the probability of transitioning to the next state, e.g.
          transition_matrix[i, j] = Pr(Q(t + 1) = j | Q(t) = i)
    prior: np.ndarray, default=None (uniform)
        Probability distribution over the states, e.g.
          prior[i] = Pr(Q(0) = i)
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

    # Infer dimensions.
    num_obs, num_states = posterior.shape

    # Define the scaling function
    scaler = normalize if scaled else lambda x: x
    # Normalize the posterior.
    # posterior = normalize(posterior, axis=1)

    if transition_matrix is None:
        transition_matrix = np.ones([num_states]*2)

    # transition_matrix = normalize(transition_matrix, axis=1)

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
    delta[idx, :] = scaler(prior * posterior[idx, :])

    for idx in range(1, num_obs):
        res = delta[idx - 1, :].reshape(1, num_states) * transition_matrix
        delta[idx, :] = scaler(np.max(res, axis=1) * posterior[idx, :])
        psi[idx, :] = np.argmax(res, axis=1)

    path[-1] = np.argmax(delta[-1, :])
    for idx in range(num_obs - 2, -1, -1):
        path[idx] = psi[idx + 1, path[idx + 1]]
    return path


def fold_array(x_in, length, stride):
    """Fold a 2D-matrix into a 3D tensor by wrapping the last dimension."""
    num_tiles = int((x_in.shape[1] - (length-stride)) / float(stride))
    return np.array([x_in[:, n*stride:n*stride + length]
                     for n in range(num_tiles)])


def run_length_encode(seq):
    """Run-length encode a sequence of items.

    Parameters
    ----------
    seq : array_like
        Sequence to compress.

    Returns
    -------
    comp_seq : list
        Compressed sequence containing (item, count) tuples.
    """
    return [(obj, len(list(group))) for obj, group in groupby(seq)]


def run_length_decode(comp_seq):
    """Run-length decode a sequence of (item, count) tuples.

    Parameters
    ----------
    comp_seq : array_like
        Sequence of (item, count) pairs to decompress.

    Returns
    -------
    seq : list
        Expanded sequence.
    """
    seq = list()
    for obj, count in seq:
        seq.extend([obj]*count)
    return seq


def slice_tile(x_in, idx, length):
    """Extract a padded tile from a matrix, along the first dimension.

    Parameters
    ----------
    x_in : np.ndarray, ndim=2
        2D Matrix to slice.
    idx : int
        Centered index for the resulting tile.
    length : int
        Total length for the output tile.

    Returns
    -------
    z_out : np.ndarray, ndim=2
        The extracted tile.
    """
    start_idx = idx - length / 2
    end_idx = start_idx + length
    tile = np.zeros([length, x_in.shape[1]])
    x_in = np.concatenate([x_in, tile], axis=0)
    if start_idx < 0:
        tile[np.abs(start_idx):, :] = x_in[:end_idx, :]
    elif end_idx > x_in.shape[0]:
        end_idx = x_in.shape[0] - start_idx
        tile[:end_idx, :] = x_in[start_idx:, :]
    else:
        tile[:, :] = x_in[start_idx:end_idx, :]
    return tile


def stratify(items, num_folds, valid_ratio=0.1):
    """Stratify a collection of items `num_folds` times into partitions for
    train, validation, and test.

    Parameters
    ----------
    items : array_like
        Collection of unique items to stratify.
    num_folds : int
        Number of times to partition the data.
    valid_ratio : scalar, 0 < r < 1.0
        Ratio of the training set to carve off for validation.

    Returns
    -------
    folds : dict of dicts
        Sets of {'train', 'valid', 'test'} sets, indexed by fold.
    """
    items = np.asarray(items)
    splitter = KFold(n=len(items), n_folds=num_folds, shuffle=True)
    folds = dict()
    for fold_idx, data_idxs in enumerate(splitter):
        train_items, test_items = items[data_idxs[0]], items[data_idxs[1]]
        num_train = len(train_items)
        train_idx = np.random.permutation(num_train)
        valid_count = int(valid_ratio * num_train)
        valid_items = train_items[train_idx[:valid_count]]
        train_items = train_items[train_idx[valid_count:]]
        folds[fold_idx] = dict(train=train_items.tolist(),
                               valid=valid_items.tolist(),
                               test=test_items.tolist())
    return folds


def gibbs(energy, beta):
    """Normalize an energy vector as a Gibbs distribution."""
    axis = {1: None, 2: 1}[energy.ndim]
    y = np.exp(-beta * energy)
    scalar = y.sum(axis=axis)
    scalar = scalar.reshape(-1, 1) if axis == 1 else scalar
    return y / scalar


def categorical_sample(pdf):
    """Randomly select a categorical index of a given PDF."""
    pdf = pdf / pdf.sum()
    return int(np.random.multinomial(1, pdf).nonzero()[0])


def boundaries_to_durations(boundaries):
    """Return the durations in a monotonically-increasing set of boundaries.

    Parameters
    ----------
    boundaries : array_like, shape=(N,)
        Monotonically-increasing scalar boundaries.

    Returns
    -------
    durations : array_like, shape=(N-1,)
        Non-negative durations.
    """
    if boundaries != np.sort(boundaries).tolist():
        raise ValueError("Input `boundaries` is not monotonically increasing.")
    return np.abs(np.diff(boundaries))


def find_closest_idx(x, y):
    """Find the closest indexes in `x` to the values in `y`."""
    return np.array([np.abs(x - v).argmin() for v in y])


def filter_empty_values(obj):
    """Filter empty objects from a dictionary.

    TODO(ejhumphrey): deprecated, no? Isn't this what filter(None, iter) does?
    """
    result = dict()
    for k in obj:
        if obj[k]:
            result[k] = obj[k]
    return result


def copy_filedirs(src, dest):
    """Safely copy `src` to `dest`, making all necessary directories."""
    dest_dir = os.path.split(dest)[0]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.copyfile(src, dest)


def equals_value(arr, value):
    """Elementwise equals between a 1-d array of objects and a single value.

    Parameters
    ----------
    arr : array_like, shape=(n,)
        Set of values / objects to test for equivalence.
    value : obj
        Object to test against each element of the input iterable.

    Returns
    -------
    out : {ndarray, bool}
        Output array of bools.
    """
    return np.array([_ == value for _ in arr], dtype=bool)


def join_endata(enmfp_data, track_data):
    """Join the id-namespaces of different fetches of EchoNest data.

    Parameters
    ----------
    enmfp_data : dict
        Set of successful results from EchoNest Musical Fingerprint queries.
    track_data : dict
        Set of results from EchoNest track queries.

    Returns
    -------
    result : dict
        Merged set of tracks; preference is given to song IDs
    """
    result = dict()
    for key in track_data:
        if key in enmfp_data:
            data = enmfp_data[key].copy()
        else:
            data = track_data[key].copy()
        uid = data.pop('id')
        data['local_key'] = key
        if uid not in result:
            result[uid] = list()
        result[uid].append(data)
    return result


def intervals_to_durations(intervals):
    """Translate a set of intervals to an array of boundaries."""
    return np.abs(np.diff(np.asarray(intervals), axis=1)).flatten()


def slice_cqt_entity(entity, length, idx=None):
    """Return a windowed slice of a chord Entity.

    Parameters
    ----------
    entity : Entity, with at least {cqt, chord_labels} fields
        Observation to window.
        Note that entity.cqt is shaped (num_channels, num_frames, num_bins).
    length : int
        Length of the sliced array.
    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Returns
    -------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    cqt = np.array([slice_tile(x, idx, length) for x in entity.cqt])
    return biggie.Entity(cqt=cqt, chord_label=entity.chord_labels[idx])


def compress_samples_to_intervals(labels, time_points):
    """Compress a set of time-aligned labels via run-length encoding.

    Parameters
    ----------
    labels : array_like
        Set of labels of a given type.
    time_points : array_like
        Points in time corresponding to the given labels.

    Returns
    -------
    intervals : np.ndarray, shape=(N, 2)
        Start and end times, in seconds.
    labels : list, len=N
        String labels corresponding to the returned intervals.
    """
    assert len(labels) == len(time_points)
    intervals, new_labels = [], []
    idx = 0
    for label, count in run_length_encode(labels):
        start = time_points[idx]
        end = time_points[min([idx + count, len(labels) - 1])]
        idx += count
        intervals += [(start, end)]
        new_labels += [label]
    return np.array(intervals), new_labels


def load_jamset(filepath):
    """Load a collection of keyed JAMS (a JAMSet) into memory.

    Parameters
    ----------
    filepath : str
        Path to a JAMSet on disk.

    Returns
    -------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    jamset = dict()
    with open(filepath) as fp:
        for k, v in json.load(fp).iteritems():
            jamset[k] = pyjams.JAMS(**v)

    return jamset


def save_jamset(jamset, filepath):
    """Save a collection of keyed JAMS (a JAMSet) to disk.

    Parameters
    ----------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    output_data = dict()
    with pyjams.JSONSupport():
        for k, jam in jamset.iteritems():
            output_data[k] = jam.__json__

    with open(filepath, 'w') as fp:
        json.dump(output_data, fp)


def convolve(entity, graph, input_key, axis=1, chunk_size=250):
    """Apply a graph convolutionally to a field in an an entity.

    Parameters
    ----------
    entity : biggie.Entity
        Observation to predict.
    graph : optimus.Graph
        Network for processing an entity.
    data_key : str
        Name of the field to use for the input.
    chunk_size : int, default=None
        Number of slices to transform in a given step. When None, parses one
        slice at a time.

    Returns
    -------
    output : biggie.Entity
        Result of the convolution operation.
    """
    # TODO(ejhumphrey): Make this more stable, somewhat fragile as-is
    time_dim = graph.inputs.values()[0].shape[2]
    values = entity.values()
    input_stepper = optimus.array_stepper(
        values.pop(input_key), time_dim, axis=axis, mode='same')
    results = dict([(k, list()) for k in graph.outputs])
    if chunk_size:
        chunk = []
        for x in input_stepper:
            chunk.append(x)
            if len(chunk) == chunk_size:
                for k, v in graph(np.array(chunk)).items():
                    results[k].append(v)
                chunk = []
        if len(chunk):
            for k, v in graph(np.array(chunk)).items():
                results[k].append(v)
    else:
        for x in input_stepper:
            for k, v in graph(x[np.newaxis, ...]).items():
                results[k].append(v)
    for k in results:
        results[k] = np.concatenate(results[k], axis=0)
    values.update(results)
    return biggie.Entity(**values)


def process_stash(stash, transform, output, input_key, verbose=False):
    """Apply an optimus transform to all the entities in a stash, producing a
    separate output stash.

    Parameters
    ----------
    stash : biggie.Stash
        Collection of entities to transform.
    transform : optimus.Graph
        Network to apply to each entity.
    output : biggie.Stash
        Stash for writing outputs.
    input_key : str
        Name of the field to use for the input.
    """
    total_count = len(stash.keys())
    for idx, key in enumerate(stash.keys()):
        output.add(key, convolve(stash.get(key), transform, input_key))
        if verbose:
            print("[{0}] {1:7} / {2:7}: {3}".format(
                  time.asctime(), idx, total_count, key))

    output.close()


def translate(x_input, dim0=0, dim1=0, fill_value=0):
    """Translate a matrix in two dimensions.

    Parameters
    ----------
    x : np.ndarray
        Input 2d matrix.
    dim0 : int
        Shift along the first axis.
    dim1 : int
        Shift along the second axis.

    Returns
    -------
    y : np.ndarray
        The translated matrix.
    """
    # Sanity check
    assert x_input.ndim == 2, "Input must be 2D; ndim=%s" % x_input.ndim
    in_dim0, in_dim1 = x_input.shape
    z_output = np.zeros([in_dim0 + 2*abs(dim0), in_dim1 + 2*abs(dim1)],
                        dtype=x_input.dtype) + fill_value
    z_output[abs(dim0):abs(dim0) + in_dim0,
             abs(dim1):abs(dim1) + in_dim1] = x_input
    dim0 = 2*abs(dim0) if dim0 < 0 else 0
    dim1 = 2*abs(dim1) if dim1 < 0 else 0
    return z_output[dim0:dim0 + in_dim0, dim1:dim1 + in_dim1]


def circshift(x_input, dim0=0, dim1=0):
    """Circular shift a matrix in two dimensions.

    For example...

          dim0
         aaa|bb      dd|ccc
    dim1 ------  ->  ------
         ccc|dd      bb|aaa

    Default behavior is a pass-through.

    Parameters
    ----------
    x : np.ndarray
        Input 2d matrix.
    dim0 : int
        Rotation along the first axis.
    dim1 : int
        Rotation along the second axis.

    Returns
    -------
    y : np.ndarray
        The circularly shifted matrix.
    """
    # Sanity check
    assert x_input.ndim == 2, "Input must be 2D; ndim=%s" % x_input.ndim

    in_d0, in_d1 = x_input.shape
    z_output = np.zeros([in_d0, in_d1])

    # Make sure the rotation is bounded on [0,d0) & [0,d1)
    dim0, dim1 = dim0 % in_d0, dim1 % in_d1
    if not dim0 and dim1:
        z_output[:, :dim1] = x_input[:, -dim1:]  # A
        z_output[:, dim1:] = x_input[:, :-dim1]  # C
    elif not dim1 and dim0:
        z_output[:dim0, :] = x_input[-dim0:, :]  # A
        z_output[dim0:, :] = x_input[:-dim0, :]  # B
    elif dim0 and dim1:
        z_output[:dim0, :dim1] = x_input[-dim0:, -dim1:]  # A
        z_output[dim0:, :dim1] = x_input[:-dim0, -dim1:]  # B
        z_output[:dim0, dim1:] = x_input[-dim0:, :-dim1]  # C
        z_output[dim0:, dim1:] = x_input[:-dim0, :-dim1]  # D
    else:
        z_output = x_input
    return z_output
