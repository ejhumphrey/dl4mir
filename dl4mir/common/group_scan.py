import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.special import gammaln

def histogram_distances(x_obs, n_bins=100, metric='euclidean', chunk_size=100):
    n_obs, n_dim = x_obs.shape
    bins = np.zeros([n_obs, n_bins])
    edges = np.zeros([n_obs, n_bins + 1])
    for idx in range(0, n_obs, chunk_size):
        dists = cdist(x_obs[idx:idx + chunk_size], x_obs, metric=metric)
        for n, d in enumerate(dists):
            be = np.histogram(d[np.not_equal(d, 0.0)], bins=n_bins)
            bins[idx + n], edges[idx + n] = be
    return bins, edges


def hypersphere_volume(radius, n_dim):
    vol = n_dim * np.log(radius)
    vol += np.log(np.power(np.pi, n_dim / 2.0))
    return vol - gammaln(n_dim / 2.0 + 1.0)


def find_next_center_index(x_obs, target_count=500, num_steps=100,
                           min_count=10):
    bins, edges = histogram_distances(x_obs)
    cdf = np.cumsum(bins, axis=1)
    x_dist = edges[:, 1:]
    best_idx = None
    safety_idx = None
    for cutoff in np.linspace(x_dist.min(), x_dist.mean(), num_steps):
        fitness = cdf[np.arange(len(cdf)), (x_dist < cutoff).argmin(axis=1)]
        idx = fitness.argmax()
        if fitness[idx] >= target_count:
            best_idx = idx
            break
        elif fitness[idx] >= min_count:
            safety_idx = idx
    if best_idx is None and safety_idx is None:
        #raise ValueError("Target Count too high! Reduce and try again.")
        return find_next_center(x_obs, target_count / 2, num_steps)
    elif safety_idx is None:
        return None, None
    return best_idx, cutoff


def find_next_radial_center(x_obs, radius):
    bins, edges = histogram_distances(x_obs)
    cdf = np.cumsum(bins, axis=1)
    fitness = cdf[np.arange(len(cdf)), (edges[:, 1:] < radius).argmin(axis=1)]
    return fitness.argmax()


def find_next_density_center(x_obs, max_density=None):
    bins, edges = histogram_distances(x_obs)
    cdf = np.cumsum(bins, axis=1)
    radius = edges[:, 1:]
    volumes = hypersphere_volume(radius, x_obs.shape[1])
    density = cdf / volumes
    if not max_density is None:
        density[density > max_density] = -np.inf
    best_idx = density.max(axis=1).argmax()
    radius = radius[best_idx, density[best_idx].argmax()]
    return best_idx, radius, density.max()


def assign_to_group(x_obs, center, cutoff, metric='euclidean'):
    """
    Returns
    -------
    in_set : np.ndarray, dtype=bool
        Binary array indicating whether or not the row belongs to the cluster.
    """
    if center.ndim == 1:
        center = center[np.newaxis, :]
    dist = cdist(x_obs, center, metric).flatten()
    return (dist <= cutoff)


def cluster(x_obs, target_count):
    done = False
    n_obs, n_dim = x_obs.shape
    centers = []
    labels = -np.ones(n_obs)
    full_idx = np.arange(n_obs)
    while not done:
        subidx, cutoff = find_next_center_index(x_obs, target_count)
        if subidx is None:
            done = True
            break
        in_set = assign_to_group(x_obs, x_obs[subidx], cutoff)
        real_idxs = full_idx[in_set]
        labels[real_idxs] = len(centers)
        centers.append(full_idx[subidx])
        print "[%s] Grouped %d datapoints" % (time.asctime(), in_set.sum())
        x_obs = x_obs[np.invert(in_set), :]
        full_idx = full_idx[np.invert(in_set)]
    return np.array(centers), labels


def radial_cluster(x_obs, radius=None, min_count=50):
    if radius is None:
        bins, edges = histogram_distances(x_obs)
        radius = edges.mean()
        print "Setting radius to %0.4f" % radius
    done = False
    n_obs, n_dim = x_obs.shape
    center_idx = []
    labels = -np.ones(n_obs)
    full_idx = np.arange(n_obs)
    while not done:
        subidx = find_next_radial_center(x_obs, radius)
        in_set = assign_to_group(x_obs, x_obs[subidx], radius)
        if in_set.sum() < min_count:
            done = True
            break
        real_idxs = full_idx[in_set]
        labels[real_idxs] = len(center_idx)
        center_idx.append(full_idx[subidx])
        print "[%s] Grouped %d datapoints" % (time.asctime(), in_set.sum())
        x_obs = x_obs[np.invert(in_set), :]
        full_idx = full_idx[np.invert(in_set)]
    return np.array(center_idx), labels


def density_cluster(x_obs, target_density=None, min_density=0.001):
    done = False
    n_obs, n_dim = x_obs.shape
    center_idx = []
    labels = -np.ones(n_obs)
    full_idx = np.arange(n_obs)
    while not done:
        subidx, radius, density = find_next_density_center(x_obs,
                                                           target_density)
        in_set = assign_to_group(x_obs, x_obs[subidx], radius)
        if density < min_density:
            done = True
            break
        real_idxs = full_idx[in_set]
        labels[real_idxs] = len(center_idx)
        center_idx.append(full_idx[subidx])
        print "[%s] Grouped %d datapoints (%0.4f)" % (time.asctime(),
                                                      in_set.sum(),
                                                      density)
        x_obs = x_obs[np.invert(in_set), :]
        full_idx = full_idx[np.invert(in_set)]
        if len(full_idx) == 0:
            done = True
    return np.array(center_idx), labels
