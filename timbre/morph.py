import dl4mir.timbre.data as D
import numpy as np
import random

from scipy.spatial.distance import cdist
from scipy.signal import lfilter
from sklearn.neighbors import kneighbors_graph

from marl.audio.fileio import FramedAudioReader


def find_nearest_points(path, stash, num_points=10000):
    x, y, keys, time_points = D.sample_embedding_stash(stash, num_points)
    path_keys, path_time_points = [], []
    for coords in path:
        idx = cdist(x, coords.reshape(1, -1)).argmin()
        path_keys.append(keys[idx])
        path_time_points.append(time_points[idx])

    return path_keys, path_time_points


def randwalk(length, dims=3, scale=0.25, mean_filt_len=5,
             means=None, scales=None):
    """Create a line using a random walk algorithm.

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    path = np.empty((dims, length))
    path[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        step = ((np.random.rand(dims) - 0.5) * scale)
        path[:, index] = path[:, index-1] + step

    w_n = np.ones(mean_filt_len) / float(mean_filt_len)
    path = lfilter(w_n, 1.0, path.T, axis=0)
    p_means = path.mean(axis=0).reshape(1, -1)
    p_scales = path.std(axis=0).reshape(1, -1)
    path = (path - p_means) / p_scales
    if not all([_ is None for _ in means, scales]):
        path = (path * scales.reshape(1, -1)) + means.reshape(1, -1)
    return path


def randpath(coords, path_length, num_neighbors=10):
    connections = kneighbors_graph(coords, num_neighbors)
    path_index = np.zeros([path_length], dtype=int)
    last_idx = np.random.randint(len(coords))
    this_idx = last_idx
    for n in range(path_length):
        while this_idx == last_idx:
            this_idx = random.choice(connections[last_idx].nonzero()[1])

        path_index[n] = this_idx
        last_idx = this_idx
    return path_index


def extract_grains(files, time_points, samplerate=44100, duration=0.5):
    grains = []
    framesize = int(samplerate * duration)
    for f, t in zip(files, time_points):
        ar = FramedAudioReader(
            f, samplerate=samplerate, framesize=framesize, channels=1,
            time_points=[t])
        grains.append(ar.next())
    return grains


def overlap_and_add(grains, w_n=None):
    framesize, channels = grains[0].shape
    if w_n is None:
        w_n = np.hanning(framesize).reshape(-1, 1)

    y_out = np.zeros([framesize * (len(grains) / 2 + 2), channels])
    for idx, x_n in enumerate(grains):
        i0 = idx * framesize / 2
        i1 = i0 + framesize
        scalar = 0.5*np.abs(x_n).max() if np.abs(x_n).max() > 0.0001 else 1
        y_out[i0:i1, :] += w_n * x_n / scalar
    return y_out
