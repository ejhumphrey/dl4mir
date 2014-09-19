"""Utility to resample the time axis of Entities in a Biggie Stash."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import json
import time

import dl4mir.common.util as util


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
    return np.array([np.abs(x - v).argmin() for v in y])


def subdivide_boundaries(time_boundaries, num_per_interval):
    durations = boundaries_to_durations(time_boundaries)

    offsets = np.arange(num_per_interval, dtype=float) / num_per_interval
    new_boundaries = list()
    for tstart, dur in zip(time_boundaries, durations):
        new_boundaries.extend((tstart + dur*offsets).tolist())

    return new_boundaries + [time_boundaries[-1]]


def beat_sync(entity, time_boundaries, new_labels=None, mode='median'):
    time_boundaries = list(time_boundaries)
    if time_boundaries[0] != 0.0:
        raise ValueError("Time boundaries should really start from 0.")

    data = entity.values()
    time_points = data.pop('time_points')
    chord_labels = data.pop('chord_labels')

    idxs = find_closest_idx(time_points, time_boundaries).tolist()
    if new_labels is None:
        chord_labels = util.boundary_pool(chord_labels, idxs, pool_func='mode')
    else:
        chord_labels = np.asarray(new_labels)

    for key in data:
        data_shape = list(data[key].shape)
        if len(time_points) in data_shape:
            axis = data_shape.index(len(time_points))
            data[key] = util.boundary_pool(data[key], idxs,
                                           pool_func=mode, axis=axis)

    return biggie.Entity(time_points=time_boundaries[:-1],
                         chord_labels=chord_labels,
                         durations=boundaries_to_durations(time_boundaries),
                         **data)


def main(args):
    dset = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])
    dout = biggie.Stash(args.output_file)
    beat_times = json.load(open(args.beat_times))
    total_count = len(dset)
    for idx, key in enumerate(dset.keys()):
        boundaries = subdivide_boundaries(beat_times[key], args.subdivide)
        dout.add(key, beat_sync(dset.get(key), boundaries))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    dout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beat-synchronize a dataset of entities.")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Path to the input biggie file.")
    parser.add_argument("beat_times",
                        metavar="beat_times", type=str,
                        help="JSON file of new times; keys must match.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to the output biggie file.")
    parser.add_argument("--subdivide",
                        metavar="--subdivide", type=int, default=1,
                        help="Subdivisions per interval.")
    main(parser.parse_args())
