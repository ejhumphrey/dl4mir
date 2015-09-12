"""Utility to resample the time axis of Entities in a Biggie Stash."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import json
import time

import dl4mir.common.util as util


def subdivide_boundaries(time_boundaries, num_per_interval):
    """Evenly subdivide a list of boundaries in time."""
    durations = util.boundaries_to_durations(time_boundaries)

    offsets = np.arange(num_per_interval, dtype=float) / num_per_interval
    new_boundaries = list()
    for tstart, dur in zip(time_boundaries, durations):
        new_boundaries.extend((tstart + dur*offsets).tolist())

    return new_boundaries + [time_boundaries[-1]]


def beat_sync(entity, time_boundaries, new_labels=None, pool_func='median'):
    """Beat-synchronize an entity to a set of boundaries in time.

    Parameters
    ----------
    entity : biggie.Entity
        Required fields {time_points, chord_labels}
    time_boundaries : array_like, shape=(N,)
        List of boundary points over which to pool data.
    new_labels : array_like
        Set of pre-aligned labels to over-ride the current ones.
    pool_func : str
        Method of pooling data; one of ['mean', 'median'].

    Returns
    -------
    new_entity : biggie.Entity
        Same fields as input entity, with additional `durations` field.
    """
    time_boundaries = list(time_boundaries)
    if time_boundaries[0] != 0.0:
        raise ValueError("Time boundaries should really start from 0.")

    data = entity.values()
    time_points = data.pop('time_points')
    chord_labels = data.pop('chord_labels')

    idxs = util.find_closest_idx(time_points, time_boundaries).tolist()
    if new_labels is None:
        chord_labels = util.boundary_pool(chord_labels, idxs, pool_func='mode')
    else:
        chord_labels = np.asarray(new_labels)

    for key in data:
        data_shape = list(data[key].shape)
        if len(time_points) in data_shape:
            axis = data_shape.index(len(time_points))
            dtype = data[key].dtype.type
            pool_func = 'mode' if dtype == np.string_ else pool_func
            data[key] = util.boundary_pool(data[key], idxs,
                                           pool_func=pool_func, axis=axis)

    return biggie.Entity(
        time_points=time_boundaries[:-1],
        chord_labels=chord_labels,
        durations=util.boundaries_to_durations(time_boundaries),
        **data)


def main(args):
    dset = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])
    dout = biggie.Stash(args.output_file)
    beat_times = json.load(open(args.beat_times))
    total_count = len(dset)
    for idx, key in enumerate(dset.keys()):
        boundaries = subdivide_boundaries(beat_times[key], args.subdivide)
        dout.add(key, beat_sync(dset.get(key),
                                boundaries,
                                pool_func=args.pool_func))
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
    parser.add_argument("--pool_func",
                        metavar="--pool_func", type=str, default='median',
                        help="Method of pooling numerical data.")
    main(parser.parse_args())
