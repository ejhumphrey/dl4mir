"""Utility to resample the time axis of Entities in a Biggie Stash."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import json
import time

import dl4mir.common.util as util


def find_closest_idx(x, y):
    return np.array([np.abs(x - v).argmin() for v in y])


def beat_sync(entity, new_times, new_labels=None, mode='median'):
    new_times = list(new_times)
    data = entity.values()
    time_points = data.pop('time_points')
    chord_labels = data.pop('chord_labels')

    idxs = find_closest_idx(time_points, new_times).tolist()
    if new_labels is None:
        chord_labels = chord_labels[idxs]
        # print "Best guess label interpolation! You should provide labels."
    else:
        chord_labels = np.asarray(new_labels)

    if idxs[0] != 0:
        idxs.insert(0, 0)
    if idxs[-1] != len(time_points) - 1:
        idxs.append(len(time_points) - 1)

    for key in data:
        if len(time_points) in [len(data[key]), len(data[key]) - 1]:
            data[key] = util.boundary_pool(data[key], idxs, pool_func=mode)

    return biggie.Entity(time_points=new_times,
                         chord_labels=chord_labels,
                         endT=time_points[-1],
                         **data)


def main(args):
    dset = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])
    dout = biggie.Stash(args.output_file)
    beat_times = json.load(open(args.beat_times))
    total_count = len(dset)
    for idx, key in enumerate(dset.keys()):
        dout.add(key, beat_sync(dset.get(key), beat_times[key]))
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
    main(parser.parse_args())
