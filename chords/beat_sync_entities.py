"""Utility to dump an Biggie Stash to a flat collection of Matlab files."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import json
import time


def beat_sync(entity, new_times):
    new_times = list(new_times)
    data = entity.values
    time_points = data.pop('time_points')
    if not time_points[0] in new_times:
        new_times.insert(0, time_points[0])
    if not time_points[-1] in new_times:
        new_times.append(time_points[-1])
    new_times.sort()

    idxs = np.array([(t > time_points).astype(int).argmin() - 1
                     for t in new_times[:-1]])
    for key in data:
        shape = data[key].shape
        if len(time_points) - 1 in shape:
            data[key] = data[key][idxs, ...]

    return biggie.Entity(time_points=new_times, **data)


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
