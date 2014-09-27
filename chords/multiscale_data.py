"""Utility to resample the time axis of Entities in a Biggie Stash."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import time
from scipy import signal


def multiscale_pool(x_in, lags, axis=0, mode='mean'):
    output = [x_in]
    for l in lags:
        if mode == 'median':
            filt = [1] * x_in.ndim
            filt[axis] = l
            z = signal.medfilt(x_in, filt)
        elif mode == 'mean':
            z = signal.lfilter(
                np.ones(l, dtype=float)/l, np.ones(1), x_in, axis=axis)
        else:
            raise ValueError("Filter mode `%s` unsupported." % mode)
        output.append(z)
    # axes = range(1, x_in.ndim + 1)
    return np.array(output)
    # return np.transpose(np.array(output), axes)


def pool_entity(entity, key, *args, **kwargs):
    values = entity.values()
    values[key] = multiscale_pool(values.pop(key), *args, **kwargs)
    return biggie.Entity(**values)


def main(args):
    stash = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])
    stash_out = biggie.Stash(args.output_file)
    total_count = len(stash)
    args = ['chroma', [4, 8, 16, 32, 64], 0, 'mean']
    for idx, key in enumerate(stash.keys()):
        stash_out.add(key, pool_entity(stash.get(key), *args))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    stash_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beat-synchronize a dataset of entities.")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Path to the input biggie file.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to the output biggie file.")
    main(parser.parse_args())
