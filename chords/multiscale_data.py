"""Utility to resample the time axis of Entities in a Biggie Stash."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import biggie
import time
from scipy import signal

import dl4mir.common.util as util


def lag_filter(x_in, lags, axis=0, mode='mean'):
    x_in = x_in.squeeze()
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
    return util.lp_scale(np.array(output), axis=-1)


def pool_entity(entity, key, *args, **kwargs):
    values = entity.values()
    values[key] = lag_filter(values.pop(key), *args, **kwargs)
    return biggie.Entity(**values)


def main(args):
    stash = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])
    stash_out = biggie.Stash(args.output_file)
    total_count = len(stash)
    args = ['cqt', [2, 4, 8, 16, 32, 64, 128], 0, 'mean']
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
