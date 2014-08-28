"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import biggie
import os
import marl.fileutils as futils
import time

import dl4mir.common.util as util


def wrap_entity(entity, length, stride):
    """"""
    entity.cqt = util.fold_array(entity.cqt.value[0], length, stride)
    return entity


def main(args):
    in_stash = biggie.Stash(args.data_file)
    futils.create_directory(os.path.split(args.output_file)[0])
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    out_stash = biggie.Stash(args.output_file)
    total_count = len(in_stash.keys())
    for idx, key in enumerate(in_stash.keys()):
        out_stash.add(
            key, wrap_entity(in_stash.get(key), args.length, args.stride))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    out_stash.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("length",
                        metavar="length", type=int,
                        help="Number of bins per CQT slice.")
    parser.add_argument("stride",
                        metavar="stride", type=int,
                        help="Number of bins between slices, i.e. an octave.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the transformed output.")
    main(parser.parse_args())
