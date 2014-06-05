"""Utility to dump an optimus file to a flat collection of Matlab files."""
import argparse
import marl.fileutils as futils
from os import path
import optimus
import scipy.io.matlab as MLAB
import time


def main(args):
    dset = optimus.File(args.input_file)
    out_dir = futils.create_directory(args.output_directory)
    total_count = len(dset)
    for idx, key in enumerate(dset.keys()):
        out_file = path.join(out_dir, "%s.mat" % key)
        entity = dset.get(key)
        MLAB.savemat(out_file, mdict=entity.values)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump an Optimus File to Matlab .mat files.")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Path to a file to export.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
