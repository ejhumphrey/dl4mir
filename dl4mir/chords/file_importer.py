"""Loader for multiple data splits into optimus Files."""

import argparse
import json
from marl import fileutils as futils
import mir_eval
import numpy as np
import optimus
from os import path
import time

# fold / split
FILE_FMT = "%s/%s.hdf5"
# TODO: Get rid of this in the future.
TIME_AXIS = 1


def create_entity(cqt_file, lab_file, cqt_params, dtype=np.float32):
    """Create an entity from the given item.

    This function exists primarily as an example, and is quite boring. However,
    it expects that each item dictionary has two keys:
        - numpy_file: str
            A valid numpy file on disk
        - label: obj
            This can be any numpy-able datatype, such as scalars, lists,
            strings, or numpy arrays. Dictionaries and None are unsupported.

    Parameters
    ----------
    item: dict
        Contains values for 'numpy_file' and 'label'.
    dtype: type
        Data type to load the requested numpy file.
    """
    data = np.load(cqt_file)
    intervals, labels = mir_eval.io.load_intervals(lab_file)
    framerate = float(cqt_params['framerate'])
    time_points = np.arange(data.shape[TIME_AXIS]) / framerate
    chord_labels = mir_eval.util.interpolate_intervals(
        intervals, labels, time_points, fill_value='N')

    return optimus.Entity(
        cqt=data.astype(dtype),
        chord_labels=chord_labels,
        time_points=time_points)


def data_to_file(keys, cqt_directory, cqt_params, lab_directory, file_handle,
                 item_parser, dtype=np.float32):
    """Load a label dictionary into an optimus file.

    Parameters
    ----------
    keys: dict of dicts
        A collection of file_pairs to load, where the keys of ``file_pairs``
        will become the keys in the file, and the corresponding values are
        sufficient information to load data into an Entity.
    file_handle: optimus.File
        Open for writing data.
    config: dict
        Dictionary containing configuration parameters for this script.
    item_parser: function
        Function that consumes a dictionary of information and returns an
        optimus.Entity. Must take ``dtype`` as an argument.
    """
    total_count = len(keys)
    for idx, key in enumerate(keys):
        cqt_file = path.join(cqt_directory, "%s.npy" % key)
        lab_file = path.join(lab_directory, "%s.lab" % key)
        file_handle.add(
            key, item_parser(cqt_file, lab_file, cqt_params, dtype))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    data_splits = json.load(open(args.split_file))
    cqt_params = json.load(open(args.cqt_params))
    output_file_fmt = path.join(args.output_directory, FILE_FMT)
    for fold in data_splits:
        for split in data_splits[fold]:
            output_file = output_file_fmt % (fold, split)
            futils.create_directory(path.split(output_file)[0])
            if args.verbose:
                print "[%s] Creating: %s" % (time.asctime(), output_file)
            fhandle = optimus.File(output_file)
            data_to_file(
                data_splits[fold][split], args.cqt_directory,
                cqt_params, args.lab_directory, fhandle, create_entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load chord data into optimus files")
    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="Path to splits of the data as JSON.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT numpy files.")
    parser.add_argument("cqt_params",
                        metavar="cqt_params", type=str,
                        help="Parameters used to compute the CQTs.")
    parser.add_argument("lab_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT numpy files.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base directory for the output files.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")

    main(parser.parse_args())
