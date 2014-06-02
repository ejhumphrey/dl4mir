"""Loader for multiple data splits into optimus Files."""

import argparse
import json
from marl import fileutils as futils
import mir_eval
import ejhumphrey.dl4mir.guitar.util as gutil
import numpy as np
import optimus
from os import path
import time
import pychords.guitar as G

# fold / split
FILE_FMT = "%s/%s.hdf5"
# TODO: Get rid of this in the future.
TIME_AXIS = 1


def coverage(tab_file):
    data = json.load(open(tab_file))
    intervals, labels = np.array(data['intervals']), data['labels']
    x_labels = np.array([float(G.OFF_CHAR != l) for l in labels])
    durations = np.abs(np.diff(intervals, axis=1)).squeeze()
    return (x_labels * durations).sum() / durations.sum()


def create_entity(cqt_file, tab_file, cqt_params, dtype=np.float32):
    """Create an entity from the given data.

    Parameters
    ----------
    cqt_file: str
        Path to a CQT numpy file.
    tab_file: str
        Path to a JSON tab file.
    """
    data = np.load(cqt_file)
    intervals, labels = gutil.load_tab(tab_file)
    framerate = float(cqt_params['framerate'])
    time_points = np.arange(data.shape[TIME_AXIS]) / framerate
    chord_labels = mir_eval.util.interpolate_intervals(
        intervals, labels, time_points, fill_value=G.NO_CHORD)

    return optimus.Entity(cqt=data.astype(dtype), fret_labels=chord_labels)


def data_to_file(keys, cqt_directory, cqt_params, tab_directory, file_handle,
                 item_parser, coverage_threshold, dtype=np.float32):
    """Load a label dictionary into an optimus file.

    Parameters
    ----------
    keys: dict of dicts
        A collection of file_pairs to load, where the keys of ``file_pairs``
        will become the keys in the file, and the corresponding values are
        sufficient information to load data into an Entity.
    cqt_directory: str
        Path to a flat collection of '.npy' CQT files.
    cqt_params: dict
        Parameter dictionary used to compute the CQTs.
    tab_directory: str
        Path to a flat collection of '.tab' (JSON) files.
    file_handle: optimus.File
        Open for writing data.
    item_parser: function
        Function that consumes a dictionary of information and returns an
        optimus.Entity. Must take ``dtype`` as an argument.
    """
    total_count = len(keys)
    for idx, key in enumerate(keys):
        cqt_file = path.join(cqt_directory, "%s.npy" % key)
        tab_file = path.join(tab_directory, "%s.tab" % key)
        if path.exists(tab_file):
            if coverage(tab_file) < coverage_threshold:
                print "[%s] Skipping: %s" % (time.asctime(), key)
        if path.exists(cqt_file) and path.exists(tab_file):
            file_handle.add(
                key, item_parser(cqt_file, tab_file, cqt_params, dtype))
            print "[%s] %12d / %12d: %s" % (time.asctime(), idx,
                                            total_count, key)
        else:
            print "[%s] Skipping: %s" % (time.asctime(), key)


def main(args):
    """Main routine for importing data."""
    data_splits = json.load(open(args.split_file))
    cqt_params = json.load(open(args.cqt_params))
    output_file_fmt = path.join(args.output_directory, FILE_FMT)
    for fold in data_splits:
        for split in data_splits[fold]:
            output_file = output_file_fmt % (fold, split)
            if path.exists(output_file):
                continue
            futils.create_directory(path.split(output_file)[0])
            if args.verbose:
                print "[%s] Creating: %s" % (time.asctime(), output_file)
            fhandle = optimus.File(output_file)
            data_to_file(
                data_splits[fold][split], args.cqt_directory,
                cqt_params, args.tab_directory, fhandle, create_entity,
                args.coverage_threshold)


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
    parser.add_argument("tab_directory",
                        metavar="tab_directory", type=str,
                        help="Directory containing CQT numpy files.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base directory for the output files.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")
    parser.add_argument("--coverage_threshold", default=0.75,
                        metavar="--coverage_threshold", type=float,
                        help="Toggle console printing.")

    main(parser.parse_args())
