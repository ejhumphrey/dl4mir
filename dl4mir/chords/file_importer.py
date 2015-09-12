"""Loader for multiple data splits into optimus Files."""

import argparse
import json
from marl import fileutils as futils
import mir_eval
import pyjams
import numpy as np
import biggie
from os import path
import time

# fold / split
FILE_FMT = "%s/%s.hdf5"
JAMS_EXT = "jams"
NPZ_EXT = "npz"


def create_chord_entity(npz_file, jams_file, dtype=np.float32):
    """Create an entity from the given files.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.
    jams_file: str
        Path to a corresponding JAMS file.
    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated chord entity, with {cqt, chord_labels, *time_points}.
    """
    entity = biggie.Entity(**np.load(npz_file))
    jam = pyjams.load(jams_file)
    intervals = np.asarray(jam.chord[0].intervals)
    labels = [str(_) for _ in jam.chord[0].labels.value]
    entity.chord_labels = mir_eval.util.interpolate_intervals(
        intervals, labels, entity.time_points, fill_value='N')
    entity.cqt = entity.cqt.astype(dtype)
    return entity


def populate_stash(keys, cqt_directory, jams_directory, stash,
                   dtype=np.float32):
    """Populate a Stash with chord data.

    Parameters
    ----------
    keys: list
        Collection of fileset keys, of which a npz- and lab-file exist.
    cqt_directory: str
        Base path for CQT npz-files.
    jams_directory: str
        Base path for reference JAMS files.
    stash: biggie.Stash
        Stash for writing entities to disk.
    dtype: type
        Data type for the cqt array.
    """
    total_count = len(keys)
    for idx, key in enumerate(keys):
        cqt_file = path.join(cqt_directory, "%s.%s" % (key, NPZ_EXT))
        jams_file = path.join(jams_directory, "%s.%s" % (key, JAMS_EXT))
        stash.add(key, create_chord_entity(cqt_file, jams_file, dtype))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    data_splits = json.load(open(args.split_file))

    output_file_fmt = path.join(args.output_directory, FILE_FMT)
    for fold in data_splits:
        for split in data_splits[fold]:
            output_file = output_file_fmt % (fold, split)
            futils.create_directory(path.split(output_file)[0])
            if args.verbose:
                print "[%s] Creating: %s" % (time.asctime(), output_file)
            stash = biggie.Stash(output_file)
            populate_stash(
                data_splits[fold][split], args.cqt_directory,
                args.jams_directory, stash, np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load chord data into optimus files")
    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="Path to splits of the data as JSON.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT npz files.")
    parser.add_argument("jams_directory",
                        metavar="jams_directory", type=str,
                        help="Directory containing reference JAMS files.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base directory for the output files.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")

    main(parser.parse_args())
