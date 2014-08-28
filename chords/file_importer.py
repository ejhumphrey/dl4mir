"""Loader for multiple data splits into optimus Files."""

import argparse
import json
from marl import fileutils as futils
import mir_eval
import dl4mir.chords.labels as L
import numpy as np
import biggie
from os import path
import time

# fold / split
FILE_FMT = "%s/%s.hdf5"
LAB_EXT = "json"
NPZ_EXT = "npz"

def create_chord_entity(npz_file, lab_file, dtype=np.float32):
    """Create an entity from the given files.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.
    lab_file: str
        Path to a corresponding lab-file.
    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated chord entity, with {cqt, chord_labels, *time_points}.
    """
    entity = biggie.Entity(**np.load(npz_file))
    intervals, labels = L.load_labeled_intervals(lab_file)
    entity.chord_labels = mir_eval.util.interpolate_intervals(
        intervals, labels, entity.time_points.value, fill_value='N')
    entity.cqt = entity.cqt.value.astype(dtype)
    return entity


def populate_stash(keys, cqt_directory, lab_directory, stash,
                   dtype=np.float32):
    """Populate a Stash with chord data.

    Parameters
    ----------
    keys: list
        Collection of fileset keys, of which a npz- and lab-file exist.
    cqt_directory: str
        Base path for CQT npz-files.
    lab_directory: str
        Base path for chord lab-files.
    stash: biggie.Stash
        Stash for writing entities to disk.
    dtype: type
        Data type for the cqt array.
    """
    total_count = len(keys)
    for idx, key in enumerate(keys):
        cqt_file = path.join(cqt_directory, "%s.%s" % (key, NPZ_EXT))
        lab_file = path.join(lab_directory, "%s.%s" % (key, LAB_EXT))
        stash.add(key, create_chord_entity(cqt_file, lab_file, dtype))
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
                args.lab_directory, stash, np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load chord data into optimus files")
    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="Path to splits of the data as JSON.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT npz files.")
    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Directory containing chord lab files.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base directory for the output files.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")

    main(parser.parse_args())
