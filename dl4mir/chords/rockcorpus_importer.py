"""Loader for multiple data splits into optimus Files."""

import argparse
import dl4mir.common.fileutil as futils
import mir_eval
import dl4mir.chords.labels as L
import numpy as np
import biggie
from os import path
import time

# fold / split
FILE_FMT = "%s/%s.hdf5"
LAB_EXT = "lab"
NPZ_EXT = "npz"


def create_chord_entity(npz_file, lab_files, dtype=np.float32):
    """Create an entity from the given files.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.
    lab_files: list
        Collection of paths to corresponding lab-files.
    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated chord entity, with {cqt, chord_labels, time_points}.
    """
    entity = biggie.Entity(**np.load(npz_file))
    chord_labels = []
    for lab_file in lab_files:
        intervals, labels = L.load_labeled_intervals(lab_file, compress=True)
        labels = mir_eval.util.interpolate_intervals(
            intervals, labels, entity.time_points, fill_value='N')
        chord_labels.append(labels)

    entity.chord_labels = np.array(chord_labels).T
    entity.cqt = entity.cqt.astype(dtype)
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
        dt_file = path.join(lab_directory, "%s_dt.%s" % (key, LAB_EXT))
        tdc_file = path.join(lab_directory, "%s_tdc.%s" % (key, LAB_EXT))
        entity = create_chord_entity(cqt_file, [dt_file, tdc_file], dtype)
        stash.add(key, entity)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    futils.create_directory(path.split(args.output_file)[0])
    if args.verbose:
        print "[%s] Creating: %s" % (time.asctime(), args.output_file)
    stash = biggie.Stash(args.output_file)
    populate_stash(
        futils.load_textlist(args.key_list), args.cqt_directory,
        args.lab_directory, stash, np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load chord data into optimus files")
    parser.add_argument("key_list",
                        metavar="key_list", type=str,
                        help="Path to a set of keys to import.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT npz files.")
    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Directory containing chord lab files.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the output stash.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")

    main(parser.parse_args())
