"""Loader for multiple data splits into optimus Files."""

import argparse
import json
from marl import fileutils as futils
import numpy as np
import biggie
from os import path
import time

FILE_FMT = "{subset}/{fold_idx}/{split}.hdf5"
NPZ_EXT = "npz"


def create_entity(npz_file, dtype=np.float32):
    """Create an entity from the given file.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.
    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated entity, with the following fields:
            {cqt, time_points, icode, note_number, fcode}.
    """
    (icode, note_number,
        fcode) = [np.array(_) for _ in futils.filebase(npz_file).split('_')]
    entity = biggie.Entity(icode=icode, note_number=note_number,
                           fcode=fcode, **np.load(npz_file))
    entity.cqt = entity.cqt.astype(dtype)
    return entity


def populate_stash(keys, cqt_directory, stash, dtype=np.float32):
    """Populate a Stash with cqt data.

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
        cqt_file = path.join(cqt_directory, "{0}.{1}".format(key, NPZ_EXT))
        stash.add(key, create_entity(cqt_file, dtype))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    partitions = json.load(open(args.split_file))

    output_file_fmt = path.join(args.output_directory, FILE_FMT)
    for set_name, subset in partitions.items():
        for fold_idx, splits in subset.items():
            for split, keys in splits.items():
                output_file = output_file_fmt.format(
                    subset=set_name, fold_idx=fold_idx, split=split)
                futils.create_directory(path.split(output_file)[0])
                if args.verbose:
                    print "[%s] Creating: %s" % (time.asctime(), output_file)
                stash = biggie.Stash(output_file)
                populate_stash(keys, args.cqt_directory, stash, np.float32)
                stash.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load chord data into optimus files")
    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="Path to splits of the data as JSON.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Directory containing CQT npz files.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Base directory for the output files.")
    parser.add_argument("--verbose",
                        metavar="--verbose", type=bool, default=True,
                        help="Toggle console printing.")

    main(parser.parse_args())
