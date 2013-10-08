"""Import a collection of CQT arrays into a Hewey DataSequenceFile.

Sample Call:
$ ipython ejhumphrey/scripts/load_chords_into_hewey.py \
/Volumes/Audio/Chord_Recognition/labeled_tracks_20130930_train0.txt \
/Volumes/Audio/Chord_Recognition/cqts/20131002 \
/Volumes/speedy/chordrec_lcn_train0_20131002.dsf
"""

import argparse
import json
import numpy as np
import os
import time

from ejhumphrey.datasets import chordutils, utils
from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen


def pair_cqt_to_lab_file(lab_file, cqt_directory):
    """Use a lab_file to recover a corresponding cqt_file.

    Parameters
    ----------
    lab_file : str
        File (or file base) to key on.
    cqt_directory : str
        Path to search.

    Returns
    -------
    cqt_file : str
        Absolute file path to a numpy array; guaranteed to exist.
    """
    cqt_file = utils.expand_filebase(
        utils.filebase(lab_file), cqt_directory, "npy")
    assert os.path.exists(cqt_file), "Could not find file: %s" % cqt_file
    return cqt_file


def create_datasequence_file(split_file, cqt_directory, filename, cqt_params):
    """
    Parameters
    ----------
    split_file : string
        Text file containing a list of lab files.
    cqt_directory : string
        Path to search for CQT numpy files.
    filename : string
        Output name for the DataSequenceFile
    cqt_params : dict
        Parameters used to compute the CQT.
    """
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)

    for i, line in enumerate(open(split_file)):
        lab_file = line.strip("\n")
        cqt_file = pair_cqt_to_lab_file(lab_file, cqt_directory)
        print "%03d: Importing %s" % (i, cqt_file)
        cqt_array = np.load(cqt_file)
        labels = chordutils.align_lab_file_to_array(
            cqt_array, lab_file, cqt_params.get("framerate"))
        metadata = {"timestamp": time.asctime(),
                    "filesource": cqt_file}
        dseq = DataSequence(value=cqt_array, label=labels, metadata=metadata)
        key = keygen.next()
        while key in file_handle:
            key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    cqt_params_file = os.path.join(args.cqt_directory, "cqt_params.txt")
    assert os.path.exists(cqt_params_file), \
        "CQT param file does not exist: %s" % cqt_params_file
    cqt_params = json.load(open(cqt_params_file))
    create_datasequence_file(
        args.split_file, args.cqt_directory, args.output_file, cqt_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import CQT arrays into a DataSequenceFile for training.")

    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="List of lab files to import; probably a training "
                        "or validation split.")

    parser.add_argument("cqt_directory",
                        metavar="track_split", type=str,
                        help="Base path to search for CQT arrays. Must also "
                        "contain a cqt_params.txt file.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the DataSequenceFile.")

    main(parser.parse_args())
