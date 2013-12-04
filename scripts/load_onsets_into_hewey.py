"""Import a dataset into a Hewey DataSequenceFile.

Sample Call:
$ python ejhumphrey/scripts/load_onsets_into_hewey.py \
/Volumes/Audio/onsets/cqts \
/Volumes/Audio/onsets/labels \
/Volumes/speedy/onsets_20131123.dsf
"""

import argparse
import glob
import numpy as np
import os
import time

from ejhumphrey.datasets import onsets, utils
from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen


def pair_data_and_labels(data_directory, lab_directory):
    """

    Parameters
    ----------
    lab_file : str
        File (or file base) to key on.
    data_directory : str
        Path to search.

    Returns
    -------
    np_file : str
        Absolute file path to a numpy array; guaranteed to exist.
    """
    lab_files = []
    np_files = []
    for lab_file in glob.glob(os.path.join(lab_directory, "*.txt")):
        np_file = utils.expand_filebase(
            utils.filebase(lab_file), data_directory, "npy")
        if os.path.exists(np_file):
            lab_files.append(lab_file)
            np_files.append(np_file)
    print "Found %d files." % len(np_files)
    return np_files, lab_files


def add_to_datasequence_file(np_files, lab_files, filename, framerate):
    """
    Parameters
    ----------
    np_files: list
    lab_files: list
    filename : string
        Output name for the DataSequenceFile
    framerate: scalar
    """
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)

    for np_file, lab_file in zip(np_files, lab_files):
        print "[%s] Importing %s" % (time.asctime(), np_file)
        np_array = np.load(np_file)
        onset_times = onsets.load_onset_labels(lab_file)
        y_true = onsets.align_onset_labels(np_array, onset_times, framerate)
        labels = ["%d" % v for v in y_true.astype(int)]
        metadata = {"timestamp": time.asctime(), "filesource": np_file}
        dseq = DataSequence(value=np_array, label=labels, metadata=metadata)
        key = keygen.next()
        while key in file_handle:
            key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    """Main import routine.
    """
    np_files, lab_files = pair_data_and_labels(
        args.data_directory, args.lab_directory)
    add_to_datasequence_file(
        np_files, lab_files, args.output_file, args.framerate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import CQT arrays into a DataSequenceFile for training.")

    parser.add_argument("data_directory",
                        metavar="data_directory", type=str,
                        help="Base path to search for data arrays.")

    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Base path to search for lab files.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the DataSequenceFile.")

    parser.add_argument("framerate",
                        metavar="framerate", type=int,
                        help="Framerate of the data arrays.")

    main(parser.parse_args())
