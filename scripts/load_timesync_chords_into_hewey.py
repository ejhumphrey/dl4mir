"""Import a collection of CQT arrays into a Hewey DataSequenceFile.

Sample Call:
$ ipython ejhumphrey/scripts/load_chords_into_hewey.py \
/Volumes/Audio/Chord_Recognition/labeled_tracks_20130930_train0.txt \
/Volumes/Audio/Chord_Recognition/cqts/20131002 \
/Volumes/speedy/chordrec_lcn_train0_20131002.dsf
"""

import argparse
import numpy as np
import os
import time

from ejhumphrey.datasets import chordutils, utils, onsets
from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen


def pair_triples(split_file, cqt_dir, lab_dir, timepoint_dir):
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

    cqts, labs, tpoints = [], [], []
    for line in open(split_file):
        fbase = line.strip("\n")
        cqt_file = os.path.join(cqt_dir, "%s.npy" % fbase)
        assert os.path.exists(cqt_file)
        cqts.append(cqt_file)

        lab_file = os.path.join(lab_dir, "%s.lab" % fbase)
        assert os.path.exists(lab_file)
        labs.append(lab_file)

        timepoint_file = os.path.join(timepoint_dir, "%s.txt" % fbase)
        assert os.path.exists(timepoint_file)
        tpoints.append(timepoint_file)

    return cqts, labs, tpoints


def add_to_datasequence_file(cqt_files, lab_files, timepoint_files,
                             framerate, pool_mode, output_filename):
    """
    Parameters
    ----------

    filename : string
        Output name for the DataSequenceFile
    framerate : scalar
        Framerate of the input array.
    pool_mode : string
        One of ['mean', 'median', 'max']
    """
    file_handle = DataSequenceFile(output_filename)
    keygen = uniform_keygen(2)

    for cqt_file, lab_file, timepoint_file in zip(cqt_files,
                                                  lab_files,
                                                  timepoint_files):
        print "[%s] Importing %s" % (time.asctime(), cqt_file)
        cqt_array = np.load(cqt_file)
        labels = chordutils.align_lab_file_to_array(
            cqt_array, lab_file, framerate)
        timepoints = onsets.load_timepoint_file(timepoint_file)
        boundary_bins = np.round(timepoints*framerate).astype(int)
        metadata = {"timestamp": time.asctime(),
                    "filesource": cqt_file}
        if pool_mode == "mean":
            pool_fx = chordutils.mean_pool
        elif pool_mode == "median":
            pool_fx = chordutils.median_pool
        else:
            raise ValueError("Unsupported pool_mode '%s'"&pool_mode)

        pooled_cqt = pool_fx(cqt_array, boundary_bins)
        pooled_labels = chordutils.majority_pool(labels, boundary_bins, False)
        dseq = DataSequence(value=pooled_cqt,
                            label=pooled_labels,
                            metadata=metadata)
        key = keygen.next()
        while key in file_handle:
            key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    cqt_files, lab_files, timepoint_files = pair_triples(
        args.split_file, args.cqt_directory,
        args.lab_directory, args.timepoint_directory)
    add_to_datasequence_file(cqt_files, lab_files, timepoint_files,
        args.framerate, args.pool_mode, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import CQT arrays into a DataSequenceFile for training.")

    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="List of filebases to import; probably a training "
                        "or validation split.")

    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Base path to search for CQT arrays. Must also "
                        "contain a cqt_params.txt file.")

    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Base path to search for lab files.")

    parser.add_argument("timepoint_directory",
                        metavar="timepoint_directory", type=str,
                        help="Base path to search for timepoint files.")

    parser.add_argument("framerate",
                        metavar="framerate", type=float,
                        help="Framerate of cqt arrays.")

    parser.add_argument("--pool_mode",
                        action='store',
                        default="median", dest='pool_mode',
                        help="Output layer of the model")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the DataSequenceFile.")

    main(parser.parse_args())
