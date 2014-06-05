"""Import a collection of CQT arrays into a Shufflr SequenceFile.

1. Consume a split file (where each line is simply "01_-_No_Reply\n")
2. For each filebase:
    a. Associate data with annotations (data and label directories)
    b. Load data, filter, downsample
    c. Align labels to data
    d. Create Sequence
    e. Add to file

Sample Call:
$ ipython ejhumphrey/scripts/load_chords_into_shufflr.py \
/Volumes/Audio/Chord_Recognition/labeled_tracks_20130930_train0.txt \
/Volumes/Audio/Chord_Recognition/cqts/20140102 \
/Volumes/speedy/chordrec_train0_20140102.shf
"""

import argparse
import numpy as np
import time

from marl.chords import labels
from marl import fileutils

from ejhumphrey.shufflr import core
from ejhumphrey.shufflr import sources
from ejhumphrey.shufflr import keyutils


def load_sequence(fbase, data_dir, lab_dir, framerate):
    data = np.load(fileutils.expand_filebase(fbase, data_dir, "npy"))
    time_points = np.arange(len(data)) / float(framerate)
    boundaries, chord_labels = labels.read_lab_file(
        fileutils.expand_filebase(fbase, lab_dir, "lab"))
    label_seq = labels.interpolate_labels(
        time_points, boundaries, chord_labels)

    return core.Sequence(value=data,
                         labels=[label_seq],
                         metadata={"fbase": fbase,
                                   "timestamp": time.asctime()})


def main(args):
    fhandle = sources.File(args.output_file)
    keygen = keyutils.uniform_keygen(2)
    for key, line in zip(keygen, open(args.split_file)):
        fbase = line.strip("\n")
        data = load_sequence(
            fbase, args.data_dir, args.lab_dir, args.framerate)
        fhandle.add(key, data)
        print "[%s] Finished: %s" % (time.asctime(), fbase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import CQT arrays into a DataSequenceFile for training.")

    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="List of filebases to load.")

    parser.add_argument("data_dir",
                        metavar="data_dir", type=str,
                        help="Base path to search for data arrays. Must also "
                        "contain a params.txt file.")

    parser.add_argument("lab_dir",
                        metavar="lab_directory", type=str,
                        help="Base path to search for lab files.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the DataSequenceFile.")

    parser.add_argument("--framerate",
                        metavar="framerate", type=float,
                        default=10.0,
                        help="Framerate of the input TFR.")

    main(parser.parse_args())
