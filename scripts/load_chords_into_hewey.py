"""Import a collection of CQT arrays into a Hewey DataSequenceFile.

Sample Call:
$ ipython ejhumphrey/scripts/load_chords_into_hewey.py \
/Volumes/Audio/Chord_Recognition/labeled_tracks_20130926_train0.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt \
/Volumes/speedy/chordrec_20130930_train0.dsf
"""

import argparse
import json
import numpy as np
import os
import time

from ejhumphrey.datasets import chordutils
from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen


def collect_track_tuples(filelist):
    """Compile a list of (audio_file, cqt_file, label_file) tuples.

    It is guaranteed that all files exist on disk.

    Returns
    -------
    results : list
        List of three item tuples; audio, cqt, and label.
    """
    results = []
    for line in open(filelist):
        audio_file = line.strip('\n')
        assert os.path.exists(audio_file), \
            "Audio file does not exist: %s" % audio_file
        cqt_file = os.path.splitext(audio_file)[0] + "-cqt.npy"
        assert os.path.exists(cqt_file), \
            "CQT file does not exist: %s" % cqt_file
        label_file = os.path.splitext(audio_file)[0] + ".lab"
        assert os.path.exists(label_file), \
            "Label file does not exist: %s" % label_file
        results.append((audio_file, cqt_file, label_file))

    return results


def create_datasequence_file(tracklist, filename, cqt_params):
    """
    Parameters
    ----------
    tracklist : string
        Path to a text file of absolute file paths.
    filename : string
        Output name for the DataSequenceFile
    cqt_params : dict
        Parameters used to compute the CQT.
    """
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)
    for i, tuples in enumerate(collect_track_tuples(tracklist)):
        audio_file, cqt_file, label_file = tuples
        print "%03d: Importing %s" % (i, audio_file)
        cqt_array = np.load(cqt_file)
        labels = chordutils.align_lab_file_to_array(
            cqt_array, label_file, cqt_params.get("framerate"))
        metadata = {"timestamp": time.asctime(),
                    "filesource": audio_file}
        dseq = DataSequence(value=cqt_array, label=labels, metadata=metadata)
        key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    cqt_params = json.load(open(args.cqt_param_file))
    create_datasequence_file(args.track_filelist, args.output_file, cqt_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("track_filelist",
                        metavar="track_filelist", type=str,
                        help="File list of CQT arrays to import.")

    parser.add_argument("cqt_param_file",
                        metavar="cqt_param_file", type=str,
                        help="Parameters used to compute the CQT.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the output Hewey filesystem.")

    main(parser.parse_args())
