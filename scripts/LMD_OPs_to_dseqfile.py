"""Import a collection of CQT arrays into a Hewey DataSequenceFile.

Sample Call:
$ ipython ejhumphrey/scripts/load_chords_into_hewey.py \
/Volumes/Audio/Chord_Recognition/labeled_tracks_20130926_train0.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt \
/Volumes/speedy/chordrec_20130930_train0.dsf
"""

import argparse
import numpy as np
import os
import time

from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen
import cPickle
from scipy.io.matlab.mio import loadmat


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

def matfile_to_datasequence(mat_file, stdev_params):
    """
    Parameters
    ----------
    mat_file : strings
        Path to a matfile.
    stdev_params : np.ndarray
        The first dimension is mean, the second standard deviation.
    """
    data = loadmat(mat_file)
    op_matrix = data.get("features")
    shp = op_matrix.shape
    op_matrix = np.reshape(op_matrix, newshape=(np.prod(shp[:2]), shp[-1])).T
    op_matrix = (op_matrix - stdev_params[0]) / stdev_params[1]
    labels = [str(data.get('genre')[0])] * len(op_matrix)
    metadata = {"timestamp": time.asctime(),
                "filesource": mat_file}
    return DataSequence(value=op_matrix, label=labels, metadata=metadata)

def create_datasequence_file(mat_files, filename, stdev_params):
    """
    Parameters
    ----------
    mat_files : list of strings
        Paths to matfiles.
    filename : string
        Output name for the DataSequenceFile
    stdev_params : np.ndarray
        The first dimension is mean, the second standard deviation.
    """
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)
    for i, mat_file in enumerate(mat_files):
        print "%03d: Importing %s" % (i, mat_file)
        dseq = matfile_to_datasequence(mat_file, stdev_params)
        key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main(args):
    stdev_params = cPickle.load(open(args.stdev_file))
    mat_files = [l.strip('/n') for l in open(args.filelist)]
    create_datasequence_file(mat_files, args.output_file, stdev_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Push CQT files through a trained deepnet.")

    parser.add_argument("filelist",
                        metavar="filelist", type=str,
                        help="Text file list of matfiles to import.")

    parser.add_argument("stdev_file",
                        metavar="stdev_file", type=str,
                        help="Parameters used to compute the CQT.")

    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filepath to write the output DataSequenceFile.")

    main(parser.parse_args())
