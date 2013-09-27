'''
Created on Sep 25, 2013

@author: ejhumphrey
'''

import numpy as np

from ejhumphrey.datasets import chordutils
from marl.hewey.core import DataSequence
from marl.hewey.file import DataSequenceFile
from marl.hewey.keyutils import uniform_keygen

import json
import os
import time

cqt_param_file = "/Volumes/Audio/Chord_Recognition/cqt_params.txt"
track_filelist = "/Volumes/Audio/Chord_Recognition/labeled_tracks_20130926.txt"
hewey_file = '/Volumes/speedy/chordrec.dsf'

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

def align_cqt_and_labels(cqt_file, lab_file, cqt_params):
    cqt = np.load(cqt_file)
    boundaries, labels = chordutils.load_labfile(lab_file)
    time_points = np.arange(len(cqt), dtype=float) / cqt_params.get("framerate")
    timed_labels = chordutils.assign_labels_to_time_points(time_points,
                                                           boundaries,
                                                           labels)
    return cqt, timed_labels

def create_datasequence_file(tracklist, filename, cqt_params):
    file_handle = DataSequenceFile(filename)
    keygen = uniform_keygen(2)
    for i, tuples in enumerate(collect_track_tuples(tracklist)):
        audio_file, cqt_file, label_file = tuples
        print "%03d: Importing %s" % (i, audio_file)
        X, y = align_cqt_and_labels(cqt_file, label_file, cqt_params)
        metadata = {"timestamp": time.asctime(),
                    "filesource": audio_file}
        dseq = DataSequence(value=X, label=y, metadata=metadata)
        key = keygen.next()
        file_handle.write(key, dseq)

    file_handle.create_tables()


def main():
    cqt_params = json.load(open(cqt_param_file))
    create_datasequence_file(track_filelist, hewey_file, cqt_params)


if __name__ == '__main__':
    main()
