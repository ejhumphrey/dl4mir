"""Write me."""

import numpy as np
import marl.chords.labels as CL
import os


filefmt = "/Volumes/Audio/Chord_Recognition/stfts/%s.npy"
labfmt = "/Volumes/Audio/Chord_Recognition/labs/%s.lab"

train_set = [os.path.split(f.strip("\n"))[-1][:-4]
             for f in open("/Users/ejhumphrey/dltut/chord_train_filelist.txt")]
test_set = [os.path.split(f.strip("\n"))[-1][:-4]
            for f in open("/Users/ejhumphrey/dltut/chord_test_filelist.txt")]

frames_per_file = 250
framesize = 4097

train_data = np.zeros([len(train_set)*frames_per_file, framesize],
                      dtype=np.float32)
train_labels = []
count = 0
for name in train_set:
    boundaries, labels = CL.read_lab_file(labfmt % name)
    dft_spec = np.load(filefmt % name)
    num_frames = len(dft_spec)
    time_points = np.arange(num_frames) / 10.0
    labels = CL.interpolate_labels(time_points, boundaries, labels)
    frame_idx = np.random.permutation(num_frames)[:frames_per_file]
    train_data[count:count + len(frame_idx)] = dft_spec[frame_idx]
    count += len(frame_idx)
    train_labels.extend([labels[n] for n in frame_idx])
    print name

test_data = np.zeros([len(test_set)*frames_per_file, framesize],
                     dtype=np.float32)
test_labels = []
count = 0
for name in test_set:
    boundaries, labels = CL.read_lab_file(labfmt % name)
    dft_spec = np.load(filefmt % name)
    num_frames = len(dft_spec)
    time_points = np.arange(num_frames) / 10.0
    labels = CL.interpolate_labels(time_points, boundaries, labels)
    frame_idx = np.random.permutation(num_frames)[:frames_per_file]
    test_data[count:count + len(frame_idx)] = dft_spec[frame_idx]
    count += len(frame_idx)
    test_labels.extend([labels[n] for n in frame_idx])
    print name
