'''
Created on Oct 29, 2013

@author: ejhumphrey
'''

from marl.audio.fileio import AudioReader
from marl.audio.timefreq import AudioReaderCQT
import numpy as np
from matplotlib.pyplot import figure, show
from ejhumphrey.datasets import chordutils
from scipy.fftpack.realtransforms import dct


#filename = "/Volumes/Audio/Chord_Recognition/uspop/sublime/Sublime/06-Santeria.mp3"
filename = "/Users/ejhumphrey/dltut/santeria_30sec.mp3"

def show_spectrogram(filename):
    reader = AudioReader(
        filename, framerate=20.0, samplerate=44100, framesize=2048, channels=1)

    w_n = np.hanning(2048)
    w_n /= 2 * w_n.sum()

    X = np.array([np.abs(np.fft.rfft(w_n * frame.flatten())) for frame in reader])

    Xs = np.fliplr(X[:int(30 * reader.framerate()), :])

    fig = figure()
    ax = fig.gca()
    ax.imshow(np.log1p(1000 * Xs).T, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Time (seconds)")
    frame_idx = np.arange(len(Xs))[::int(reader.framerate() * 10)]
    time_idx = frame_idx / reader.framerate()
    ax.set_xticks(frame_idx)
    ax.set_xticklabels(time_idx)
    ax.set_ylabel("Frequency (Hz)")
    bin_idx = np.arange(Xs.shape[1])[::Xs.shape[1] / 8]
    freq_idx = bin_idx * reader.samplerate() / reader.framesize()
    ax.set_yticks(bin_idx)
    ax.set_yticklabels(freq_idx[::-1])
    show()


def chroma_matrix(starting_root, num_octaves, bins_per_octave):
    N = num_octaves * bins_per_octave
    matrix = np.zeros([N, 12])
    pitch_step = bins_per_octave / 12
    idx = np.arange(N / pitch_step, dtype=int) * pitch_step
    matrix[idx, (starting_root + idx / pitch_step) % 12] = 1.0
    return matrix

def chromagram(filename):
    cqt = AudioReaderCQT(freq_min=27.5, octaves=8)
    reader = AudioReader(filename, framerate=20.0, samplerate=cqt.samplerate(),
                         channels=1, framesize=cqt.framesize())
    X = cqt(reader).squeeze()

    Xs = X[:int(30 * reader.framerate()), :]

    fig = figure()
    ax = fig.gca()
    ax.imshow(np.fliplr(np.log1p(1 * Xs)).T, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Time (seconds)")
    frame_idx = np.arange(len(Xs))[::int(reader.framerate() * 10)]
    time_idx = frame_idx / reader.framerate()
    ax.set_xticks(frame_idx)
    ax.set_xticklabels(time_idx)
    ax.set_ylabel("Pitch")

    ax.set_yticks([])
    ax.set_yticklabels([])
    show()


    cmatrix = chroma_matrix(9, num_octaves=8, bins_per_octave=12)
    cmatrix /= cmatrix.sum(axis=0)[np.newaxis, :]
    print Xs.shape, cmatrix.shape
    C = np.dot(Xs, cmatrix)

    fig = figure()
    ax = fig.gca()
    ax.imshow(C.T, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Time (seconds)")
    frame_idx = np.arange(len(C))[::int(reader.framerate() * 10)]
    time_idx = frame_idx / reader.framerate()
    ax.set_xticks(frame_idx)
    ax.set_xticklabels(time_idx)
    ax.set_ylabel("Pitch Class")

    ax.set_yticks(range(12))
    ax.set_yticklabels(chordutils.pitch_classes)
    show()

    fig = figure()
    ax = fig.gca()
    ax.imshow(np.fliplr(dct(np.log1p(1 * Xs), axis=1)[:, 1:13]).T, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Time (seconds)")
    frame_idx = np.arange(len(C))[::int(reader.framerate() * 10)]
    time_idx = frame_idx / reader.framerate()
    ax.set_xticks(frame_idx)
    ax.set_xticklabels(time_idx)
    ax.set_ylabel("DCT Coefficient")

    ax.set_yticks([])
    ax.set_yticklabels([])
    show()
