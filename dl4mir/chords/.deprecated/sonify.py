import numpy as np
import mir_eval
import marl.audio.sox as sox
import marl.audio.utils as util

import scipy.io.wavfile as wavfile

import dl4mir.chords.labels as L
from dl4mir.chords.output_labels import posterior_to_labeled_intervals


def sample_index(stash, index, num_per_track):
    pairs = []
    for key in index:
        idx = np.array(index[key])
        np.random.shuffle(idx)
        for i in idx[:num_per_track]:
            pairs.append([key, i / 20.0])

    return pairs


def concatenate_clips(input_files, start_times, end_times, output_file):
    temp_files = []
    for input_file, start, end in zip(input_files, start_times, end_times):
        temp_files.append(util.temp_file("wav"))
        sox.trim(input_file, temp_files[-1], start, end)

    sox.concatenate(temp_files, output_file)


def lab_to_wav(label_file, wav_file, samplerate=44100.0):
    intervals, labels = L.load_labeled_intervals(label_file)
    x_n = mir_eval.sonify.chords(labels, np.asarray(intervals), samplerate)
    x_n *= 2**15
    wavfile.write(wav_file, samplerate, x_n.astype(np.int16))


def posterior_to_wav(posterior, framerate, wav_file, viterbi_penalty=0.0,
                     medfilt=0, samplerate=44100.0):
    intervals, labels = posterior_to_labeled_intervals(
        posterior, framerate=framerate, viterbi_penalty=viterbi_penalty,
        medfilt=medfilt)
    x_n = mir_eval.sonify.chords(labels, np.asarray(intervals), samplerate)
    x_n *= 2**15
    wavfile.write(wav_file, samplerate, x_n.astype(np.int16))
