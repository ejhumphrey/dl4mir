"""Extract beat / timing data from a collection of audio."""

import argparse
import json
import librosa
import marl
from . import fileutil as F
from multiprocessing import Pool
import numpy as np
import time

NUM_CPUS = 12  # Use None for system max.
EXT = ".json"


def extract_timing_data(filename, samplerate=22050, channels=1, hop_length=64):
    x_n, fs = marl.audio.read(filename, samplerate, channels)
    onset_env = librosa.onset.onset_strength(
        x_n.squeeze(), fs, hop_length=hop_length, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=fs, hop_length=hop_length)
    beat_times = librosa.frames_to_time(
        beat_frames, sr=fs, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=fs, hop_length=hop_length)
    onset_times = librosa.frames_to_time(
        onset_frames, sr=fs, hop_length=hop_length)
    duration = len(x_n) / fs
    return dict(onset_times=onset_times.tolist(),
                beat_times=beat_times.tolist(),
                tempo=tempo,
                duration=duration)


def process_one(file_pair):
    timing_data = extract_timing_data(file_pair.first)
    with open(file_pair.second, 'w') as fp:
        json.dump(timing_data, fp, indent=2)
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)


def main(args):
    """Main routine for staging parallelization."""
    output_dir = F.create_directory(args.output_directory)
    pairs = F.map_path_file_to_dir(args.textlist_file, output_dir, EXT)
    if NUM_CPUS == 1:
        for fp in pairs:
            process_one(fp)
    else:
        pool = Pool(processes=NUM_CPUS)
        pool.map_async(func=process_one, iterable=pairs)
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beat-track a collection of audio.")
    parser.add_argument("textlist_file",
                        metavar="textlist_file", type=str,
                        help="A text file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save JSON output.")
    main(parser.parse_args())
