#!/usr/bin/env python
"""Compute CQTs for a collection of audio files listed in a text file.

Calling this script consumes two files:

 1. textlist_file: A text file of newline separated filepaths to audio.
 2. cqt_params: A JSON-encoded text file of parameters for the CQT.

The former can be compiled by any means. The JSON encoded text can be created
by copy-pasting the following and writing the result to a file.

import json
params = {"q": 0.75,
          "freq_min": 27.5,
          "octaves": 8,
          "samplerate": 16000.0,
          "bins_per_octave": 24,
          "framerate": 20,
          "alignment": 'center',
          "channels": 1}
print json.dumps(params, indent=2)
fh = open("cqt_params.txt", "w")
json.dump(params, fh, indent=2)
fh.close()

This script writes the output files under the given output directory:

  "/some/audio/file.mp3" maps to "${output_dir}/file.npz"

Sample Call:
$ python marl/scripts/audio_files_to_cqt_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt \
/Volumes/Audio/cqt_files
"""

import argparse
import librosa
import time

from multiprocessing import Pool
import marl
from marl import fileutils as F

NUM_CPUS = 12  # Use None for system max.
SAMPLERATE = 22050
EXT = ".mp3"


def run_hpss(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    x_n, fs = marl.audio.read(file_pair.first, SAMPLERATE, channels=1)
    y_h = librosa.effects.harmonic(x_n.ravel())
    marl.audio.write(file_pair.second, y_h, fs)
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)


def main(args):
    """Main routine for staging parallelization."""

    pool = Pool(processes=NUM_CPUS)
    output_dir = F.create_directory(args.output_directory)
    pool.map_async(
        func=run_hpss,
        iterable=F.map_path_file_to_dir(args.textlist_file, output_dir, EXT))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CQT representations for a "
                    "collection of audio files")
    parser.add_argument("textlist_file",
                        metavar="textlist_file", type=str,
                        help="A text file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
