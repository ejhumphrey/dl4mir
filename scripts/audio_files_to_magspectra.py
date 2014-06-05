#!/usr/bin/env python
"""Compute CQTs for a collection of audio files listed in a text file.

Calling this script consumes two files:

 1. file_list: A text file of newline separated filepaths to audio.
 2. cqt_params: A JSON-encoded text file of parameters for the CQT.

The former can be compiled by any means. The JSON encoded text can be created
by copy-pasting the following and writing the result to a file.

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

This script writes the output files under the same directory as the inputs,
with a '-cqt.npy' post-fix added to base file path. For example...

  "/some/audio/file.mp3" maps to "/some/audio/file-cqt.npy"

Sample Call:
bash$ python marl/scripts/audio_files_to_cqt_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt
"""

import argparse
from multiprocessing import Pool
import os
import time

import numpy as np

from marl.audio.fileio import AudioReader
from ejhumphrey.datasets.utils import filebase, expand_filebase

NUM_CPUS = None  # Use None for system max.

# Global dict
global_params = dict()


class Pair(object):
    def __init__(self, first, second):
        self.first, self.second = first, second


def audio_file_to_npy_file(filepath, tag=""):
    """Post-fix a tag to a filebase, preserving the file extension.

    Example:
    >>> print audio_file_to_npy_file('/path/to/some/file.txt', 'mytag')
      /path/to/some/file-mytag.txt
    """
    base_file = os.path.splitext(filepath)[0]
    ext = ".npy"
    if tag:
        ext = "-%s" % tag + ext
    return base_file + ext


def audio_file_to_stft(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """

    reader = AudioReader(filepath=file_pair.first,
                         samplerate=global_params.get("samplerate"),
                         framesize=global_params.get("framesize"),
                         framerate=global_params.get("framerate"),
                         channels=global_params.get("channels"))
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.save(file_pair.second, stft(reader).squeeze())


def stft(reader):
    """Compute the STFT of an initialized FramedReader.
    """
    win = np.hanning(reader.framesize())[:, np.newaxis]
    return np.array([np.abs(np.fft.rfft(win * frame,
                                        axis=0)) for frame in reader])


def iofiles(file_list, output_dir):
    """Generator for input/output file pairs.

    Parameters
    ----------
    file_list : str
        Path to a human-readable text file of absolute file paths.
    output_dir : str
        Base directory to write outputs under the same file base.

    Yields
    ------
    file_pair : Pair of strings
        first=input_file, second=output_file
    """
    for line in open(file_list, "r"):
        input_file = line.strip("\n")
        output_file = expand_filebase(filebase(input_file), output_dir, ".npy")
        yield Pair(input_file, output_file)


def create_output_directory(output_directory):
    """
    Returns
    -------
    output_dir : str
        Expanded path, that now certainly exists.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print "[%s] Output Directory: %s" % (time.asctime(), output_directory)
    return output_directory


def main(args):
    """Main routine for staging parallelization."""

    output_dir = create_output_directory(args.output_directory)
    global_params['framerate'] = args.framerate
    global_params['framesize'] = args.framesize
    global_params['samplerate'] = args.samplerate
    global_params['channels'] = args.channels

    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=audio_file_to_stft,
                   iterable=iofiles(args.file_list,
                                    output_dir))
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CQT representations for a "
                    "collection of audio files")
    parser.add_argument("file_list",
                        metavar="file_list", type=str,
                        help="A text file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    parser.add_argument("--samplerate",
                        metavar="samplerate", type=float,
                        default=11025.0, action="store",
                        help="Samplerate for analysis.")
    parser.add_argument("--channels",
                        metavar="channels", type=int,
                        default=1, action="store",
                        help="Number of channels.")
    parser.add_argument("--framerate",
                        metavar="framerate", type=float,
                        default=10.0, action="store",
                        help="Number of analysis frames per second.")
    parser.add_argument("--framesize", action="store",
                        metavar="framesize", type=int,
                        default=8192,
                        help="Number of samples per frame.")
    main(parser.parse_args())
