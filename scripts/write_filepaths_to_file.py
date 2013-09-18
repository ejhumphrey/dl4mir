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

This script writes the output files under the same directory as the inputs,
with a '-cqt.npy' post-fix added to base file path. For example...

  "/some/audio/file.mp3" maps to "/some/audio/file-cqt.npy"
"""

import argparse
import json
from multiprocessing import Pool
import os
import time

import numpy as np

from marl.audio.timefreq import AudioReaderCQT
from marl.audio.fileio import AudioReader

NUM_CPUS = None  # Use None for system max.
CQT_TAG = "cqt"

# Global dict
transform = dict()

class Pair(object):
    def __init__(self, first, second):
        self.first, self.second = first, second

def create_cqt(param_file):
    params = json.loads(open(param_file))
    transform["params"] = params
    transform["cqt"] = AudioReaderCQT(
        q=params.get("q"), freq_min=params.get("freq_min"),
        octaves=params.get("octaves"), samplerate=params.get("samplerate"),
        bins_per_octave=params.get("bins_per_octave"))

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

def audio_file_to_cqt(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    cqt = transform["cqt"]
    params = transform["params"]
    reader = AudioReader(filepath=file_pair.first,
                        samplerate=cqt.samplerate(),
                        framesize=cqt.framesize(),
                        framerate=params.get("framerate"),
                        alignment=params.get("alignment"),
                        channels=params.get("channels"))
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.save(file_pair.second, cqt(reader).squeeze())


def iofiles(file_list):
    """Generator for input/output file pairs.

    Parameters
    ----------
    file_list : str
        Path to a human-readable text file of absolute audio file paths.

    Yields
    ------
    file_pair : Pair of strings
        first=input_file, second=output_file
    """
    for line in open(file_list, "r"):
        audio_filepath = line.strip("\n")
        npy_filepath = audio_file_to_npy_file(audio_filepath, tag=CQT_TAG)
        yield Pair(audio_filepath, npy_filepath)


def main(args):
    """Main routine for staging parallelization."""
    pool = Pool(processes=NUM_CPUS)
    if args.chunksize:
        pool.map_async(func=audio_file_to_cqt,
                       iterable=iofiles(args.file_list),
                       chunksize=args.chunksize)
    else:
        pool.map_async(func=audio_file_to_cqt,
                       iterable=iofiles(args.file_list))
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CQT representations for a "
                    "collection of audio files")
    parser.add_argument("--file_list",
                        metavar="F", type=str,
                        help="A text file with a list of audio filepaths.")
    parser.add_argument("--cqt_params",
                        metavar="P", type=str,
                        help="A JSON text file with parameters for the CQT.")
    parser.add_argument("--chunksize",
                        metavar="S", type=int,
                        default=0,
                        help="Chunksize for map_async.")
    main(parser.parse_args())
