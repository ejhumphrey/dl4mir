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
$ python audio_files_to_cqt_arrays.py \
rwc_filelist.txt \
cqt_arrays \
--cqt_params=params.json \
--num_cpus=2
"""
from __future__ import print_function

import argparse
from joblib import delayed
from joblib import Parallel
import json
import numpy as np
import time

from dl4mir.common.cqt import cqt
import dl4mir.common.fileutil as futil

EXT = ".npz"
DEFAULT_PARAMS = dict(
    filepath=None, q=1.0, freq_min=27.5, octaves=7, bins_per_octave=36,
    samplerate=11025.0, channels=1, bytedepth=2, framerate=20.0,
    overlap=None, stride=None, time_points=None, alignment='center',
    offset=0)


def audio_file_to_cqt(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        The input_file (first) and output file (second) tuple.

    Returns
    -------
    status : bool
        True on completion.
    """
    kwargs = dict(**DEFAULT_PARAMS)
    kwargs.update(filepath=file_pair.first)
    time_points, cqt_spectra = cqt(**kwargs)
    np.savez(file_pair.second, time_points=time_points, cqt=cqt_spectra)
    print("[{0}] Finished: {1}".format(time.asctime(), file_pair.first))
    return True


def main(textlist, output_directory, cqt_params=None, num_cpus=-1):
    if cqt_params:
        DEFAULT_PARAMS.update(json.load(open(cqt_params)))

    output_dir = futil.create_directory(output_directory)
    pool = Parallel(n_jobs=num_cpus)
    dcqt = delayed(audio_file_to_cqt)
    iterargs = futil.map_path_file_to_dir(textlist, output_dir, EXT)
    return pool(dcqt(x) for x in iterargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("textlist",
                        metavar="textlist", type=str,
                        help="A text file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    parser.add_argument("--cqt_params",
                        metavar="cqt_params", type=str,
                        default='',
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--num_cpus", type=int,
                        metavar="num_cpus", default=-1,
                        help="Number of CPUs over which to parallelize "
                             "computations.")

    args = parser.parse_args()
    main(args.textlist, args.output_directory,
         args.cqt_params, args.num_cpus)
