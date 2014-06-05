#!/usr/bin/env python
"""Apply LCN to a collection of numpy arrays.

Sample Call:
ipython ejhumphrey/scripts/apply_lcn_to_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt
"""

import argparse
from multiprocessing import Pool
import numpy as np
import os
from scipy.signal import medfilt
import time

NUM_CPUS = None  # Use None for system max.
FILTER_LEN = 'filter_len'
DECIMATE = 'decimate'

# Global dict
PARAMS = dict()


class Pair(object):
    def __init__(self, first, second):
        self.first, self.second = first, second


def apply_medfilt(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output_file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    L, M = PARAMS.get(FILTER_LEN), PARAMS.get(DECIMATE)
    data = medfilt(np.load(file_pair.first), [L, 1])
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.save(file_pair.second, data[::M])


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
        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
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
    PARAMS[FILTER_LEN] = args.filter_len
    PARAMS[DECIMATE] = args.decimate

    output_dir = create_output_directory(args.output_directory)
    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=apply_medfilt,
                   iterable=iofiles(args.file_list, output_dir))
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
    parser.add_argument("--filter_len",
                        metavar="filter_len", type=int,
                        default=51,
                        help="Length of the median filter (must be odd).")

    parser.add_argument("--decimate",
                        metavar="decimate", type=int,
                        default=20,
                        help="Decimation factor for the TFR.")
    main(parser.parse_args())
