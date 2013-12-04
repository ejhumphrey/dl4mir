#!/usr/bin/env python
"""Apply LCN to a collection of numpy arrays.

Sample Call:
ipython ejhumphrey/scripts/apply_lcn_to_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt
"""

import argparse
import json
from multiprocessing import Pool
import numpy as np
import os
from scipy.signal.signaltools import convolve2d
from scipy.signal.windows import gaussian
import time

NUM_CPUS = None  # Use None for system max.

# Global dict
transform = dict()

class Pair(object):
    def __init__(self, first, second):
        self.first, self.second = first, second


def lcn(X, kernel):
    """Apply Local Contrast Normalization (LCN) to an array.

    Parameters
    ----------
    X : np.ndarray
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = X - Xh
    S = np.sqrt(convolve2d(np.power(V, 2.0),
                           kernel,
                           mode='same',
                           boundary='symm'))
    S2 = np.zeros(S.shape) + S.mean()
    S2[S > S.mean()] = S[S > S.mean()]
    if S2.sum() == 0.0:
        S2 += 1.0
    return V / S2

def create_kernel(dim0, dim1):
    """
    Parameters
    ----------
    dim0 : int
    dim1 : int

    Returns
    -------
    kernel : np.ndarray
    """
    dim0_weights = np.hamming(dim0 * 2 + 1)[:dim0]
    dim1_weights = gaussian(dim1, dim1 * 0.25, True)
    kernel = dim0_weights[:, np.newaxis] * dim1_weights[np.newaxis, :]
    transform["kernel"] = kernel / kernel.sum()


def apply_lcn(file_pair):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """

    Z = lcn(np.load(file_pair.first), transform.get("kernel"))
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.save(file_pair.second, Z)


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
    create_kernel(args.dim0, args.dim1)
    output_dir = create_output_directory(args.output_directory)
    param_file = open(os.path.join(output_dir, "lcn_params.txt"), "w")
    json.dump({"dim0":args.dim0, "dim1":args.dim1}, param_file, indent=2)
    param_file.close()

    pool = Pool(processes=NUM_CPUS)
    pool.map_async(func=apply_lcn,
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
    parser.add_argument("dim0",
                        metavar="dim0", type=int,
                        help="First dimension of the array.")
    parser.add_argument("dim1",
                        metavar="dim1", type=int,
                        help="Second dimension of the array.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
