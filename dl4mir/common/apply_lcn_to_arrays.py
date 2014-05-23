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
import marl.fileutils as futils
import time

NUM_CPUS = None  # Use None for system max.

# Global dict
transform = dict()


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
    S = np.sqrt(
        convolve2d(np.power(V, 2.0), kernel, mode='same', boundary='symm'))
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
    data = np.load(file_pair.first)
    if data.ndim == 2:
        Z = lcn(data, transform["kernel"])
    elif data.ndim == 3:
        Z = np.array([lcn(data[:, :, n], transform["kernel"])
                      for n in range(data.shape[-1])])
    else:
        raise ValueError("No idea what to do with a %d-dim array." % data.ndim)
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.save(file_pair.second, Z)


def main(args):
    """Main routine for staging parallelization."""
    create_kernel(args.dim0, args.dim1)
    output_dir = futils.create_directory(args.output_directory)
    with open(os.path.join(output_dir, "lcn_params.json"), "w") as fp:
        json.dump({"dim0": args.dim0, "dim1": args.dim1}, fp, indent=2)

    pool = Pool(processes=NUM_CPUS)
    pool.map_async(
        func=apply_lcn,
        iterable=futils.map_path_file_to_dir(
            args.textlist_file, output_dir, '.npy'))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CQT representations for a "
                    "collection of audio files")
    parser.add_argument("textlist_file",
                        metavar="textlist_file", type=str,
                        help="A textlist file with of audio filepaths.")
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
