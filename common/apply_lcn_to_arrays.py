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

import dl4mir.common.util as util


NUM_CPUS = None  # Use None for system max.

# Global dict
transform = dict()
EXT = ".npz"


def lcn(X, kernel):
    """Apply Local Contrast Normalization (LCN) to an array.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = X - Xh
    S = np.sqrt(convolve2d(np.power(V, 2.0),
                kernel, mode='same', boundary='symm'))
    S2 = np.zeros(S.shape) + S.mean()
    S2[S > S.mean()] = S[S > S.mean()]
    if S2.sum() == 0.0:
        S2 += 1.0
    return V / S2


def lcn2(X, kernel, mean_scalar=1.0):
    """Apply Local Contrast Normalization (LCN) to an array.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = X - Xh
    S = np.sqrt(convolve2d(np.power(V, 2.0),
                kernel, mode='same', boundary='symm'))
    thresh = np.exp(np.log(S + np.power(2.0, -5)).mean(axis=-1))
    S = S*np.greater(S - thresh.reshape(-1, 1), 0)
    S += 1.0*np.equal(S, 0.0)
    return V / S


def mm_lcn(X, kernel=None, rho=0):
    if kernel is None:
        dim0, dim1 = 15, 37
        dim0_weights = np.hamming(dim0 * 2 + 1)[:dim0]
        dim1_weights = np.hamming(dim1)
        kernel = dim0_weights[:, np.newaxis] * dim1_weights[np.newaxis, :]

    kernel /= kernel.sum()
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = util.hwr(X - Xh)
    S = np.sqrt(
        convolve2d(np.power(V, 2.0), kernel, mode='same', boundary='symm'))
    S2 = np.zeros(S.shape) + S.mean()
    S2[S > S.mean()] = S[S > S.mean()]
    if S2.sum() == 0.0:
        S2 += 1.0
    return V / S2**rho


def highpass(X, kernel):
    """

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    return X - Xh


def local_l2norm(X, kernel):
    local_mag = np.sqrt(convolve2d(np.power(X, 2.0),
                        kernel, mode='same', boundary='symm'))
    local_mag = local_mag + 1.0*(local_mag == 0.0)
    return X / local_mag


def lcn3(X, kernel):
    x_hp = highpass(X, kernel)
    x_73 = local_l2norm(x_hp, np.hanning(73).reshape(1, -1))
    x_37 = local_l2norm(x_hp, np.hanning(37).reshape(1, -1))
    x_19 = local_l2norm(x_hp, np.hanning(19).reshape(1, -1))
    # x_11 = local_l2norm(x_hp, np.hanning(11).reshape(1, -1))
    x_multi = np.array([x_73, x_37, x_19]).transpose(1, 2, 0)
    w = create_mask()**2.0
    return (x_multi * w).sum(axis=-1)


def create_mask():
    w = np.sin(np.pi*np.arange(36)/36.)
    w_73 = np.zeros(252)
    w_37 = np.zeros(252)
    w_19 = np.zeros(252)
    # w_11 = np.zeros(252)

    w_73[:18] = 1.0
    w_73[18:36] = w[18:]

    w_37[18:36] = w[:18]
    w_37[36:72] = 1.0
    w_37[72:90] = w[18:]

    w_19[72:90] = w[:18]
    w_19[90:] = 1.0
    # w_19[90:162] = 1.0
    # w_19[162:180] = w[18:]

    # w_11[162:180] = w[:18]
    # w_11[180:] = 1.0
    return np.array([w_73, w_37, w_19]).T.reshape(1, 252, 3)


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


def apply_lcn(file_pair, key='cqt'):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    data = dict(**np.load(file_pair.first))
    if data[key].ndim == 2:
        data[key] = lcn3(data[key], transform["kernel"])
    elif data[key].ndim == 3:
        data[key] = np.array([lcn3(x, transform["kernel"]) for x in data[key]])
    else:
        raise ValueError("Cannot transform a %d-dim array." % data[key].ndim)
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    np.savez(file_pair.second, **data)


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
            args.textlist_file, output_dir, EXT))
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
