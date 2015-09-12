#!/usr/bin/env python
"""Apply LCN to a collection of numpy arrays.

Sample Call:
$ python apply_lcn_to_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt
"""
from __future__ import print_function
import argparse
import json
from multiprocessing import Pool
import numpy as np
import os
from scipy.signal.signaltools import convolve2d
from scipy.signal.windows import gaussian
import time

from . import util
from . import fileutil as futil


NUM_CPUS = None  # Use None for system max.

# Global dict
KERNEL = 'kerel'
PARAMS = dict()
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


def lcn_v2(X, kernel, mean_scalar=1.0):
    """Apply an alternative version of local contrast normalization (LCN) to an
    array.

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


def lcn_mauch(X, kernel=None, rho=0):
    """Apply a version of local contrast normalization (LCN), inspired by
    Mauch, Dixon (2009), "Approximate Note Transcription...".

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).
    rho : scalar
        Scalar applied to the final output for heuristic range control.

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """

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
    """Produce a highpass kernel from its lowpass complement.

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
    """Apply local l2-normalization over an input with a given kernel.

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
    local_mag = np.sqrt(convolve2d(np.power(X, 2.0),
                        kernel, mode='same', boundary='symm'))
    local_mag = local_mag + 1.0*(local_mag == 0.0)
    return X / local_mag


def lcn_octaves(X, kernel):
    """Apply octave-varying contrast normalization to an input with a given
    kernel.

    Notes:
    * This is the variant introduced in the LVCE Section of Chapter 5.
    * This approach is painfully heuristic, and tuned for the dimensions used
        in this work (36 bpo, 7 octaves).

    Parameters
    ----------
    X : np.ndarray, ndim=2, shape[1]==252.
        CQT representation, with 36 bins per octave and 252 filters.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.shape[-1] != 252:
        raise ValueError(
            "Apologies, but this method is currently designed for input "
            "representations with a last dimension of 252.")
    x_hp = highpass(X, kernel)
    x_73 = local_l2norm(x_hp, np.hanning(73).reshape(1, -1))
    x_37 = local_l2norm(x_hp, np.hanning(37).reshape(1, -1))
    x_19 = local_l2norm(x_hp, np.hanning(19).reshape(1, -1))
    x_multi = np.array([x_73, x_37, x_19]).transpose(1, 2, 0)
    w = _create_triband_mask()**2.0
    return (x_multi * w).sum(axis=-1)


def _create_triband_mask():
    """Build a summation mask for the octaves defined in Chapter 5.

    The resulting mask tensor looks (roughly) like the following, indexed by
    the final axis:
             __
          0 |  \__      |
          1 |  /  \_____|
          2 |     /     |

    Note: Again, this is admittedly ad hoc, and warrants attention in the
    future.

    Returns
    -------
    mask : np.ndarray, shape=(1, 252, 3)
        Sine-tapered summation mask to smoothly blend three representations
        with logarithmically increasing window widths.
    """
    w = np.sin(np.pi*np.arange(36)/36.)
    w_73 = np.zeros(252)
    w_37 = np.zeros(252)
    w_19 = np.zeros(252)

    w_73[:18] = 1.0
    w_73[18:36] = w[18:]

    w_37[18:36] = w[:18]
    w_37[36:72] = 1.0
    w_37[72:90] = w[18:]

    w_19[72:90] = w[:18]
    w_19[90:] = 1.0
    return np.array([w_73, w_37, w_19]).T.reshape(1, 252, 3)


def create_kernel(dim0, dim1):
    """Create a two-dimensional LPF kernel, with a half-Hamming window along
    the first dimension and a Gaussian along the second.

    Parameters
    ----------
    dim0 : int
        Half-Hamming window length.
    dim1 : int
        Gaussian window length.

    Returns
    -------
    kernel : np.ndarray
        The 2d LPF kernel.
    """
    dim0_weights = np.hamming(dim0 * 2 + 1)[:dim0]
    dim1_weights = gaussian(dim1, dim1 * 0.25, True)
    kernel = dim0_weights[:, np.newaxis] * dim1_weights[np.newaxis, :]
    return kernel / kernel.sum()


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
        data[key] = lcn_octaves(data[key], PARAMS[KERNEL])
    elif data[key].ndim == 3:
        data[key] = np.array([lcn_octaves(x, PARAMS[KERNEL])
                              for x in data[key]])
    else:
        raise ValueError(
            "Cannot transform a {}-dim array.".format(data[key].ndim))
    print("[{0}] Finished: {1}".format(time.asctime(), file_pair.first))
    np.savez(file_pair.second, **data)


def main(args):
    # Set the kernel globally.
    PARAMS[KERNEL] = create_kernel(args.dim0, args.dim1)

    output_dir = futil.create_directory(args.output_directory)
    with open(os.path.join(output_dir, args.param_file), "w") as fp:
        json.dump({"dim0": args.dim0, "dim1": args.dim1}, fp, indent=2)

    pool = Pool(processes=NUM_CPUS)
    pool.map_async(
        func=apply_lcn,
        iterable=futil.map_path_file_to_dir(
            args.textlist_file, output_dir, EXT))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--param_file", type=str,
                        metavar="param_file", default="lcn_params.json",
                        help="Directory to save the parameters used.")
    main(parser.parse_args())
