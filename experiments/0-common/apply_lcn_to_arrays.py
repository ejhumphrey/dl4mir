#!/usr/bin/env python
"""Apply LCN to a collection of numpy arrays.

Sample Call:
$ python apply_lcn_to_arrays.py \
/Volumes/Audio/Chord_Recognition/rwc_filelist.txt \
/Volumes/Audio/Chord_Recognition/cqt_params.txt
"""
from __future__ import print_function

import argparse
from joblib import delayed
from joblib import Parallel
import json
import numpy as np
import os
import time

from dl4mir.common import fileutil as futil
from dl4mir.common.lcn import lcn_octaves as lcn
from dl4mir.common.lcn import create_kernel

# Globals
KERNEL = 'kernel'
PARAMS = dict()
EXT = ".npz"


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
        data[key] = lcn(data[key], PARAMS[KERNEL])
    elif data[key].ndim == 3:
        data[key] = np.array([lcn(x, PARAMS[KERNEL])
                              for x in data[key]])
    else:
        raise ValueError(
            "Cannot transform a {}-dim array.".format(data[key].ndim))
    print("[{0}] Finished: {1}".format(time.asctime(), file_pair.first))
    np.savez(file_pair.second, **data)
    return True


def main(textlist, dim0, dim1, output_directory, param_file, num_cpus=-1):
    """Apply Local Contrast Normalization to a collection of files.

    Parameters
    ----------
    textlist : str
        A text list of npz filepaths.
    dim0 : int
        First dimension of the filter kernel (time).
    dim1 : int
        Second dimension of the filter kernel (frequency).
    output_directory : str
        Directory to save output arrays.
    param_file : str
        Directory to save the parameters used.
    num_cpus : int, default=-1
        Number of CPUs over which to parallelize computations.
    """
    # Set the kernel globally.
    PARAMS[KERNEL] = create_kernel(dim0, dim1)

    output_dir = futil.create_directory(output_directory)
    with open(os.path.join(output_dir, param_file), "w") as fp:
        json.dump({"dim0": dim0, "dim1": dim1}, fp, indent=2)

    pool = Parallel(n_jobs=num_cpus)
    dlcn = delayed(apply_lcn)
    iterargs = futil.map_path_file_to_dir(textlist, output_dir, EXT)
    return pool(dlcn(x) for x in iterargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("textlist",
                        metavar="textlist", type=str,
                        help="A text list of npz filepaths.")
    parser.add_argument("dim0",
                        metavar="dim0", type=int,
                        help="First dimension of the array (time).")
    parser.add_argument("dim1",
                        metavar="dim1", type=int,
                        help="Second dimension of the filter kernel (freq).")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    parser.add_argument("--param_file", type=str,
                        metavar="param_file", default="lcn_params.json",
                        help="Directory to save the parameters used.")
    parser.add_argument("--num_cpus", type=int,
                        metavar="num_cpus", default=-1,
                        help="Number of CPUs over which to parallelize "
                             "computations.")
    args = parser.parse_args()
    main(args.textlist, args.dim0, args.dim1, args.output_directory,
         args.param_file, args.num_cpus)
