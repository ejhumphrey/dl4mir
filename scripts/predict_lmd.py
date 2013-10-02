#!/usr/bin/env python
"""Consumes a file list of matfiles, producing posteriorgrams.

Importantly, a definition file is assumed to exist in the same directory
as the provided parameters, like "*.definition".

Sample Call:
MODELDIR=/Volumes/Audio/LMD/models/test000
ipython ejhumphrey/scripts/predict_lmd.py \
/Volumes/Audio/LMD/splits/test00_20131001.txt \
/Volumes/Audio/LMD/LMD_scalars240x10_mid.pkl \
$MODELDIR/test000_0050000-20131001_204255m453.params \
/Volumes/Audio/LMD/posteriors
"""

import argparse
import glob
from multiprocessing import Pool
import numpy as np
import os
import time

from ejhumphrey.dnn.core.graphs import Network
from ejhumphrey.datasets.lmd import matfile_to_datasequence
import cPickle

NUM_CPUS = 0
transform = dict()

class Pair(object):
    def __init__(self, first, second):
        self.first, self.second = first, second

    def __str__(self):
        return "(%s, %s)" % (self.first, self.second)


def predict_matfile(iopair):
    net = transform.get("network")
    dseq = matfile_to_datasequence(iopair.first, transform.get("stdev_params"))
    inputs = net.empty_inputs()
    inputs[net.input_name] = dseq.value()
    result = net(inputs)
    np.save(iopair.second, result)

    print "[%s] Finished: %s" % (time.asctime(),
                                 os.path.split(iopair.second)[-1])


def create_network(param_file):
    def_files = glob.glob(os.path.join(os.path.split(param_file)[0],
                                       "*.definition"))
    assert len(def_files) == 1, \
        "More than one definition file found? %s" % def_files

    net = Network.load(def_files[0], param_file)
    net.compile()
    return net


def iofiles(file_list, output_dir):
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
        mat_file = line.strip("\n")
        output_file = os.path.join(output_dir, os.path.split(mat_file)[-1])
        yield Pair(mat_file, output_file)


def create_output_directory(output_directory, param_file):
    """
    Returns
    -------
    output_dir : str
        Expanded path, that now certainly exists.
    """
    param_base = os.path.split(os.path.splitext(param_file)[0])[-1]
    output_dir = os.path.join(output_directory, param_base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print "[%s] Output Directory: %s" % (time.asctime(), output_dir)
    return output_dir


def main(args):
    """Main routine for staging parallelization."""
    output_dir = create_output_directory(args.output_directory,
                                         args.param_file)

    transform['network'] = create_network(args.param_file)
    transform['stdev_params'] = cPickle.load(open(args.stdev_file))

    if NUM_CPUS:
        pool = Pool(processes=NUM_CPUS)
        pool.map_async(func=predict_matfile,
                       iterable=iofiles(args.file_list,
                                        output_dir))
        pool.close()
        pool.join()
    else:
        [predict_matfile(iopair) for iopair in iofiles(args.file_list,
                                                       output_dir)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict LMD posteriors.")

    parser.add_argument("file_list",
                        metavar="file_list", type=str,
                        help="Text file of filepaths to predict.")

    parser.add_argument("stdev_file",
                        metavar="stdev_file", type=str,
                        help="WRITEME.")

    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Pickled dictionary of parameters.")

    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output posteriors.")

    main(parser.parse_args())
