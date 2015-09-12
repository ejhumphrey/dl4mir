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
import marl.fileutils as futils
import time
import cPickle
from sklearn.decomposition import PCA

NUM_CPUS = 4  # Use None for system max.

# Global dict
EXT = "_pca.pkl"

def run(file_pair):
    """Fit PCA for a input numpy array.

    Parameters
    ----------
    file_pair : Pair of strings
        input_file and output file

    Returns
    -------
    Nothing, but the output file is written in this call.
    """
    x_obs = np.load(file_pair.first)
    shp = x_obs.shape
    pca = PCA().fit(x_obs.reshape(shp[0], np.prod(shp[1:])))
    print "[%s] Finished: %s" % (time.asctime(), file_pair.first)
    with open(file_pair.second, 'w') as fp:
        cPickle.dump(pca, fp)


def main(args):
    """Main routine for staging parallelization."""
    output_dir = futils.create_directory(args.output_directory)
    pool = Pool(processes=NUM_CPUS)
    pool.map_async(
        func=run,
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
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
