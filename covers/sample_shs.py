"""
Sample Call:

$ ipython ejhumphrey/covers/sample_shs.py \
/Volumes/Audio/LargeScaleCoverID \
50 \
/Volumes/Audio/LargeScaleCoverID/data_subsample_451d_20131007.npy
"""


import os
import argparse
import numpy as np
import glob
import time
import cPickle


def main(args):
    total_samples = 50000 * args.samples_per_datapoint
    sample = np.zeros([total_samples, 451], dtype='float32')
    pk_files = glob.glob(os.path.join(args.base_dir, "*.pk"))
    sample_count = 0
    for f in pk_files:
        print "%s: Loading %s" % (time.asctime(), f)
        data = cPickle.load(open(f))
        print "%s: Finished .. parsing data." % time.asctime()
        for item in data:
            x = item[0][np.isfinite(item[0].sum(axis=1))]
            if x.shape[0] == 0:
                print "%s: All NaNs - skipping %s." % (time.asctime(), item[-1])
                continue
            idx = np.random.permutation(x.shape[0])[:min([x.shape[0],
                                                          args.samples_per_datapoint])]
            sample[sample_count:sample_count + len(idx)] = x[idx, :451]
            sample_count += len(idx)

    np.save(args.output_file, sample[:sample_count])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a collection of SHS 2DFMCs.")
    parser.add_argument("base_dir",
                        metavar="base_dir", type=str,
                        help="Base path of pickle files.")
    parser.add_argument("samples_per_datapoint",
                        metavar="samples_per_datapoint", type=int,
                        help="Number of samples to pull per file.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File to save data.")
    main(parser.parse_args())
