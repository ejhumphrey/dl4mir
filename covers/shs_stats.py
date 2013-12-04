"""
Sample Call:

$ ipython ejhumphrey/covers/shs_stats.py \
/Volumes/Audio/LargeScaleCoverID/feature_subsample/ \
/Volumes/Audio/LargeScaleCoverID/subsample_stats_20131014.pk
"""


import os
import argparse
import glob
import time
import cPickle

def main(args):

    stats = dict()
    pk_files = glob.glob(os.path.join(args.base_dir, "*.pk"))

    for f in pk_files:
        print "%s: Loading %s" % (time.asctime(), f)
        data = cPickle.load(open(f))
        print "%s: Finished .. parsing data." % time.asctime()
        for idx, item in enumerate(data):
            x = item[0][:, :451]
            key = os.path.join(f, "%03d" % idx)
            stats[key] = (x.mean(axis=0), x.std(axis=0))

    fp = open(args.output_file, 'w')
    cPickle.dump(stats, fp)
    fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a collection of SHS 2DFMCs.")
    parser.add_argument("base_dir",
                        metavar="base_dir", type=str,
                        help="Base path of pickle files.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File to save data.")
    main(parser.parse_args())
