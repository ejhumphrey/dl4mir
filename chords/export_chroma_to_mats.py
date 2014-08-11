"""Utility to dump a biggie Stash to a flat collection of Matlab files."""
import argparse
import marl.fileutils as futils
import numpy as np
from os import path
import json
import biggie
import scipy.io.matlab as MLAB
import time

# TMC does no filename parsing...
FILE_EXT = "mp3.mat"


def entity_to_mdict(entity, labseg):
    chroma = entity.chroma.value.T
    time_points = entity.time_points.value
    labseg = np.array(labseg, dtype=np.uint32)
    # This is a gross hack. Where is this +1 index issue coming from?
    L = min([chroma.shape[1], len(time_points), len(labseg)])
    if chroma.shape[0] != 12:
        raise ValueError(
            "First dimension of chroma must be 12, shape=%s" % chroma.shape)
    chroma_obj = np.empty([1, 1], dtype=object)
    chroma_obj[0, 0] = chroma[:, :L]
    return dict(
        time_points=time_points[:L].reshape(L, 1),
        chroma=chroma_obj,
        labseg=labseg[:L],
        endT=float(entity.endT.value))


def main(args):
    dset = biggie.Stash(args.input_file)
    labseg = json.load(open(args.labseg))
    out_dir = futils.create_directory(args.output_directory)
    total_count = len(dset)
    for idx, key in enumerate(dset.keys()):
        out_file = path.join(out_dir, "%s.%s" % (key, FILE_EXT))
        mdict = entity_to_mdict(dset.get(key), labseg[key])
        MLAB.savemat(out_file, mdict=mdict)
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump an biggie Stash to Matlab .mat files.")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Path to a file to export.")
    parser.add_argument("labseg",
                        metavar="labseg", type=str, default='',
                        help="JSON file of TMC's lab segment data; "
                        " keys must match.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
