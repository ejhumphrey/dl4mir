import argparse
import json
import mir_eval
import glob
import os
import pyjams

import dl4mir.common.fileutil as futil


def main(args):
    metadata = dict()
    if args.annotation_metadata:
        metadata.update(json.load(open(args.annotation_metadata)))

    labs_dt = glob.glob(os.path.join(args.lab_directory, "*_dt.lab"))
    labs_tdc = glob.glob(os.path.join(args.lab_directory, "*_tdc.lab"))
    jfmt = os.path.join(args.output_directory, "%s.jams")

    for dt, tdc in zip(labs_dt, labs_tdc):
        jam = pyjams.JAMS()
        intervals, labels = mir_eval.io.load_labeled_intervals(dt)
        annot_dt = jam.chord.create_annotation()
        pyjams.util.fill_range_annotation_data(
            intervals[:, 0], intervals[:, 1], labels, annot_dt)
        annot_tdc = jam.chord.create_annotation()
        intervals, labels = mir_eval.io.load_labeled_intervals(tdc)
        pyjams.util.fill_range_annotation_data(
            intervals[:, 0], intervals[:, 1], labels, annot_tdc)
        annot_tdc.sandbox.key = "human/RockCorpus/tdeclerqc"
        annot_dt.sandbox.key = "human/RockCorpus/dtemperley"
        jam.sandbox.local_key = futil.filebase(dt).replace("_dt", '')
        pyjams.save(jam, jfmt % jam.sandbox.local_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory of labeled interval files to JAMS.")
    parser.add_argument("lab_directory",
                        metavar="annotation_set", type=str,
                        help="JSON object of keys and corresponding labfiles.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output JAMS.")
    parser.add_argument("--annotation_metadata",
                        metavar="--annotation_metadata", type=str, default='',
                        help="JSON object with track-wise metadata.")
    main(parser.parse_args())
