import argparse
import json
import mir_eval
import os
import time
import pyjams

from dl4mir.common import util
import marl.fileutils as futil


def main(args):
    metadata = dict()
    if args.annotation_metadata:
        metadata.update(json.load(open(args.annotation_metadata)))

    jamset = dict()
    for key, lab_files in json.load(open(args.annotation_set)).items():
        jamset[key] = pyjams.JAMS()
        for f in [lab_files]:
            intervals, labels = mir_eval.io.load_labeled_intervals(str(f))
            annot = jamset[key].chord.create_annotation()
            pyjams.util.fill_range_annotation_data(
                intervals[:, 0], intervals[:, 1], labels, annot)

            annot.annotation_metadata.update(**metadata.get(key, {}))
            annot.sandbox.source_file = f
            annot.sandbox.timestamp = time.asctime()

    futil.create_directory(os.path.split(args.output_file)[0])
    util.save_jamset(jamset, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory of labeled interval files to JAMS.")
    parser.add_argument("annotation_set",
                        metavar="annotation_set", type=str,
                        help="JSON object of keys and corresponding labfiles.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Directory to save the output JAMSet.")
    parser.add_argument("--annotation_metadata",
                        metavar="--annotation_metadata", type=str, default='',
                        help="JSON object with metadata under the same keys.")
    main(parser.parse_args())
