import argparse
import json
import mir_eval
import os
import pyjams


def main(args):
    metadata = dict()
    if args.annotation_metadata:
        metadata.update(json.load(open(args.annotation_metadata)))

    for key, labfiles in json.load(open(args.annotation_set)).items():
        output_file = os.path.join(args.output_dir, "%s.jams" % key)
        if os.path.exists(output_file):
            jam = pyjams.load(output_file)
        else:
            jam = pyjams.JAMS()
        for labfile in labfiles:
            intervals, labels = mir_eval.io.load_labeled_intervals(labfile)
            annot = jam.chord.create_annotation()
            pyjams.util.fill_range_annotation_data(
                intervals[:, 0], intervals[:, 1], labels, annot)

            annot.annotation_metadata.update(metadata.get(key, {}))

        pyjams.save(jam, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory of labeled interval files to JAMS.")
    parser.add_argument("annotation_set",
                        metavar="annotation_set", type=str,
                        help="JSON object of keys and corresponding labfiles.")
    parser.add_argument("output_dir",
                        metavar="output_dir", type=str,
                        help="Directory to save output JAMS.")
    parser.add_argument("--annotation_metadata",
                        metavar="--annotation_metadata", type=str, default='',
                        help="JSON object with metadata under the same keys.")
    main(parser.parse_args())
