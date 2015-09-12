import argparse
import glob
import json
import mir_eval
from os.path import join
import marl.fileutils as futils

INTERVALS = 'intervals'
LABELS = 'labels'


def main(args):
    annotations = dict()
    for labfile in glob.glob(join(args.lab_directory, "*.lab")):
        key = futils.filebase(labfile)
        intervals, labels = mir_eval.io.load_intervals(labfile)
        annotations[key] = {
            INTERVALS: intervals.tolist(),
            LABELS: labels}

    with open(args.output_file, 'w') as fp:
        json.dump(annotations, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect files in a directory matching a pattern.")
    parser.add_argument("lab_directory",
                        metavar="input_directory", type=str,
                        help="Path to a directory to scrape.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
