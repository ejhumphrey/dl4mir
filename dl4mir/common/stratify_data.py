import argparse
import json

import dl4mir.common.fileutil as futils
import dl4mir.common.util as util


def main(args):
    files = futils.load_textlist(args.textlist)
    keys = [futils.filebase(f) for f in files]
    folds = util.stratify(keys, args.num_folds, args.valid_ratio)
    with open(args.output_file, 'w') as fp:
        json.dump(folds, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect files in a directory matching a pattern.")
    parser.add_argument("textlist",
                        metavar="textlist", type=str,
                        help="Path to a textlist file.")
    parser.add_argument("num_folds",
                        metavar="num_folds", type=int,
                        help="Number of splits for the data.")
    parser.add_argument("valid_ratio",
                        metavar="valid_ratio", type=float,
                        help="Ratio of the training data for validation.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File to save the output splits as JSON.")
    main(parser.parse_args())
