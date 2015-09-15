"""Stratify a textlist into a number of disjoint partitions."""

import argparse
import json

import dl4mir.common.fileutil as futils
import dl4mir.common.util as util


def main(textlist, num_folds, valid_ratio, output_file):
    """Stratify a textlist into a number of disjoint partitions.

    Parameters
    ----------
    textlist : str
        Path to a textlist file.
    num_folds : int
        Number of splits for the data.
    valid_ratioratio : scalar, in [0, 1.0)
        Ratio of the training data for validation.
    output_file_file : str
        File to save the output splits as JSON.
    """
    files = futils.load_textlist(textlist)
    keys = [futils.filebase(f) for f in files]
    folds = util.stratify(keys, num_folds, valid_ratio)
    with open(output_file, 'w') as fp:
        json.dump(folds, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    main(args.textlist, args.num_folds, args.valid_ratio, args.output_file)
