"""Rename the 'last' parameter set in a list."""

import argparse
import dl4mir.common.fileutil as futil
import shutil


def main(args):
    param_files = futil.load_textlist(args.param_textlist)
    param_files.sort()
    shutil.copyfile(param_files[-1], args.param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("param_textlist",
                        metavar="param_textlist", type=str,
                        help="Collection of parameter archive filepaths.")
    # Outputs
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path for renaming best parameters.")
    main(parser.parse_args())
