"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import optimus
import biggie
import os
import time

import dl4mir.common.fileutil as futils
import dl4mir.common.util as util


def main(args):
    transform = optimus.load(args.transform_file, args.param_file)
    stash = biggie.Stash(args.data_file)
    process_stash(stash, transform, args.output_file, args.input_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("input_key",
                        metavar="input_key", type=str,
                        help="Entity field to transform with the graph.")
    parser.add_argument("transform_file",
                        metavar="transform_file", type=str,
                        help="Optimus graph definition.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path to a parameter archive for the graph.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the transformed output.")
    main(parser.parse_args())
