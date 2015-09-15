"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import biggie
import optimus
import os

import dl4mir.common.fileutil as futil
import dl4mir.common.util as util


def main(stash_file, input_key, transform_file,
         param_file, output_file, verbose=True):
    transform = optimus.load(transform_file, param_file)
    stash = biggie.Stash(stash_file)
    futil.create_directory(os.path.split(output_file)[0])
    output = biggie.Stash(output_file)
    util.process_stash(stash, transform, output, input_key, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("stash_file",
                        metavar="stash_file", type=str,
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
    args = parser.parse_args()
    main(args.stash_file, args.input_key, args.transform_file,
         args.param_file, args.output_file)
