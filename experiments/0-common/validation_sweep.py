import argparse
import biggie
import optimus
import os

import dl4mir.common.fileutil as futils
from dl4mir.common import util


def params_to_output_file(param_file, output_dir):
    fbase = futils.filebase(param_file)
    return os.path.join(output_dir, "{0}.hdf5".format(fbase))


def main(args):
    param_files = futils.load_textlist(args.param_textlist)
    param_files.sort()
    param_files = param_files[args.start_index::args.stride]

    transform = optimus.load(args.transform_file)
    stash = biggie.Stash(args.validation_file, cache=True)
    output_dir = futils.create_directory(args.output_dir)

    for fidx, param_file in enumerate(param_files):
        transform.load_param_values(param_file)
        output_file = params_to_output_file(param_file, output_dir)
        futils.create_directory(os.path.split(output_file)[0])
        if os.path.exists(output_file):
            os.remove(output_file)

        output = biggie.Stash(output_file)
        util.process_stash(stash, transform, output,
                           args.field, verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("validation_file",
                        metavar="validation_file", type=str,
                        help="Path to a Stash file for validation.")
    parser.add_argument("transform_file",
                        metavar="transform_file", type=str,
                        help="Validator graph definition.")
    parser.add_argument("param_textlist",
                        metavar="param_textlist", type=str,
                        help="Path to save the training results.")
    # Outputs
    parser.add_argument("output_dir",
                        metavar="output_dir", type=str,
                        help="Path for saving JAMS annotations.")
    parser.add_argument("--field",
                        metavar="--field", type=str, default='cqt',
                        help="Field of each entity to use as the input.")
    parser.add_argument("--start_index",
                        metavar="--start_index", type=int, default=0,
                        help="Starting parameter index.")
    parser.add_argument("--stride",
                        metavar="--stride", type=int, default=1,
                        help="Parameter stride.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide console output.")
    main(parser.parse_args())
