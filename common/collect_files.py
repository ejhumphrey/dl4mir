import argparse
import glob
import marl.fileutils as futils
from os.path import join


def main(args):
    files = glob.glob(join(args.input_directory, args.file_pattern))
    files.sort()
    futils.dump_textlist(files, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect files in a directory matching a pattern.")
    parser.add_argument("input_directory",
                        metavar="input_directory", type=str,
                        help="Path to a directory to scrape.")
    parser.add_argument("file_pattern",
                        metavar="file_pattern", type=str,
                        help="Pattern to use for file matching.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Directory to save output arrays.")
    main(parser.parse_args())
