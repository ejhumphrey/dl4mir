"""Collect files in a directory matching a pattern.

Comparable to `ls {input_directory}/{file_pattern} > {output_file}`, with the
exception that files are always sorted regardless of operating system.
"""

import argparse
import glob
from os.path import join

import dl4mir.common.fileutil as futil


def main(input_directory, file_pattern, output_file):
    """Collect files in a directory matching a filepattern.

    Parameters
    ----------
    input_directory : str
        Path to a directory to scrape.
    file_pattern : str
        Pattern to use for file matching.
    output_file : str
        Directory to save output textfile.
    """
    files = glob.glob(join(input_directory, file_pattern))
    files.sort()
    futil.dump_textlist(files, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory",
                        metavar="input_directory", type=str,
                        help="Path to a directory to scrape.")
    parser.add_argument("file_pattern",
                        metavar="file_pattern", type=str,
                        help="Pattern to use for file matching.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Directory to save output textfile.")
    args = parser.parse_args()
    main(args.input_directory, args.file_pattern, args.output_file)
