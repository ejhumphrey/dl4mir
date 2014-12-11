"""Utility to count the class distribution in a Biggie Stash."""
import argparse
import marl.fileutils as futils

from os import path
import biggie
import json
import dl4mir.chords.data as D
import dl4mir.chords.lexicon as lex


def main(args):
    stash = biggie.Stash(args.input_file)
    futils.create_directory(path.split(args.output_file)[0])

    stats = dict()
    vocab = lex.Strict(157)

    stats['prior'] = D.class_prior_v157(stash, vocab).tolist()

    with open(args.output_file, 'w') as fp:
        json.dump(stats, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beat-synchronize a dataset of entities.")
    parser.add_argument("input_file",
                        metavar="input_file", type=str,
                        help="Path to the input biggie file.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to the output JSON file.")
    main(parser.parse_args())
