"""Viterbi decode a stash of posteriors and output labeled intervals."""

import argparse
import json
import biggie
import os
import marl.fileutils as futils
import time
from multiprocessing import Pool

import pyjams

from dl4mir.chords.lexicon import Strict
from dl4mir.chords.decode import decode_stash_parallel


NUM_CPUS = 12


def main(args):
    stash = biggie.Stash(args.posterior_file)
    output_dir = futils.create_directory(args.output_directory)

    vocab = Strict(157)
    results = decode_stash_parallel(stash, args.penalty, vocab, NUM_CPUS)

    parts = args.posterior_file.split('outputs/')[-1].split('/')
    model, dropout, fold_idx, split = parts[:4]
    config_params = dict(model=model, fold_idx=fold_idx,
                         split=split, dropout=dropout)

    for key, annot in results.iteritems():
        output_file = os.path.join(output_dir, "%s.jams" % key)
        jam = pyjams.JAMS(chord=[annot])

        annot.annotation_metadata.annotator = dict(
            from_file=args.posterior_file,
            timestamp=time.asctime())

        jam.sandbox.track_id = key
        annot.sandbox.update(**config_params)

        pyjams.save(jam, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an biggie stash of chord posteriors.")
    parser.add_argument("penalty",
                        metavar="penalty", type=float,
                        help="Viterbi self-transition penalty.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for output JAMS files.")
    main(parser.parse_args())
