"""Viterbi decode a stash of posteriors and output labeled intervals."""

import argparse
import biggie
import os
import marl.fileutils as futils
import time

import pyjams

from dl4mir.chords.lexicon import Strict
from dl4mir.chords.decode import decode_stash_parallel


NUM_CPUS = 8


def posterior_stash_to_jams(stash, penalty_values, output_directory,
                            vocab, model_params):

    # Sweep over the default penalty values.
    for penalty in penalty_values:
        print "[{0}] \tStarting p = {1:0.2}".format(time.asctime(), penalty)
        results = decode_stash_parallel(stash, penalty, vocab, NUM_CPUS)

        # Create a subdirectory for each penalty value.
        output_dir = futils.create_directory(
            os.path.join(output_directory, "{0}".format(penalty)))
        output_fmt = os.path.join(output_dir, "{0}.jams")
        for key, annot in results.iteritems():
            jam = pyjams.JAMS(chord=[annot])

            jam.sandbox.track_id = key
            annot.sandbox.update(timestamp=time.asctime(), **model_params)

            pyjams.save(jam, output_fmt.format(key))


def main(args):
    vocab = Strict(157)
    for f in futils.load_textlist(args.posterior_filelist):
        print "[{0}] Decoding {1}".format(time.asctime(), f)
        # Read the whole stash to memory because the hdf5 reference doesn't
        #   survive parallelization.
        stash = biggie.Stash(f)
        keys = stash.keys()
        stash = {k: biggie.Entity(**stash.get(k).values()) for k in keys}

        # Parse the posterior stash filepath for its model's params
        parts = os.path.splitext(f)[0].split('outputs/')[-1].split('/')
        model, dropout, fold_idx, split, checkpoint = parts
        model_params = dict(model=model, dropout=dropout, fold_idx=fold_idx,
                            split=split, checkpoint=checkpoint)

        posterior_stash_to_jams(
            stash, args.penalty_values, args.output_directory,
            vocab, model_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_filelist",
                        metavar="posterior_filelist", type=str,
                        help="Path to an biggie stash of chord posteriors.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for output JAMS files.")
    parser.add_argument("--penalty_values", default=[-30.0],
                        metavar="--penalty_values", type=float, nargs='+',
                        help="JSON file containing parameters for Viterbi.")
    main(parser.parse_args())
