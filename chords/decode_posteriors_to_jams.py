"""Viterbi decode one of several posterior stashes and output JAMS files.


Example Call:

$ python dl4mir/chords/decode_posteriors_to_jams.py \
path/to/filelist.txt \
path/to/estimations \
--config=viterbi_params.json
"""

import argparse
import biggie
import json
import os
import time

import marl.fileutils as futils
import pyjams

from dl4mir.chords import PENALTY_VALUES
from dl4mir.chords.lexicon import Strict
from dl4mir.chords.decode import decode_stash_parallel

from dl4mir.common import util

NUM_CPUS = 8


def posterior_stash_to_jams(stash, penalty_values, output_directory,
                            vocab, model_params):
    """Decode a stash of posteriors to JAMS and write to disk.

    Parameters
    ----------
    stash : biggie.Stash
        Posteriors to decode.
    penalty_values : array_like
        Collection of penalty values with which to run Viterbi.
    output_directory : str
        Base path to write out JAMS files; each collection will be written as
        {output_directory}/{penalty_values[i]}.jamset
    vocab : dl4mir.chords.lexicon.Vocab
        Map from posterior indices to string labels.
    model_params : dict
        Metadata to associate with the annotation.
    """
    # Sweep over the default penalty values.
    for penalty in penalty_values:
        print "[{0}] \tStarting p = {1}".format(time.asctime(), penalty)
        results = decode_stash_parallel(stash, penalty, vocab, NUM_CPUS)

        output_file = os.path.join(
            output_directory, "{0}.jamset".format(penalty))

        jamset = dict()
        for key, annot in results.iteritems():
            annot.sandbox.update(timestamp=time.asctime(), **model_params)
            jam = pyjams.JAMS(chord=[annot])
            jam.sandbox.track_id = key
            jamset[key] = jam

        futils.create_directory(output_directory)
        util.save_jamset(jamset, output_file)


def main(args):
    penalty_values = list(PENALTY_VALUES)
    if args.config:
        penalty_values = json.load(open(args.config))['penalty_values']

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

        output_dir = os.path.join(args.output_directory, checkpoint)
        posterior_stash_to_jams(
            stash, penalty_values, output_dir, vocab, model_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_filelist",
                        metavar="posterior_filelist", type=str,
                        help="Textlist of posterior stashes.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for output JAMS files.")
    parser.add_argument("--config", default='',
                        metavar="--config", type=str,
                        help="Optional JSON file with parameters for Viterbi.")
    main(parser.parse_args())
