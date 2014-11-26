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
from dl4mir.chords import util as cutil
from dl4mir.common import util


NUM_CPUS = 12


def fx(args):
    entity, penalty, vocab, idx, key = args
    print "[%s] %12d / %s" % (time.asctime(), idx, key)
    return cutil.posterior_to_labeled_intervals(entity, penalty, vocab)


def arg_gen(stash, keys, penalty, vocab):
    for idx, key in enumerate(keys):
        entity = stash.get(key)
        yield (biggie.Entity(**entity.values()), penalty, vocab, idx, key)


def main(args):
    stash = biggie.Stash(args.posterior_file)
    output_dir = futils.create_directory(args.output_directory)
    stats = json.load(open(args.validation_stats))
    penalty = float(stats['best_config']['penalty'])
    vocab = Strict(157)

    keys = stash.keys()
    pool = Pool(processes=NUM_CPUS)
    results = pool.map(fx, arg_gen(stash, keys, penalty, vocab))
    pool.close()
    pool.join()

    for key, res in zip(keys, results):
        intervals, labels = res
        output_file = os.path.join(output_dir, "%s.jams" % key)
        jam = pyjams.JAMS()
        annot = jam.chord.create_annotation()
        pyjams.util.fill_range_annotation_data(
            intervals[:, 0], intervals[:, 1], labels, annot)
        annot.annotation_metadata.data_source = 'machine estimation'
        annot.annotation_metadata.annotator = dict(
            from_file=args.posterior_file,
            timestamp=time.asctime(),
            **stats['best_config'])
        pyjams.save(jam, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Path to an biggie stash of chord posteriors.")
    parser.add_argument("validation_stats",
                        metavar="validation_stats", type=str,
                        help="Output of the validation parameter sweep.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path for output JAMS files.")
    main(parser.parse_args())