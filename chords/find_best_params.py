import argparse
import marl.fileutils as futils
import optimus
import json
import biggie
import time
import os

# import shutil

import dl4mir.chords.aggregate_likelihood_estimations as ALE
import dl4mir.chords.score_estimations as SE
import dl4mir.chords.lexicon as lex
import dl4mir.common.convolve_graph_with_dset as C

import numpy as np
PENALTY_VALUES = [-1, -2.5, -5, -7.5, -10, -12.5, -15.0, -20.0, -25, -30, -40]
# PENALTY_VALUES = -1.5 - np.arange(10, dtype=float)/5.0


def sweep_penalty(entity, transform, p_vals):
    """Predict an entity over a set of penalty values."""
    z = C.convolve(entity, transform)
    estimations = dict()
    for p in p_vals:
        estimations[p] = ALE.estimate_classes(
            z, prediction_fx=ALE.viterbi, penalty=p)
    return estimations


def sweep_stash(stash, transform, p_vals):
    """Predict all the entities in a stash."""
    stash_estimations = dict([(p, dict()) for p in p_vals])
    for idx, key in enumerate(stash.keys()):
        entity_estimations = sweep_penalty(stash.get(key), transform, p_vals)
        for p in p_vals:
            stash_estimations[p][key] = entity_estimations[p]
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(stash), key)
    return stash_estimations


def sweep_param_files(param_files, stash, transform, p_vals,
                      lexicon, log_file, overwrite=False):
    param_stats = dict([(f, dict()) for f in param_files])
    if os.path.exists(log_file):
        param_stats.update(json.load(open(log_file)))

    for count, f in enumerate(param_files):
        try:
            transform.load_param_values(f)
            stash_estimations = sweep_stash(stash, transform, p_vals)
            for p in p_vals:
                if not param_stats[f].get(p) or overwrite:
                    param_stats[f][p] = SE.compute_scores(stash_estimations[p],
                                                          lexicon)[0]
                stat_str = SE.stats_to_string(param_stats[f][p])
                print "[%s] %s (%0.3f) \n%s" % (time.asctime(), f, p, stat_str)
            with open(log_file, 'w') as fp:
                json.dump(param_stats, fp, indent=2)
        except KeyboardInterrupt:
            print "Stopping early after %d parameter archives." % count
            break

    return param_stats


def main(args):

    stash = biggie.Stash(args.validation_file, cache=True)
    transform = optimus.load(args.transform_file)

    param_files = futils.load_textlist(args.param_textlist)
    param_files.sort()
    vocab = lex.Strict(157)
    param_stats = sweep_param_files(
        param_files[4::10], stash, transform, PENALTY_VALUES,
        vocab, args.stats_file)

    # shutil.copyfile(best_params, args.param_file)


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
    # parser.add_argument("param_file",
    #                     metavar="param_file", type=str,
    #                     help="Path for renaming best parameters.")
    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="Path for saving performance statistics.")
    main(parser.parse_args())
