import argparse
import marl.fileutils as futils
from multiprocessing import Pool
import optimus
import json
import biggie
import time
import numpy as np
import os

import shutil

import dl4mir.chords.aggregate_likelihood_estimations as ALE
import dl4mir.chords.score_estimations as SE
import dl4mir.chords.lexicon as lex
import dl4mir.common.transform_stash as TS
import dl4mir.chords.util as util


PENALTY_VALUES = [-1, -2.5, -5, -7.5, -10, -12.5,
                  -15.0, -20.0, -25, -30, -40]
NUM_CPUS = None


def sort_pvals(pvals):
    pidx = np.argsort(np.array(pvals, dtype=float))
    return [pvals[i] for i in pidx[::-1]]


def stats_to_matrix(validation_stats):
    stats = util.filter_empty_values(validation_stats)
    keys = stats.keys()
    keys.sort()

    pvals = sort_pvals(stats[keys[0]].keys())
    metrics = stats[keys[0]].values()[0].keys()
    metrics.sort()
    return np.array([[[stats[k][p][m] for m in metrics] for p in pvals]
                     for k in keys])


def sweep_penalty(entity, transform, p_vals):
    """Predict an entity over a set of penalty values."""
    z = TS.convolve(entity, transform, 'cqt')
    estimations = dict()
    for p in p_vals:
        estimations[p] = ALE.estimate_classes(
            z, prediction_fx=ALE.viterbi, penalty=p)
    return estimations


def parallel_sweep_penalty(entity, transform, p_vals):
    z = TS.convolve(entity, transform, 'cqt')
    pool = Pool(processes=NUM_CPUS)
    threads = [pool.apply_async(ALE.estimate_classes,
                                (biggie.Entity(posterior=z.posterior,
                                               chord_labels=z.chord_labels), ),
                                dict(prediction_fx=ALE.viterbi, penalty=p))
               for p in p_vals]
    pool.close()
    pool.join()

    return dict([(p, t.get()) for p, t in zip(p_vals, threads)])


def sweep_stash(stash, transform, p_vals):
    """Predict all the entities in a stash."""
    stash_estimations = dict([(p, dict()) for p in p_vals])
    for idx, key in enumerate(stash.keys()):
        entity_estimations = parallel_sweep_penalty(
            stash.get(key), transform, p_vals)
        for p in p_vals:
            stash_estimations[p][key] = entity_estimations[p]
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(stash), key)
    return stash_estimations


def sweep_param_files(param_files, stash, transform, p_vals,
                      lexicon, log_file, overwrite=False):
    param_stats = dict([(f, dict()) for f in param_files])
    if os.path.exists(log_file):
        print "Param file found: %s" % log_file
        param_stats.update(json.load(open(log_file)))

    for count, f in enumerate(param_files):
        try:
            transform.load_param_values(f)
            if not param_stats[f] or overwrite:
                print "Sweeping parameters: %s" % f
                stash_estimations = sweep_stash(stash, transform, p_vals)
                for p in p_vals:
                    param_stats[f][str(p)] = SE.compute_scores(
                        stash_estimations[p], lexicon)[0]
            for p in p_vals:
                stat_str = SE.stats_to_string(param_stats[f][str(p)])
                print "[%s] %s (%0.3f) \n%s" % (time.asctime(), f, p, stat_str)
            with open(log_file, 'w') as fp:
                json.dump(param_stats, fp, indent=2)
        except KeyboardInterrupt:
            print "Stopping early after %d parameter archives." % count
            break

    return param_stats


def select_best(validation_stats):
    smat = stats_to_matrix(validation_stats)
    hmeans = 2.0 / (1.0 / smat[:, :, :2]).sum(axis=-1)
    key_idx = hmeans.argmax() / hmeans.shape[1]
    keys = validation_stats.keys()
    keys.sort()
    return keys[key_idx]


def main(args):
    stash = biggie.Stash(args.validation_file)
    transform = optimus.load(args.transform_file)

    param_files = futils.load_textlist(args.param_textlist)
    param_files.sort()
    vocab = lex.Strict(157)
    param_stats = sweep_param_files(
        param_files[4::10], stash, transform, PENALTY_VALUES,
        vocab, args.stats_file)
    best_param_file = select_best(param_stats)
    shutil.copyfile(best_param_file, args.param_file)


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
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path for renaming best parameters.")
    parser.add_argument("stats_file",
                        metavar="stats_file", type=str,
                        help="Path for saving performance statistics.")
    main(parser.parse_args())
