import argparse
import marl.fileutils as futils
import optimus
import json
import biggie
import time

# import shutil

import dl4mir.chords.aggregate_likelihood_estimations as ALE
import dl4mir.chords.score_estimations as SE
import dl4mir.common.convolve_graph_with_dset as C


PENALTY_VALUES = [0, -5, -10, -25, -40]


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
    for idx, key in enumerate(stash.keys()[:100]):
        entity_estimations = sweep_penalty(stash.get(key), transform, p_vals)
        for p in p_vals:
            stash_estimations[p][key] = entity_estimations[p]
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, len(stash), key)
    return stash_estimations


def sweep_param_files(param_files, stash, transform, p_vals):
    param_stats = dict([(f, dict()) for f in param_files])
    for f in param_files:
        transform.load_param_values(f)
        stash_estimations = sweep_stash(stash, transform, p_vals)
        for p in p_vals:
            param_stats[f][p] = SE.compute_scores(stash_estimations[p])[0]
            stat_str = SE.stats_to_string(param_stats[f][p])
            print "[%s] %s (%d) \n%s" % (time.asctime(), f, p, stat_str)

    return param_stats


def main(args):
    transform = optimus.load(args.transform_file)

    stash = biggie.Stash(args.validation_file, cache=True)

    param_files = futils.load_textlist(args.param_textlist)
    param_files.sort()
    param_stats = sweep_param_files(
        param_files, stash, transform, PENALTY_VALUES)

    with open(args.stats_file, 'w') as fp:
        json.dump(param_stats, fp, indent=2)

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
