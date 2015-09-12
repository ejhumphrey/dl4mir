import argparse
import json
import os
import numpy as np
import shutil

np.set_printoptions(precision=4, suppress=True)


def main(args):
    """{param_file, statistic, metric}"""
    with open(args.results_file) as fp:
        score_dict = json.load(fp)

    filenames = score_dict.keys()
    filenames.sort()

    stats = score_dict.values()[0].keys()
    stats.sort()

    metrics = score_dict.values()[0].values()[0].keys()
    metrics.sort()

    scores = np.zeros([len(filenames), len(stats), len(metrics)])
    for i, f in enumerate(filenames):
        for j, s in enumerate(stats):
            for k, m in enumerate(metrics):
                scores[i, j, k] = score_dict[f][s][m]

    # Geometric mean over metrics and stats and find argmax
    idx = np.exp(np.log(scores).mean(axis=-1).mean(axis=-1)).argmax()
    best_file = filenames[idx]
    print "Best param file: {0}".format(best_file)
    print metrics
    print scores[idx]

    checkpoint, penalty = os.path.splitext(best_file)[0].split('/')[-2:]
    with open(args.config_file, 'w') as fp:
        json.dump(
            dict(checkpoint=checkpoint, penalty_values=[penalty]), fp)
    param_file = os.path.split(best_file)[0].replace("estimations", "models")
    shutil.copyfile(
        param_file.replace('valid/', '') + '.npz',
        args.best_param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("results_file",
                        metavar="results_file", type=str,
                        help="Path to JSON object of scores.")
    # Outputs
    parser.add_argument("best_param_file",
                        metavar="best_param_file", type=str,
                        help="Path for renaming the best param archive.")
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path for saving the config params as JSON.")
    main(parser.parse_args())
