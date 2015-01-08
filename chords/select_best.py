import argparse
import json
import os
import numpy as np


def main(args):
    """{param_file, statistic, metric}"""
    with open(args.results_file) as fp:
        score_dict = json.load(fp)

    filenames = score_dict.keys()
    filenames.sort()

    stats = score_dict.values()[0].keys()
    stats.sort()

    metrics = score_dict.values()[0].values().keys()
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
    print score_dict[best_file]

    checkpoint, penalty = os.path.splitext(best_file)[0].split('/')[-2:]
    with open(args.output_file) as fp:
        json.dump(dict(checkpoint=checkpoint, penalty=penalty), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("results_file",
                        metavar="results_file", type=str,
                        help="Path to JSON object of scores.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for saving the results as JSON.")
    main(parser.parse_args())
