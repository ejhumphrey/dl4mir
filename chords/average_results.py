import argparse
import json
import numpy as np
import tabulate

import marl.fileutils as futil


def main(args):
    """{param_file, statistic, metric}"""
    scores = [json.load(open(f)).values()[0]
              for f in futil.load_textlist(args.score_textlist)]

    stats = scores[0].keys()
    stats.sort()

    metrics = scores[0].values()[0].keys()
    metrics.sort()

    table = np.zeros([len(scores), len(stats), len(metrics)])
    for i, score in enumerate(scores):
        for j, s in enumerate(stats):
            for k, m in enumerate(metrics):
                table[i, j, k] = score[s][m]

    aves = table.mean(axis=0)
    stdevs = table.std(axis=0)

    res = []
    for j, s in enumerate(stats):
        res.append([s])
        for k, m in enumerate(metrics):
            val = "${0:0.3}\pm{1:0.3}$".format(aves[j, k], stdevs[j, k])
            res[-1].append(val)

    print tabulate.tabulate(res, headers=metrics)

    # with open(args.output_file, 'w') as fp:
    #     json.dump(
    #         dict(checkpoint=checkpoint, penalty_values=[penalty]), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("score_textlist",
                        metavar="score_textlist", type=str,
                        help="List of JSON score objects.")
    # # Outputs
    # parser.add_argument("output_file",
    #                     metavar="output_file", type=str,
    #                     help="Path for saving the final output.")
    main(parser.parse_args())
