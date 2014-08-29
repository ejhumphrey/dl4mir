import argparse
import numpy as np
import marl.fileutils as futils
import optimus
import biggie
import time
import sklearn.metrics as metrics

from os import path
import shutil

import dl4mir.chords.data as D
import dl4mir.common.streams as S

POSTERIOR = 'posterior'


def average_prf(stream, predictor, num_obs=100):
    y_true, y_pred = [], []
    for n in range(NUM_BATCHES):
        data = stream.next()
        y_true.append(data['chord_idx'])
        y_pred.append(predictor(data['cqt'])[POSTERIOR].argmax(axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return [metrics.precision_score(y_true, y_pred),
            metrics.recall_score(y_true, y_pred),
            metrics.f1_score(y_true, y_pred)]


def find_best_param_file(param_files, validator, stream, metric='f1'):
    score_idx = dict(precision=0, recall=1, f1=2)[metric]
    best_score = -np.inf
    best_param_file = ''
    param_files.sort()
    for idx, pf in enumerate(param_files):
        key = path.split(pf)[-1]
        try:
            np.load(pf)
        # What was the error? and why did this happen?
        except:
            print "Warning: Opening '%s' failed." % pf
            continue
        validator.param_values = np.load(pf)
        scores = average_prf(stream, validator)
        score_str = "/".join(["0.4f" % v for v in scores])
        print "[%s] %4d: (%s) %s" % (time.asctime(), idx, score_str, key)
        if scores[score_idx] > best_score:
            best_score = scores[score_idx]
            best_param_file = pf
            print " >>> New best: %0.4f @ %s" % (best_score, key)

    return best_params


def main(args):
    transform = optimus.load(args.transform_file)
    time_dim = transform.inputs.values()[0].shape[2]

    stash = biggie.Stash(args.data_file)
    stream = S.minibatch(
        D.create_chord_stream(stash, time_dim, pitch_shift=0),
        batch_size=200)

    best_params = find_best_param_file(
        param_files=futils.load_textlist(args.param_textlist),
        stream=stream,
        transform=transform)

    shutil.copyfile(best_params, args.param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
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
    main(parser.parse_args())
