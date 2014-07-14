import argparse
import numpy as np
import marl.fileutils as futils
import optimus
from os.path import split
from ejhumphrey.dl4mir.guitar import transformers as T
import shutil

FRET_DIM = 12
SOURCE_ARGS = dict(
    batch_size=100,
    refresh_prob=0.,
    cache_size=500)
NUM_OBS = 100

LABEL_MAPS = {
    "tabber": T.fret_indexes_to_bitmap(FRET_DIM),
}


def average_loss(source, predictor):
    param_loss = 0.0
    for n in range(NUM_OBS):
        param_loss += predictor(**source.next())[optimus.Graph.TOTAL_LOSS]
    return param_loss / float(NUM_OBS)


def find_best_param_file(param_files, validator, source):
    best_loss = np.inf
    best_params = ''
    for pf in param_files:
        validator.param_values = np.load(pf)
        param_loss = average_loss(source, validator)
        if param_loss < best_loss:
            best_loss = param_loss
            best_params = pf
            print "New best: %0.4f @ %s" % (best_loss, split(pf)[-1])
    return best_params


def main(args):
    validator = optimus.load(args.validator_file)
    time_dim = validator.inputs.values()[0].shape[2]

    source = optimus.Queue(
        optimus.File(args.data_file),
        transformers=[
            T.cqt_sample(time_dim),
            LABEL_MAPS[validator.name]],
        **SOURCE_ARGS)

    best_params = find_best_param_file(
        param_files=futils.load_textlist(args.param_textlist),
        source=source,
        validator=validator)

    shutil.copyfile(best_params, args.param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("validator_file",
                        metavar="validator_file", type=str,
                        help="Validator graph definition.")
    parser.add_argument("param_textlist",
                        metavar="param_textlist", type=str,
                        help="Path to save the training results.")
    # Outputs
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path for renaming best parameters.")
    main(parser.parse_args())
