"""Example for loading data into an optimus File.

The provided json file should look something like the following:

{
  "some_other_key": {
    "numpy_file": "/path/to/a/different/file.npy",
    "label_file": "/path/to/a/lab/file.lab",
  },
  ...
}

"""

import argparse
import json
import mir_eval
import numpy as np
import optimus
import time
import os
import marl.fileutils as futils


def create_entity(cqt_file, lab_file, config, dtype=np.float32):
    """Create an entity from the given item.

    This function exists primarily as an example, and is quite boring. However,
    it expects that each item dictionary has two keys:
        - numpy_file: str
            A valid numpy file on disk
        - label: obj
            This can be any numpy-able datatype, such as scalars, lists,
            strings, or numpy arrays. Dictionaries and None are unsupported.

    Parameters
    ----------
    item: dict
        Contains values for 'numpy_file' and 'label'.
    dtype: type
        Data type to load the requested numpy file.
    """
    data = np.load(cqt_file)
    new_axes = config.get('transpose', range(3))
    data = data.transpose(new_axes)

    intervals, labels = mir_eval.io.load_intervals(lab_file)
    time_points = np.arange(data.shape[1]) / float(config['frame_rate'])
    chord_labels = mir_eval.util.interpolate_intervals(
        intervals, labels, time_points, 'N')

    return optimus.Entity(cqt=data.astype(dtype), chord_labels=chord_labels)


def data_to_file(file_keys, cqt_directory, lab_directory, file_handle,
                 item_parser, config, dtype=np.float32):
    """Load a label dictionary into an optimus file.

    Parameters
    ----------
    file_pairs: dict of dicts
        A collection of file_pairs to load, where the keys of ``file_pairs``
        will become the keys in the file, and the corresponding values are
        sufficient information to load data into an Entity.
    file_handle: optimus.File
        Open for writing data.
    config: dict
        Dictionary containing configuration parameters for this script.
    item_parser: function
        Function that consumes a dictionary of information and returns an
        optimus.Entity. Must take ``dtype`` as an argument.
    """
    total_count = len(file_keys)
    for idx, key in enumerate(file_keys):
        cqt_file = os.path.join(cqt_directory, "%s.npy" % key)
        lab_file = os.path.join(lab_directory, "%s.lab" % key)
        file_handle.add(key, item_parser(cqt_file, lab_file, config, dtype))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    fhandle = optimus.File(args.output_file)
    file_keys = futils.load_textlist(args.split_file)
    config = json.load(open(args.config_file))
    data_to_file(file_keys, args.cqt_directory, args.lab_directory, fhandle,
                 create_entity, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import a collection of numpy files to optimus.")
    parser.add_argument("split_file",
                        metavar="split_file", type=str,
                        help="Textlist of filebases.")
    parser.add_argument("cqt_directory",
                        metavar="cqt_directory", type=str,
                        help="Basepath of CQTs.")
    parser.add_argument("lab_directory",
                        metavar="lab_directory", type=str,
                        help="Basepath of lab files.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filename for the output.")
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path to json file containing settings")

    main(parser.parse_args())
