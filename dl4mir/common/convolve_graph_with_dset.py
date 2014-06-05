"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import optimus
import os
import marl.fileutils as futils
import time


def convolve(entity, graph, data_key='cqt', chunk_size=250):
    """Convolve a given network over an entity.

    Parameters
    ----------
    entity: optimus.Entity
        Observation to predict.
    graph: optimus.Graph
        Network for processing an entity.
    data_key: str
        Name of the field to use for the input.
    chunk_size: int, default=None
        Number of slices to transform in a given step. When None, parses one
        slice at a time.

    Returns
    -------
    new_entity
    """
    time_dim = graph.inputs.values()[0].shape[2]
    data = entity.values
    data_stepper = optimus.array_stepper(
        data.pop(data_key), time_dim, axis=1, mode='same')
    results = dict([(k, list()) for k in graph.outputs])
    if chunk_size:
        chunk = []
        for value in data_stepper:
            chunk.append(value)
            if len(chunk) == chunk_size:
                for k, v in graph(np.array(chunk)).items():
                    results[k].append(v)
                chunk = []
        if len(chunk):
            for k, v in graph(np.array(chunk)).items():
                results[k].append(v)
    else:
        for value in data_stepper:
            for k, v in graph(value[np.newaxis, ...]).items():
                results[k].append(v)
    for k in results:
        results[k] = np.concatenate(results[k], axis=0)
    data.update(results)
    return optimus.Entity(**data)


def main(args):
    transform = optimus.load(args.transform_file, args.param_file)

    fin = optimus.File(args.data_file)
    futils.create_directory(os.path.split(args.output_file)[0])
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    fout = optimus.File(args.output_file)
    total_count = len(fin.keys())
    for idx, key in enumerate(fin.keys()):
        fout.add(key, convolve(fin.get(key), transform, data_key='cqt'))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("transform_file",
                        metavar="transform_file", type=str,
                        help="Validator graph definition.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path to the parameters for this graph.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the transformed output.")
    main(parser.parse_args())
