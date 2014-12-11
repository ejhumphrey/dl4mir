"""Apply a graph convolutionally to datapoints in an optimus file."""

import argparse
import numpy as np
import optimus
import biggie
import os
import marl.fileutils as futils
import time


def convolve(entity, graph, input_key, axis=1, chunk_size=250):
    """Convolve a given network over an entity.

    TODO: Use pescador.

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
    # TODO(ejhumphrey): Make this more stable, super fragile as-is
    time_dim = graph.inputs.values()[0].shape[2]
    values = entity.values()
    input_stepper = optimus.array_stepper(
        values.pop(input_key), time_dim, axis=axis, mode='same')
    results = dict([(k, list()) for k in graph.outputs])
    if chunk_size:
        chunk = []
        for x in input_stepper:
            chunk.append(x)
            if len(chunk) == chunk_size:
                for k, v in graph(np.array(chunk)).items():
                    results[k].append(v)
                chunk = []
        if len(chunk):
            for k, v in graph(np.array(chunk)).items():
                results[k].append(v)
    else:
        for x in input_stepper:
            for k, v in graph(x[np.newaxis, ...]).items():
                results[k].append(v)
    for k in results:
        results[k] = np.concatenate(results[k], axis=0)
    values.update(results)
    return biggie.Entity(**values)


def main(args):
    transform = optimus.load(args.transform_file, args.param_file)

    in_stash = biggie.Stash(args.data_file)

    futils.create_directory(os.path.split(args.output_file)[0])
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    out_stash = biggie.Stash(args.output_file)
    total_count = len(in_stash.keys())
    for idx, key in enumerate(in_stash.keys()):
        out_stash.add(
            key, convolve(in_stash.get(key), transform, input_key='cqt'))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)

    out_stash.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("input_key",
                        metavar="input_key", type=str,
                        help="Entity field to transform with the graph.")
    parser.add_argument("transform_file",
                        metavar="transform_file", type=str,
                        help="Transformation graph definition.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path to a parameter archive for the graph.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the transformed output.")
    main(parser.parse_args())
