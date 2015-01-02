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


def process_stash(stash, transform, output_file, input_key):
    futils.create_directory(os.path.split(output_file)[0])
    if os.path.exists(output_file):
        os.remove(output_file)

    output = biggie.Stash(output_file)
    total_count = len(stash.keys())
    for idx, key in enumerate(stash.keys()):
        output.add(key, convolve(stash.get(key), transform, input_key))
        print "[{0}] {1:7} / {2:7}: {3}".format(
            time.asctime(), idx, total_count, key)

    output.close()


def main(args):
    transform = optimus.load(args.transform_file, args.param_file)
    stash = biggie.Stash(args.data_file)
    process_stash(stash, transform, args.output_file, args.input_key)


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
