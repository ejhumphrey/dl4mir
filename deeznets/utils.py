"""
"""

import json
import os
import time

TIME_FMT = "%Y%m%d_%H%M%S"


def convert(obj):
    """Convert unicode to strings.

    Known issue: Uses dictionary comprehension, and is incompatible with 2.6.
    """
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [convert(element) for element in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj


def timestamp():
    """Returns a string representation of the time, like:
    YYYYMMDD_HHMMSSmMMM
    """
    return time.strftime(TIME_FMT) + "m%03d" % int((time.time() % 1) * 1000)


def json_save(obj, filename):
    """Serialize data to disk.

    Parameters
    ----------
    obj : iterable
        Data structure to serialize.
    filename : string
        Path to write data.
    """
    base_directory = os.path.split(filename)[0]
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Save json-encoded architecture.
    file_handle = open(filename, "w")
    json.dump(obj, file_handle, indent=2)
    file_handle.close()


def json_load(filename):
    """Serialize data to disk.

    Parameters
    ----------
    obj : iterable
        Data structure to serialize.
    filename : string
        Path to write data.
    """
    return convert(json.load(open(filename)))


def edges_to_connections(edges):
    """
    Parameters
    ----------
    edges: list of tuples (source, sink)
    """
    connections = dict()
    for source, sink in edges:
        if not source in connections:
            connections[source] = []
        connections[source].append(sink)
    return connections


def connections_to_edges(connections):
    """
    Parameters
    ----------
    connections: dict of lists
        Dictionary listing the sinks for each source.
    """
    edges = []
    for source, sinks in connections.items():
        edges.extend([(source, sink) for sink in sinks])
    return edges
