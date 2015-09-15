from biggie import util
import numpy as np
import pescador


def _pipeline(stream, functions):
    """writemeee."""
    while True:
        value = stream.next()
        for fx in functions:
            if value is None:
                break
            value = fx(value)
        yield value


def minibatch(stream, batch_size, functions=None):
    """Buffer a stream into a minibatch.

    Parameters
    ----------
    stream : iterator
        A stream of Entities.
    batch_size : int
        Number of observations to buffer into a batch.
    functions : list of callables
        Sequence of functions to apply in order to each entity.

    Yields
    ------
    batch : dict of np.ndarrays
        Key-value object mapping fields of the Entities to arrays of data.
    """
    stream = _pipeline(stream, functions) if functions else stream
    batch_stream = pescador.buffer_stream(stream, batch_size)
    while True:
        yield util.unpack_entity_list(batch_stream.next(),
                                      filter_nulls=True)


def mux(streams, weights):
    """Multiplex multiple streams into one.

    Parameters
    ----------
    streams : list of iterators, len=n
        Collection of different streams to sample.
    weights : array_like, len=n
        Corresponding probabilities of choosing each stream.
    """
    weights = np.array(weights)
    assert weights.sum() > 0
    weights /= float(weights.sum())
    while True:
        idx = pescador.categorical_sample(weights)
        yield next(streams[idx])
