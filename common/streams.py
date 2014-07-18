"""writemeee."""

import pescador
import biggie.util


def pipeline(stream, functions):
    """writemeee."""
    while True:
        value = stream.next()
        for fx in functions:
            if value is None:
                break
            value = fx(value)
        yield value


def minibatch(stream, batch_size, functions=None):
    """writemeee."""
    stream = pipeline(stream, functions) if functions else stream
    batch_stream = pescador.buffer_stream(stream, batch_size)
    while True:
        yield biggie.util.unpack_entity_list(batch_stream.next(),
                                             filter_nulls=True)
