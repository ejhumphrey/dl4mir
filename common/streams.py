"""writemeee."""

import pescador
import biggie.util
import numpy as np


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

def mux(streams, weights):
    weights = np.array(weights)
    assert weights.sum() > 0
    weights /= float(weights.sum())
    while True:
        idx = pescador.categorical_sample(weights)
        yield streams[idx].next()

import Queue
import threading
import time


def lazy_range(num_elements, delay):
    for n in xrange(num_elements):
        time.sleep(delay)
        yield n


class ThreadedStream(threading.Thread):
    """Wraps a generator to stream data in a separate thread."""
    def __init__(self, stream, size=1, delay=0.005):
        assert size > 0
        threading.Thread.__init__(self)
        self._stream = stream
        self._size = size
        self._delay = delay
        self._queue = Queue.Queue(size)
        self._lock = threading.Lock()
        self._exhausted = False
        self._done = False

        # Initial setup
        while not self._queue.full() and not self._exhausted:
            self.__load_next__()
        threading.Thread.start(self)

    def run(self):
        while not self._done:
            if not self._queue.full() and not self._exhausted:
                self.__load_next__()
            else:
                time.sleep(self._delay)

    def next(self):
        if self._exhausted and self._queue.empty():
            self.stop()
            raise StopIteration

        while self._queue.empty():
            time.sleep(self._delay)

        self._lock.acquire()
        x = self._queue.get(0)
        self._lock.release()
        return x

    def __load_next__(self):
        if self._queue.full() or self._exhausted:
            return
        try:
            x = self._stream.next()
        except StopIteration:
            self._exhausted = True
            return
        self._lock.acquire()
        self._queue.put(x)
        self._lock.release()

    def stop(self):
        self._done = True

    def __del__(self):
        self.stop()

    def __iter__(self):
        while True:
            yield self.next()
