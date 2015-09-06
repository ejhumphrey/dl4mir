import numpy as np


def parabola(xlim=(-5, 5), scale=1, offset=(0, 0)):
    """Sample from a parabolic distribution:

        y = scale * (x - offset[0]) ** 2 + offset[1]

    Parameters
    ----------
    xlim : tuple, len=2
        X-limits on which to sample the parabola.
    scale : scalar, default=1
        Scaling coefficient.
    offset : tuple, len=2
        X and Y coordinate offsets.

    Yields
    ------
    xy : ndarray, shape=(2,)
        An (x, y) coordinate pair.
    """
    assert len(xlim) == 2
    while True:
        x = np.random.rand()*np.abs(np.diff(xlim)) + xlim[0]
        y = scale * np.power(x - offset[0], 2.0) - offset[1]
        yield np.array([x, y]).squeeze()


def gaussian2d(mean, std):
    """Sample from a Gaussian (normal) distribution.

    Parameters
    ----------
    mean : tuple, len=2
        Sample means for (x, y).
    std : tuple, len=2
        Sample standard deviations for (x, y).

    Yields
    ------
    xy : ndarray, shape=(2,)
        An (x, y) coordinate pair.
    """
    assert len(mean) == len(std) == 2
    while True:
        x = np.random.normal(loc=mean[0], scale=std[0])
        y = np.random.normal(loc=mean[1], scale=std[1])
        yield np.array([x, y])


def merge(streams, probs=None):
    """Stochastically merge a collection of streams.

    Parameters
    ----------
    streams : array_like, len=n
        Collection of streams from which to draw samples.
    probs : array_like, len=n, or None (default)
        Probability of drawing a sample from each stream; if None,
        a uniform distribution is used.

    Yields
    ------
    Same as streams[i]
    """
    if probs is None:
        probs = np.ones(len(streams))
    probs = np.asarray(probs, dtype=float) / np.sum(probs)
    while True:
        idx = np.random.choice(len(streams), p=probs)
        yield streams[idx].next()


def batch(streams, batch_size, probs=None):
    """Batch sample a collection of streams.

    Parameters
    ----------
    streams : array_like, len=n
        Collection of streams from which to draw samples.
    batch_size : int
        Number of samples to return on each batch.
    probs : array_like, len=n, or None (default)
        Probability of drawing a sample from each stream; if None,
        a uniform distribution is used.

    Yields
    ------
    data : dict
        A batch of data with keys [``x_input``, ``y_target``].
        Note that y_target corresponds to the index of the stream from
        which the observation came.
    """
    if probs is None:
        probs = np.ones(len(streams))
    probs = np.asarray(probs, dtype=float) / np.sum(probs)
    while True:
        x_input, y_target = [], []
        while len(y_target) < batch_size:
            idx = np.random.choice(len(streams), p=probs)
            x_input.append(streams[idx].next())
            y_target.append([idx])
        yield dict(x_input=np.array(x_input), y_target=np.array(y_target))
