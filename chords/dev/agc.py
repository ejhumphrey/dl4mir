import numpy as np
from scipy import signal


def agc(X, t_scale=0.5):
    ftsr = 20.0
    tsd = np.round(t_scale * ftsr) / 2
    htlen = 6 * tsd  # Go out to 6 sigma
    twin = np.exp(-0.5 * (((np.arange(-htlen, htlen + 1)) / tsd) ** 2)).T

    ndcols = X.shape[1]

    # reflect ends to get smooth stuff
    x = np.hstack((np.fliplr(X[:, :htlen]),
                   X,
                   np.fliplr(X[:, -htlen:]),
                   np.zeros((X.shape[0], htlen))))
    fbg = signal.lfilter(twin, 1, x, 1)

    # strip "warm up" points
    fbg = fbg[:, twin.size + np.arange(ndcols)]

    # Remove any zeros in E (shouldn't be any, but who knows?)
    fbg[fbg <= 0] = np.min(fbg[fbg > 0])
    return X / fbg