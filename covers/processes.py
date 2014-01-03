"""
"""

import numpy as np
import cPickle


def hwr(x):
    """Pointwise half-wave rectification.

    Implements:
        z = max(x, 0)

    Parameters
    ----------
    x : np.ndarray
        Input data.

    Returns
    -------
    z : same as x
        Output data.
    """
    return 0.5 * (x + np.abs(x))


def shrinkage(x, theta):
    """Pointwise hard-shrinkage function.

    Implements:
        z = sign(x) * max(|x| - theta, 0)

    Parameters
    ----------
    x : np.ndarray
        Input data.
    theta : scalar
        Must be non-negative.

    Returns
    -------
    z : same as x
        Output data.
    """
    assert theta > 0, "Theta must be non-negative."
    return x + 0.5 * (np.abs(x - theta) - np.abs(x + theta))

def l2_normalize(x):
    """Constrain the tensor rows (axes > 0) to be unit normed in L2-space.
    """
    z = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    u = np.sqrt(np.power(z, 2.0).sum(axis= -1))
    u[u == 0] = 1.0
    u = np.reshape(u, newshape=[x.shape[0]] + [1] * (x.ndim - 1))
    return z / u

def create_process(args):
    """Convenience factory to create a process from a dictionary.

    args : dict
        Must at least contain a "name" key.
    """
    return Processes.get(args.get("name"))(**args)


class Process(dict):
    """Base Process class; all processes inherit from me.

    Conventions
    -----------
    All subclasses should implement and populate two class variables:

    name : str
        String identifier for the process.
    kwargs : list
        Collection of string arguments this layer should require.

    Additionally, each subclass should implement a __call__() method, and
    subsequently register itself in the Processes dictionary at the bottom of
    this file.
    """

    name = "base_process"
    kwargs = []
    def __init__(self, **kwargs):
        for name in self.kwargs:
            assert name in kwargs

        self.update(**kwargs)
        self.update(name=self.name)

    def __call__(self, x):
        return x

    def save(self):
        pass


class LogScale(Process):

    name = "logscale"
    kwargs = ['scalar']
    def __init__(self, **kwargs):
        Process.__init__(self, **kwargs)

    def __call__(self, x):
        assert (x > 0).all()
        return np.log1p(self.get("scalar") * x)


class L2Norm(Process):

    name = "l2norm"
    kwargs = []
    def __init__(self, **kwargs):
        Process.__init__(self, **kwargs)

    def __call__(self, x):
        return l2_normalize(x)



class Shrinkage(Process):

    name = "shrinkage"
    kwargs = ['theta']
    def __init__(self, **kwargs):
        Process.__init__(self, **kwargs)
        assert self.get("theta") > 0, "Theta must be non-negative."

    def __call__(self, x):
        return shrinkage(x, self.get("theta"))


class RectifiedLinear(Process):

    name = "relu"
    kwargs = ['theta']
    def __init__(self, **kwargs):
        Process.__init__(self, **kwargs)

    def __call__(self, x):
        return hwr(x - self.get("theta"))


class ParamProcess(Process):
    """Parameterized Process class; use when the process requires numpy arrays.

    Conventions
    -----------
    All subclasses should implement and populate three class variables:

    name : str
        String identifier for the process.
    kwargs : list
        Collection of string arguments required by this process.
    params : list
        Collection of parameter names required by this process.

    Additionally, each subclass should implement a __call__() method, and
    subsequently register itself in the Processes dictionary at the bottom of
    this file.
    """

    name = "param_process"
    kwargs = ["param_file"]
    params = []
    def __init__(self, **kwargs):
        self.param_values = dict()
        kwargs['param_file'] = kwargs.get("param_file", '')
        if kwargs.get("param_file"):
            self.param_values.update(
                cPickle.load(open(kwargs.get("param_file"))))
        else:
            for name in self.params:
                self.param_values[name] = kwargs.pop(name)

        Process.__init__(self, **kwargs)

    def save(self, filename):
        fh = open(filename, 'w')
        cPickle.dump(self.param_values, fh)
        fh.close()


class DotProduct(ParamProcess):

    name = "dotproduct"
    kwargs = ["param_file"]
    params = ['weights']
    def __init__(self, **kwargs):
        ParamProcess.__init__(self, **kwargs)

    def __call__(self, x):
        W = self.param_values.get("weights")
        assert x.shape[1] == W.shape[1]
        return np.dot(x, W.T)


class Standardize(ParamProcess):

    name = "standardize"
    kwargs = ["param_file"]
    params = ['mu', 'sig']
    def __init__(self, **kwargs):
        ParamProcess.__init__(self, **kwargs)

    def __call__(self, x):
        mu = self.param_values.get("mu")
        if mu.ndim == 1:
            mu = mu[np.newaxis, :]
        sig = self.param_values.get("sig")
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]
        assert x.shape[1] == mu.shape[1] == sig.shape[1]
        sig[sig == 0] = 1.0
        return (x - mu) / sig


class PCA(ParamProcess):

    name = "pca"
    kwargs = ['param_file', 'whiten']
    params = ['components', 'eigenvalues', 'mu']
    def __init__(self, **kwargs):
        ParamProcess.__init__(self, **kwargs)

    def __call__(self, x):
        raise NotImplementedError("Haven't gotten here yet.")



Processes = {LogScale.name: LogScale,
             L2Norm.name: L2Norm,
             Shrinkage.name: Shrinkage,
             RectifiedLinear.name:RectifiedLinear,
             DotProduct.name: DotProduct,
             Standardize.name: Standardize, }
