import numpy as np
from scipy.signal.signaltools import convolve2d
from scipy.signal.windows import gaussian

from util import hwr


def lcn(X, kernel):
    """Apply Local Contrast Normalization (LCN) to an array.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = X - Xh
    S = np.sqrt(convolve2d(np.power(V, 2.0),
                kernel, mode='same', boundary='symm'))
    S2 = np.zeros(S.shape) + S.mean()
    S2[S > S.mean()] = S[S > S.mean()]
    if S2.sum() == 0.0:
        S2 += 1.0
    return V / S2


def lcn_v2(X, kernel, mean_scalar=1.0):
    """Apply an alternative version of local contrast normalization (LCN) to an
    array.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = X - Xh
    S = np.sqrt(convolve2d(np.power(V, 2.0),
                kernel, mode='same', boundary='symm'))
    thresh = np.exp(np.log(S + np.power(2.0, -5)).mean(axis=-1))
    S = S*np.greater(S - thresh.reshape(-1, 1), 0)
    S += 1.0*np.equal(S, 0.0)
    return V / S


def lcn_mauch(X, kernel=None, rho=0):
    """Apply a version of local contrast normalization (LCN), inspired by
    Mauch, Dixon (2009), "Approximate Note Transcription...".

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).
    rho : scalar
        Scalar applied to the final output for heuristic range control.

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if kernel is None:
        dim0, dim1 = 15, 37
        dim0_weights = np.hamming(dim0 * 2 + 1)[:dim0]
        dim1_weights = np.hamming(dim1)
        kernel = dim0_weights[:, np.newaxis] * dim1_weights[np.newaxis, :]

    kernel /= kernel.sum()
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    V = hwr(X - Xh)
    S = np.sqrt(
        convolve2d(np.power(V, 2.0), kernel, mode='same', boundary='symm'))
    S2 = np.zeros(S.shape) + S.mean()
    S2[S > S.mean()] = S[S > S.mean()]
    if S2.sum() == 0.0:
        S2 += 1.0
    return V / S2**rho


def highpass(X, kernel):
    """Produce a highpass kernel from its lowpass complement.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    Xh = convolve2d(X, kernel, mode='same', boundary='symm')
    return X - Xh


def local_l2norm(X, kernel):
    """Apply local l2-normalization over an input with a given kernel.

    Parameters
    ----------
    X : np.ndarray, ndim=2
        Input representation.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    local_mag = np.sqrt(convolve2d(np.power(X, 2.0),
                        kernel, mode='same', boundary='symm'))
    local_mag = local_mag + 1.0*(local_mag == 0.0)
    return X / local_mag


def lcn_octaves(X, kernel):
    """Apply octave-varying contrast normalization to an input with a given
    kernel.

    Notes:
    * This is the variant introduced in the LVCE Section of Chapter 5.
    * This approach is painfully heuristic, and tuned for the dimensions used
        in this work (36 bpo, 7 octaves).

    Parameters
    ----------
    X : np.ndarray, ndim=2, shape[1]==252.
        CQT representation, with 36 bins per octave and 252 filters.
    kernel : np.ndarray
        Convolution kernel (should be roughly low-pass).

    Returns
    -------
    Z : np.ndarray
        The processed output.
    """
    if X.shape[-1] != 252:
        raise ValueError(
            "Apologies, but this method is currently designed for input "
            "representations with a last dimension of 252.")
    x_hp = highpass(X, kernel)
    x_73 = local_l2norm(x_hp, np.hanning(73).reshape(1, -1))
    x_37 = local_l2norm(x_hp, np.hanning(37).reshape(1, -1))
    x_19 = local_l2norm(x_hp, np.hanning(19).reshape(1, -1))
    x_multi = np.array([x_73, x_37, x_19]).transpose(1, 2, 0)
    w = _create_triband_mask()**2.0
    return (x_multi * w).sum(axis=-1)


def _create_triband_mask():
    """Build a summation mask for the octaves defined in Chapter 5.

    The resulting mask tensor looks (roughly) like the following, indexed by
    the final axis:
             __
          0 |  \__      |
          1 |  /  \_____|
          2 |     /     |

    Note: Again, this is admittedly ad hoc, and warrants attention in the
    future.

    Returns
    -------
    mask : np.ndarray, shape=(1, 252, 3)
        Sine-tapered summation mask to smoothly blend three representations
        with logarithmically increasing window widths.
    """
    w = np.sin(np.pi*np.arange(36)/36.)
    w_73 = np.zeros(252)
    w_37 = np.zeros(252)
    w_19 = np.zeros(252)

    w_73[:18] = 1.0
    w_73[18:36] = w[18:]

    w_37[18:36] = w[:18]
    w_37[36:72] = 1.0
    w_37[72:90] = w[18:]

    w_19[72:90] = w[:18]
    w_19[90:] = 1.0
    return np.array([w_73, w_37, w_19]).T.reshape(1, 252, 3)


def create_kernel(dim0, dim1):
    """Create a two-dimensional LPF kernel, with a half-Hamming window along
    the first dimension and a Gaussian along the second.

    Parameters
    ----------
    dim0 : int
        Half-Hamming window length.
    dim1 : int
        Gaussian window length.

    Returns
    -------
    kernel : np.ndarray
        The 2d LPF kernel.
    """
    dim0_weights = np.hamming(dim0 * 2 + 1)[:dim0]
    dim1_weights = gaussian(dim1, dim1 * 0.25, True)
    kernel = dim0_weights[:, np.newaxis] * dim1_weights[np.newaxis, :]
    return kernel / kernel.sum()
