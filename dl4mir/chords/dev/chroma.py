import numpy as np
import dirichlet
from scipy.special import gammaln
import pescador
import sklearn.metrics as metrics
from sklearn.mixture import GMM

from marl.utils.matrix import circshift

import dl4mir.common.util as util
import dl4mir.chords.data as D
import dl4mir.chords.pipefxs as FX


def loglikelihood(D, a):
    '''Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).

    Parameters
    ----------
    D : 2D array
        where ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a : array
        Parameters for the Dirichlet distribution.

    Returns
    -------
    logl : float
        The log likelihood of the Dirichlet distribution'''
    a_sum = gammaln(a.sum())
    gamma_sum = gammaln(a).sum()
    return a_sum - gamma_sum + ((a - 1)*np.log(D)).sum(axis=1)


def load_dataset(stash, ):
    partition_labels = util.partition(stash, D.quality_map)
    qual_obs = []
    for q in range(13):
        qindex = util.index_partition_arrays(partition_labels, [q])
        entity_pool = [pescador.Streamer(D.chroma_stepper, k, stash, qindex)
                       for k in qindex]
        stream = pescador.mux(entity_pool, n_samples=None, k=50, lam=None,
                              with_replacement=False)
        obs = np.array([x for x in FX.rotate_chroma_to_root(stream, 0)])
        qual_obs.append(util.normalize(obs, axis=1))

    return qual_obs


def train_dirichlet(qual_obs):
    alphas = np.array([dirichlet.mle(obs) for obs in qual_obs])
    alphas = np.concatenate([circshift(alphas, 0, r)
                             for r in range(12)], axis=0)
    logl = [np.array([loglikelihood(obs, a) for a in alphas]).T
            for obs in qual_obs]
    y_pred = np.concatenate([logl[n].argmax(axis=1) for n in range(13)])
    y_true = np.concatenate([np.array([n]*len(logl[n])) for n in range(13)])
    print score(y_true, y_pred)


def train_gmm(qual_obs, num_components=5, covariance_type='diag'):
    gmms = [GMM(num_components, covariance_type=covariance_type).fit(obs)
            for obs in qual_obs]
    logl = [np.concatenate([np.array([m.score(circshift(obs, 0, r))
                                      for m in gmms])
                            for r in range(12)], axis=0).T
            for obs in qual_obs]
    y_pred = np.concatenate([logl[n].argmax(axis=1) for n in range(13)])
    y_true = np.concatenate([np.array([n]*len(logl[n])) for n in range(13)])
    print score(y_true, y_pred)
    return gmms


def score(y_true, y_pred):
    precision_weighted = metrics.precision_score(
        y_true, y_pred, average='weighted')
    precision_ave = np.mean(metrics.precision_score(
        y_true, y_pred, average=None)[::12])

    recall_weighted = metrics.recall_score(
        y_true, y_pred, average='weighted')
    recall_ave = np.mean(metrics.recall_score(
        y_true, y_pred, average=None)[::12])

    f1_weighted = metrics.f1_score(
        y_true, y_pred, average='weighted')
    f1_ave = np.mean(metrics.f1_score(
        y_true, y_pred, average=None)[::12])

    stat_line = "  Precision: %0.4f\t Recall: %0.4f\tf1: %0.4f"
    res1 = "Weighted: " + stat_line % (100*precision_weighted,
                                       100*recall_weighted,
                                       100*f1_weighted)

    res2 = "Averaged: " + stat_line % (100*precision_ave,
                                       100*recall_ave,
                                       100*f1_ave)
    res3 = "-"*72
    outputs = [res3, res1, res2, res3]
    return "\n".join(outputs)


def flatten_features(qual_obs):
    x, y = list(), list()
    for q, obs in enumerate(qual_obs):
        for r in range(12):
            x.append(circshift(obs, 0, r))
            y.append(len(obs)*[r + q*12])

    return np.concatenate(x, axis=0), np.concatenate(y)