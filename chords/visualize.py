import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show
import dl4mir.chords.labels as L
import dl4mir.common.util as util


COLORS = [
    [0.75,  0.1,  0.1],
    [0.75,  0.75,  0.1],
    [0.75,  0.1,  0.4],
    [0.4,  0.1,  0.75],
    [0.1,  0.75,  0.4],
    [0.1,  0.75,  0.75],
    [0.1,  0.1,  0.75],
    [0.1,  0.4,  0.75],
    [0.1,  0.75,  0.1],
    [0.75,  0.1,  0.75],
    [0.4,  0.75,  0.1],
    [0.75,  0.4,  0.1],
    [0.,   0.,   0.]]

MARKERS = (
    'v',  # maj
    'o',  # min
    '^',  # maj7
    '8',  # min7
    '<',  # 7
    '>',  # maj6
    'p',  # min6
    's',  # dim
    '*',  # aug
    'd',  # sus2
    'D',  # sus4
    'h',  # dim7
    'H',  # hdim7
    'x')  # no-chord

IMSHOW_ARGS = dict(interpolation='nearest', aspect='auto', origin='lower')


def colored_marker(index):
    if index is None:
        return dict(c=np.zeros(3), marker='x')
    return dict(c=COLORS[index % 12], marker=MARKERS[int(index) / 12])


def scatter_chords(x, y):
    assert x.shape[1] == 3
    assert len(x) == len(y)
    fig = figure()
    ax = fig.gca(projection='3d')
    for chord_idx in np.unique(y):
        xc = x[chord_idx == y].T
        ax.scatter3D(xc[0], xc[1], xc[2], **colored_marker(chord_idx))


def scatter_6d(x, y, draw_legend=True):
    assert x.shape[1] == 6
    assert len(x) == len(y)
    fig = figure()
    for n in range(3):
        ax = fig.add_subplot(131 + n)
    for chord_idx in np.unique(y):
        xc = x[chord_idx == y, 2*n:2*(n+1)].T
        ax.scatter(xc[0], xc[1], **colored_marker(chord_idx))
    if draw_legend:
        legend()


def legend(size=100):
    fig = figure()
    ax = fig.gca()
    for row, m in enumerate(MARKERS[:-1]):
        for col, c in enumerate(COLORS):
            ax.scatter(col, row, c=c, marker=m, s=size)

    ax.set_xticks(range(12))
    ax.set_xticklabels(L.ROOTS)
    ax.set_yticks(range(13))
    ax.set_yticklabels(L.QUALITIES[157][:-1])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 12.5)


def draw_posterior_lazy(entity, vocab_dim=157, **viterbi_args):
    chord_idx = L.chord_label_to_class_index(entity.chord_labels, vocab_dim)
    pred_idx = util.viterbi(entity.posterior, **viterbi_args)
    draw_posterior(entity.posterior, chord_idx, pred_idx)


def draw_posterior(posterior, chord_idx, pred_idx):
    fig = figure()
    ax = fig.gca()
    ax.imshow(posterior.T, **IMSHOW_ARGS)
    x_max, y_max = posterior.shape
    ax.plot(chord_idx, 'k', linewidth=3, alpha=0.66)
    ax.plot(pred_idx, 'w', linewidth=3, alpha=0.66)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time")
    ax.set_ylabel("Chord Index")


def filter_empty_values(obj):
    result = dict()
    for k in obj:
        if obj[k]:
            result[k] = obj[k]
    return result


def sort_pvals(pvals):
    pidx = np.argsort(np.array(pvals, dtype=float))
    return [pvals[i] for i in pidx[::-1]]


def plot_validation_sweep(validation_stats, pvals=None, iter_idx=-4, ax=None,
                          metric='recall'):
    if ax is None:
        fig = figure()
        ax = fig.gca()

    stats = dict()
    for k in validation_stats:
        if validation_stats[k]:
            stats[k] = validation_stats[k]

    keys = stats.keys()
    keys.sort()

    if pvals is None:
        pvals = sort_pvals(stats[keys[0]].keys())

    n_iter = [int(k.split('-')[iter_idx]) for k in keys]

    for p, c in zip(pvals, COLORS):
        rw = [stats[k][p]['%s_weighted' % metric] for k in keys]
        ax.plot(n_iter, rw, color=c)
        ra = [stats[k][p]['%s_averaged' % metric] for k in keys]
        ax.plot(n_iter, ra, '--', color=c)

    ax.hlines(0.5785, 0, max(n_iter), 'k')
    ax.hlines(0.6241, 0, max(n_iter), 'k', linewidth=2)
    ax.hlines(0.4731, 0, max(n_iter), 'k', linestyles='dashed', linewidth=2)
    ax.hlines(0.4509, 0, max(n_iter), 'k', linestyles='dashed')

    ax.set_ylabel("Iteration")
    ax.set_ylabel("Recall")
    ax.set_xlim(min(n_iter), max(n_iter))
    ax.set_ylim(0.25, 0.75)

    return ax


def stats_to_matrix(validation_stats):
    stats = filter_empty_values(validation_stats)
    keys = stats.keys()
    keys.sort()

    pvals = sort_pvals(stats[keys[0]].keys())
    metrics = stats[keys[0]].values()[0].keys()
    metrics.sort()
    return np.array([[[stats[k][p][m] for m in metrics] for p in pvals]
                     for k in keys])


def draw_chord_boundaries(entity, ax, ymin, ymax):
    labels = np.array(util.run_length_encode(entity.chord_labels))
    boundaries = np.array([0] + np.cumsum(labels[:, 1].astype(int)).tolist())
    ax.vlines(boundaries, ymin=ymin, ymax=ymax, linewidth=3,
              color='k', alpha=0.66)
    ax.set_xticks((boundaries[:-1] + (np.diff(boundaries) / 2)).tolist())
    ax.set_xticklabels(labels[:, 0].tolist(), rotation=-45.0)
    return ax


def plot_chroma(entity, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.gca()
    ax.imshow(entity.chroma.T, interpolation='nearest',
              aspect='auto', origin='lower')
    ax.set_yticks(range(12))
    ax.set_yticklabels(L.ROOTS)
    return draw_chord_boundaries(entity, ax, ymin=-0.5, ymax=11.5)


def plot_posterior(entity, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.gca()
    ax.imshow(entity.posterior.T, interpolation='nearest',
              aspect='auto', origin='lower')
    return draw_chord_boundaries(entity, ax, ymin=-0.5, ymax=155.5)


def plot_cqt(entity, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.gca()
    ax.imshow(entity.cqt[0].T, interpolation='nearest',
              aspect='auto', origin='lower')
    return draw_chord_boundaries(entity, ax, ymin=-0.5, ymax=251.5)


def plot_piano_roll(entity, ax=None):
    if ax is None:
        fig = figure()
        ax = fig.gca()
    ax.imshow(entity.pitch.T, interpolation='nearest',
              aspect='auto', origin='lower')
    return draw_chord_boundaries(entity, ax, ymin=-0.5, ymax=84.5)


"""
plt.plot(a_iter, a[0,5,:], 'g');plt.plot(a_iter, a[1,5,:], 'g--')
plt.plot(g_iter, g[0,4,:], 'b');plt.plot(g_iter, g[1,4,:], 'b--')
plt.plot(d_iter, d[0,1,:], 'c');plt.plot(d_iter, d[1,1,:], 'c--')
plt.plot(e_iter, e[0,0,:], 'm');plt.plot(e_iter, e[1,0,:], 'm--')
plt.plot(f_iter, f[0,-3,:], 'y');plt.plot(f_iter, f[1,-3,:], 'y--')

"""
