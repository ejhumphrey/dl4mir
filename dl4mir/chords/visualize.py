import numpy as np

from matplotlib.pyplot import figure, subplot
from matplotlib import gridspec
import mir_eval
from scipy.spatial import distance
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


def plot_validation_sweep(validation_stats, pvals, iter_idx=-4, ax=None,
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


def generate_colorspace(num_colors):
    x = np.linspace(0.5, 0.9, 5)
    colorspace = [_.flatten()
                  for _ in np.meshgrid(x, x, x, xindexing='ij')]
    colorspace = np.array(colorspace).T
    best_dist = 0.0
    best_colorspace = None
    for n in range(50):
        np.random.shuffle(colorspace)
        dist_mat = distance.cdist(*([colorspace[:num_colors]]*2))
        if dist_mat.mean() > best_dist:
            best_dist = dist_mat.mean()
            best_colorspace = np.array(colorspace[:num_colors])
    return best_colorspace


def plot_chord_regions(index_map, vocab, colorspace=None):
    X = np.zeros(list(index_map.shape) + [3], dtype=float)
    uidx = np.unique(index_map).tolist()
    for holdout in 156, None:
        if holdout in uidx:
            uidx.remove(holdout)

    if colorspace is None:
        colorspace = generate_colorspace(len(uidx))

    for n, i in enumerate(uidx):
        X[index_map == i] = colorspace[n]
    X[index_map == 156] = 0.25, 0.25, 0.25
    X[np.equal(index_map, None)] = 1.0, 1.0, 1.0

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    ax = subplot(gs[0])
    ax.imshow(X, aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time")

    legend = subplot(gs[1])
    for n in range(len(uidx)):
        legend.bar(n, 1.0, color=colorspace[n], width=1.0)

    legend.set_xticks(np.arange(len(uidx)) + 0.5)
    legend.set_xticklabels(vocab.index_to_label(uidx))
    legend.set_xlim(0, len(uidx))
    legend.set_yticks([])
    return ax, legend


def plot_labeled_intervals(intervals, labels, colorspace=None):
    time_points, labels = mir_eval.util.intervals_to_samples(intervals, labels)
    X = np.zeros([1, len(labels), 3], dtype=float)
    unique_labels = np.unique(labels).tolist()
    for holdout in L.NO_CHORD, L.SKIP_CHORD:
        if holdout in unique_labels:
            unique_labels.remove(holdout)

    if colorspace is None:
        colorspace = generate_colorspace(len(unique_labels))

    for idx, l in enumerate(unique_labels):
        X[0, util.equals_value(labels, l)] = colorspace[idx]
    X[0, util.equals_value(labels, L.NO_CHORD)] = 0.25, 0.25, 0.25
    X[0, util.equals_value(labels, L.SKIP_CHORD)] = 1.0, 1.0, 1.0

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    ax = subplot(gs[0])
    ax.imshow(X, aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time")

    legend = subplot(gs[1])
    for n in range(len(unique_labels)):
        legend.bar(n, 1.0, color=colorspace[n], width=1.0)

    legend.set_xticks(np.arange(len(unique_labels)) + 0.5)
    legend.set_xticklabels(unique_labels)
    legend.set_xlim(0, len(unique_labels))
    legend.set_yticks([])
    return ax, legend


def cqt_compare(a, b, fig=None, cmap='jet'):
    if fig is None:
        fig = figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for ax, x in zip([ax1, ax2], [a, b]):
        ax.imshow(x.T, interpolation='nearest', aspect='auto', origin='lower',
                  cmap=cmap)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Time")


def macro_vs_micro_scatter(tmc_scores, deep_net_scores):
    """Plot macro versus class-micro statistics.

    Parameters
    ----------
    tmc_scores : np.ndarray, shape=(2,)
        Baseline scores.
    deep_net_scores : np.ndarray, shape=(2, 4, 3)
        Scores to plot for comparison.

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The axes handle of the figure.
    """
    fig = figure()
    ax = fig.gca()
    r = (tmc_scores ** 2.0).sum()**0.5
    ax.scatter(tmc_scores[0, 0], tmc_scores[1, 0], marker='x', color='k', s=50)
    for j, m in enumerate(['o', '^', 's']):
        for i, c in enumerate(['r', 'g', 'b', 'y']):
            ax.scatter(deep_net_scores[0, i, j],
                       deep_net_scores[1, i, j],
                       marker=m, color=c, s=50)

    n = np.linspace(0, np.pi/2.0, 1000)
    ax.plot(r*np.cos(n), r*np.sin(n), 'k--')
    # ax.set_xlim(0.55, 0.8)
    # ax.set_ylim(0.35, 0.6)
    ax.set_ylabel("Recall - Averaged")
    ax.set_xlabel("Recall - Weighted")
    return ax
