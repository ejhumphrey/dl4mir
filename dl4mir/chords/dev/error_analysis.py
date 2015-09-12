import biggie
import dl4mir.chords.lexicon as lex
import numpy as np


def extract_max_and_ref(stash, vocab):
    keys = stash.keys()
    (y_true, l_true, frame_idx,
        key_idx, y_max, l_max) = [list() for _ in range(6)]
    for ki, key in enumerate(keys):
        x = stash.get(key)
        y = vocab.label_to_index(x.chord_labels)
        for fi, yi in enumerate(y):
            if yi is None:
                continue
            y_true.append(yi)
            l_true.append(x.posterior[fi, yi])
            y_max.append(x.posterior[fi].argmax())
            l_max.append(x.posterior[fi].max())
            key_idx.append(ki)
            frame_idx.append(fi)

    return [np.array(_)
            for _ in y_true, l_true, y_max, l_max, key_idx, frame_idx]


def stash_to_arrays(stash, vocab):
    keys = stash.keys()
    y_true, posteriors, key_idx, frame_idx = [list() for _ in range(4)]
    for ki, key in enumerate(keys):
        x = stash.get(key)
        y = vocab.label_to_index(x.chord_labels)
        for fi, yi in enumerate(y):
            if yi is None:
                continue
            y_true.append(yi)
            posteriors.append(x.posterior[fi])
            key_idx.append(ki)
            frame_idx.append(fi)

    return [np.array(_) for _ in y_true, posteriors, key_idx, frame_idx]


"""
trackwise_moia_likelihoods = np.array([moia_likelihoods[np.equal(kidx, i)].mean() for i in range(840)])
x2[np.arange(len(x)), y_true] -= x2.max()*2



estimations = json.load(open(est_files[8]))
for k,v in estimations.iteritems():
    results[k] = SE.compute_scores({k:v}, vocab)

keys = estimations.keys()
metrics = results[k][0].keys()
metrics.sort()
stats = np.array([[results[k][0][m] for m in metrics] for k in keys])
rw_idx = stats[:,5].argsort()
support = np.array([results[k][1].sum() for k in keys])

for i in rw_idx[:40]:
    if support[i]/20.0 > 30.0:
        print i, keys[i], support[i], stats[i]
"""
