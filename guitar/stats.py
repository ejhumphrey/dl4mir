import pychords.guitar as G


def count_chords(entity):
    counts = dict()
    for l in entity.fret_labels.value:
        if not l in counts:
            counts[l] = 0
        counts[l] += 1
    return counts


def add_dicts(a, b):
    x = a.copy()
    for key, value in b.iteritems():
        if not key in x:
            x[key] = 0
        x[key] += value
    return x


def tally_dataset(dset):
    counts = dict()
    for k in dset.keys():
        counts = add_dicts(counts, count_chords(dset.get(k)))
    return counts


"""
FRET_MAX = 11
train_counts = S.tally_dataset(train)
fret_histo = np.zeros([6, FRET_MAX])
for k, c in train_counts.iteritems():
    frets = np.array(G.decode(k))
    fret_histo[np.arange(6), frets] += c

fret_histo3 = np.zeros([6, FRET_MAX])
for k, c in train_counts.iteritems():
    print k
    frets = np.array(G.decode(k))
    if 0 in frets.tolist():
        fret_histo3[np.arange(6), frets] += c
        continue
    if frets.sum() == -6:
        fret_histo3[np.arange(6), frets] += c
        continue
    frets -= frets[frets>0].min()*np.ones(6, dtype=int)*np.greater(frets, 0)
    while frets.max() < (FRET_MAX-1):
        frets += np.ones(6, dtype=int)*np.greater_equal(frets, 0)
        fret_histo3[np.arange(6), frets] += c

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.imshow(h, interpolation='nearest',aspect='auto', vmin=0, vmax=1)
ax2.imshow(h2, interpolation='nearest',aspect='auto', vmin=0, vmax=1)
ax3.imshow(h3, interpolation='nearest',aspect='auto', vmin=0, vmax=1)
ax3.set_xlabel("Fret Number")

"""
