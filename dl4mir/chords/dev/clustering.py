import argparse
import joblib
import numpy as np
import os
import time

import biggie
import pescador
import ml_scraps.HartiganOnline as HOL

import dl4mir.chords.pipefxs as FX
from dl4mir.common import util
import dl4mir.chords.data as D
import dl4mir.common.streams as S

NUM_CPUS = 8
QUANTIZERS = dict()


def compute_chord_averages(stash, win_length=20, num_obs=5000):
    quality_partition = util.partition(stash, quality_map)
    qual_indexes = [util.index_partition_arrays(quality_partition, [q])
                    for q in range(13)]
    qual_pools = [[pescador.Streamer(chord_sampler, key, stash, 20, q_idx)
                   for key in q_idx] for q_idx in qual_indexes]
    obs_aves = []
    for pool in qual_pools:
        base_stream = pescador.mux(pool, n_samples=None, k=50, lam=5)
        for root in range(12):
            stream = FX.rotate_chord_to_root(base_stream, root)
            x_obs = np.array([stream.next().cqt for _ in range(num_obs)])
            obs_aves.append(x_obs.mean(axis=0).squeeze())
            print len(obs_aves)

    null_index = util.index_partition_arrays(quality_partition, [13])
    null_pool = [pescador.Streamer(chord_sampler, key, stash, 20, null_index)
                 for key in null_index]
    stream = pescador.mux(null_pool, n_samples=None, k=50, lam=5)
    x_obs = np.array([stream.next().cqt for _ in range(num_obs)])
    obs_aves.append(x_obs.mean(axis=0).squeeze())
    return np.array(obs_aves)


def sample_chord_qualities(stash, output_dir, win_length=20, num_obs=10000):
    quality_partition = util.partition(stash, quality_map)
    qual_indexes = [util.index_partition_arrays(quality_partition, [q])
                    for q in range(13)]
    qual_pools = [[pescador.Streamer(chord_sampler, key, stash, 20, q_idx)
                   for key in q_idx] for q_idx in qual_indexes]
    futil.create_directory(output_dir)
    print "[%s] Starting loop" % time.asctime()
    for qual, pool in enumerate(qual_pools):
        base_stream = pescador.mux(pool, n_samples=None, k=50, lam=5)
        for root in range(12):
            stream = FX.rotate_chord_to_root(base_stream, root)
            x_obs = np.array([stream.next().cqt for _ in range(num_obs)])
            chord_idx = qual*12 + root
            np.save(os.path.join(output_dir, "%03d.npy" % chord_idx), x_obs)
            print "[%s] %3d" % (time.asctime(), chord_idx)
    null_index = util.index_partition_arrays(quality_partition, [13])
    null_pool = [pescador.Streamer(chord_sampler, key, stash, 20, null_index)
                 for key in null_index]
    stream = pescador.mux(null_pool, n_samples=None, k=50, lam=5)
    x_obs = np.array([stream.next().cqt for _ in range(num_obs)])
    np.save(os.path.join(output_dir, "156.npy"), x_obs)


def fit_pcas(sample_files, num_components=2520):
    pcas = []
    base_statistics = []
    new_statistics = []
    for f in sample_files:
        x_obs = np.load(f)
        x_obs = x_obs.reshape(x_obs.shape[0], np.prod(x_obs.shape[1:]))
        base_statistics.append([x_obs.mean(axis=0), x_obs.std(axis=0)])
        pcas.append(PCA(num_components))
        z_rot = pcas[-1].fit_transform(x_obs)
        new_statistics.append((z_rot.mean(axis=0), z_rot.std(axis=0)))
        print "[%s] %3d" % (time.asctime(), len(pcas))

    return pcas, base_statistics, new_statistics


def chord_streamer(stash, win_length, partition_labels=None,
                   vocab_dim=157, working_size=4, valid_idx=None,
                   n_samples=5000, batch_size=50):
    """Return a stream of chord samples, with uniform quality presentation."""
    if partition_labels is None:
        partition_labels = util.partition(stash, D.chord_map)

    if valid_idx is None:
        valid_idx = range(vocab_dim)

    chord_pool = []
    chord_idx = []
    for idx in valid_idx:
        print "Opening %d ..." % idx
        subindex = util.index_partition_arrays(partition_labels, [idx])
        entity_pool = [pescador.Streamer(D.chord_sampler, key, stash,
                                         win_length, subindex)
                       for key in subindex.keys()]
        if len(entity_pool) == 0:
            continue
        stream = pescador.mux(
            entity_pool, n_samples=n_samples, k=working_size, lam=20)
        batch = S.minibatch(FX.map_to_chord_index(stream, vocab_dim),
                            batch_size=batch_size)
        chord_pool.append(batch)
        chord_idx.append(idx)
    print "Done!"
    return chord_pool, np.array(chord_idx)


def _update(quant, data):
    x = data['cqt']
    quant.partial_fit(x.reshape(len(x), np.prod(x.shape[1:])))
    return quant


def multi_fit(streams, n_clusters, n_iter):
    quants = [HOL.HartiganOnline(n_clusters=n_clusters, max_iter=np.inf)
              for idx in range(len(streams))]

    pool = joblib.Parallel(n_jobs=NUM_CPUS)
    for n in range(n_iter):
        try:
            quants = pool(joblib.delayed(_update)(q, s.next())
                      for q, s in zip(quants, streams))
            print "[%s] Iter: %4d" % (time.asctime(), n)
        except KeyboardInterrupt:
            print "Stopping Early..."
            break
    return quants


def main(args):
    stash = biggie.Stash(args.training_file)
    valid_idx = range(157)
    streams, chord_idx = chord_streamer(
        stash, win_length=20, working_size=4, valid_idx=valid_idx,
        n_samples=None, batch_size=50)

    quants = multi_fit(streams, args.n_clusters, args.n_iter)
    centers = np.array([q.cluster_centers_ for q in quants])
    counts = np.array([q.cluster_sizes_ for q in quants], dtype=int)
    np.savez(args.output_file, centers=centers, counts=counts,
             chord_idx=chord_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("training_file",
                        metavar="training_file", type=str,
                        help="Path to an optimus file of chord posteriors.")
    parser.add_argument("n_clusters",
                        metavar="n_clusters", type=int,
                        help="")
    parser.add_argument("n_iter",
                        metavar="n_iter", type=int,
                        help="")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to an numpy archive output.")
    main(parser.parse_args())
