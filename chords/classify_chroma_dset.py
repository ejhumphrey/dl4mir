"""Apply a graph convolutionally to datapoints in a Biggie Stash."""

import argparse
import numpy as np
import mir_eval.chord as chord_eval
import biggie
import os
import marl.fileutils as futil
import time
from scipy.spatial.distance import cdist

from dl4mir.chords import labels


def generate_templates(vocab=157):
    chord_labels = []
    for qual in labels.QUALITIES[vocab][:-1]:
        for root in labels.ROOTS:
            chord_labels.append("%s:%s" % (root, qual))
    chord_labels.append("N")
    templates = np.zeros([vocab, 12])
    for idx, label in enumerate(chord_labels):
        root, semitones, bass = chord_eval.encode(label)
        templates[idx, :] = chord_eval.rotate_bitmap_to_root(semitones, root)
    templates[-1, :] = 0.0001
    return templates


def classify_chroma(entity, templates):
    """

    Parameters
    ----------
    posterior: np.ndarray
        Posteriorgram of chord classes.
    viterbi_penalty: scalar, in (0, inf)
        Self-transition penalty; higher values produce more "stable" paths.

    Returns
    -------
    predictions: dict
        Chord labels and dense count vectors.
    """
    data = entity.values
    posterior = 1.0 - cdist(data.pop("chroma"), templates, 'cosine')
    scalar = posterior.sum(axis=1)[:, np.newaxis]
    scalar[scalar == 0] = 1.0
    return biggie.Entity(posterior=posterior/scalar, **data)


def main(args):
    chromas = biggie.File(args.chroma_file)
    templates = generate_templates(157)
    futil.create_directory(os.path.split(args.posterior_file)[0])
    posteriors = biggie.File(args.posterior_file)
    for idx, key in enumerate(chromas.keys()):
        posteriors.add(key, classify_chroma(chromas.get(key), templates))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx,
                                        len(chromas), key)
    posteriors.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("chroma_file",
                        metavar="chroma_file", type=str,
                        help="Biggie Stash of chroma features.")
    # Outputs
    parser.add_argument("posterior_file",
                        metavar="posterior_file", type=str,
                        help="Biggie Stash of resulting likelihoods.")
    main(parser.parse_args())
