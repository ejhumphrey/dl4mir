import numpy as np

import pescador

from dl4mir.common import util
from dl4mir.guitar import VOICINGS
import dl4mir.guitar.fretutil as futil


def cqt_sampler(key, stash, win_length=20, index=None, max_samples=None,
                sample_func=util.slice_cqt_entity):
    """Generator for sampling windowed cqt observations from an entity.

    Parameters
    ----------
    key : str
        Key for the entity of interest; must be consistent across both `stash`
        and `index`, when the latter is provided.
    stash : dict_like
        Dict or biggie.Stash of chord entities.
    win_length: int
        Length of centered observation window for the CQT.
    index: dict of index arrays, default=None
        Indexing object for constrained sampling of the chord entity.
        If provided, must have a np.ndarray of integers under `key`; otherwise,
        this method will fail.
    max_samples: int, or None
        Maximum number of samples to return from this Generator; if None, runs
        indefinitely.

    Yields
    ------
    sample: biggie.Entity with fields {cqt, chord_label}
        The windowed chord observation.
    """
    entity = stash.get(key)
    num_samples = len(entity.chord_labels)

    if index is None:
        index = {key: np.arange(num_samples)}

    valid_samples = index.get(key, [])
    idx = np.inf
    max_samples = np.inf if max_samples is None else max_samples
    count = 0
    while count < max_samples and len(valid_samples):
        if idx >= len(valid_samples):
            np.random.shuffle(valid_samples)
            idx = 0
        yield sample_func(entity, win_length, valid_samples[idx])
        idx += 1
        count += 1


def create_fret_stream(stash, win_length, working_size=50, voicings=VOICINGS,
                       sample_func=util.slice_cqt_entity):
    """Return an unconstrained stream of chord samples with class indexes.

    Parameters
    ----------
    stash : biggie.Stash
        A collection of chord entities.
    win_length : int
        Length of a given tile slice.
    working_size : int
        Number of open streams at a time.

    Returns
    -------
    stream : generator
        Data stream of windowed cqt-fret entities.
    """
    entity_pool = [pescador.Streamer(cqt_sampler, key, stash,
                                     win_length, sample_func=sample_func)
                   for key in stash.keys()]

    stream = pescador.mux(entity_pool, None, working_size, lam=25)
    return futil.fret_mapper(stream, voicings)
