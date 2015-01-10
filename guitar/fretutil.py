"""Routines for managing guitar chord shapes.

Note: The concept of 'off' is represented numerically by -1. This should allow
for graceful wrap-around in bitmap representations (off is the last index).


Conventions
-----------
tab : str
    String representation of active guitar notes.
frets : list of ints
    Numerical representation of the active frets.
notes : list of ints
    Numerical representation of active note numbers per string.
"""

import numpy as np
import biggie
import music21

#       Strings    E2  A2  D3  G3  B3  E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
OFF_CHAR = 'X'
NO_CHORD = ','.join([OFF_CHAR] * 6)


def decode(tab, delimiter=',', off_char=OFF_CHAR):
    """Decode a tab to fret numbers.

    Parameters
    ----------
    tab : str
        Tab-formatted guitar representation.
    delimiter : str
        Spacer for tab format.
    off_char : str
        Character used for non-played strings.

    Returns
    -------
    frets : list
        Integers of active frets; -1 indicates 'off'.
    """
    tab = tab.upper()
    frets = []
    for x in tab.split(delimiter):
        x = x.strip("P ")
        frets.append(-1 if x == off_char else int(x))
    return frets


def encode(frets, delimiter=',', off_char='X'):
    """
    Parameters
    ----------
    frets : list
        Integers of active frets; -1 indicates 'off'.
    delimiter : str
        Spacer for tab format.
    off_char : str
        Character used for non-played strings.

    Returns
    -------
    tab : str
        Tab-formatted guitar representation.
    """
    tab = list(frets)
    for n in range(len(tab)):
        tab[n] = str(tab[n]) if n >= 0 else off_char
    return delimiter.join(tab)


def frets_to_chroma(frets):
    """
    Parameters
    ----------
    frets: array_like
        Integer representation of active frets.

    Returns
    -------
    chroma: np.ndarray, shape=(12,)
        Chroma bitvector of active pitch classes.
    """
    chroma = np.zeros(12, dtype=int)
    for x, s in zip(frets, STANDARD_TUNING):
        if x >= 0:
            chroma[(x + s) % 12] = 1
    return chroma


def frets_to_note_number(frets):
    """Translate a fret array to MIDI note numbers.

    Parameters
    ----------
    frets: array_like
        Integer representation of active frets.

    Returns
    -------
    note_numbers : list
        Active note numbers.
    """
    note_numbers = list()
    for x, s in zip(frets, STANDARD_TUNING):
        if x >= 0:
            note_numbers.append(x + s)
    return note_numbers


def fret_mapper(stream, voicings, num_frets=9):
    """Stream filter for mapping chord label entities to frets.

    Parameters
    ----------
    stream : generator
        Yields {cqt, chord_label} entities, or None.
    voicings : dict
        Map of chord labels to tab strings.

    Yields
    ------
    entity : biggie.Entity
        Fretted entity with {cqt, frets}, or None if out of gamut.
    """
    for entity in stream:
        if entity is None:
            yield entity
        tab = voicings.get(str(entity.chord_label), None)
        if tab is None:
            yield None
        else:
            frets = {'{0}_index'.format(s): i % num_frets
                     for s, i in zip('EADGBe', decode(tab))}
            yield biggie.Entity(cqt=entity.cqt, **frets)


DEGREES = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']


def note_numbers_to_interval_chord(note_numbers):
    c = music21.chord.Chord(note_numbers)
    intervals = ",".join([DEGREES[_] for _ in c.normalForm])
    return "{0}:({1})".format(c.root().name, intervals)


def frets_to_interval_chord(frets):
    note_numbers = frets_to_note_number(frets)
    if not note_numbers:
        return "N"
    return note_numbers_to_interval_chord(note_numbers)


class GuitarChords(dict):
    def __call__(self, frets):
        frets = tuple(frets)
        if not frets in self:
            self[frets] = frets_to_interval_chord(frets)
        return self[frets]


class GuitarTabs(dict):
    def __call__(self, frets):
        frets = tuple(frets)
        if not frets in self:
            self[frets] = encode(frets)
        return self[frets]


ENCODERS = {
    'chords': GuitarChords(),
    'tabs': GuitarTabs()}
