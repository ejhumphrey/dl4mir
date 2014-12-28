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
