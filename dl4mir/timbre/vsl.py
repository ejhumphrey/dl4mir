"""Tools to manipulate VSL data collection."""

import numpy as np
import glob
import os
import re
import marl.fileutils as futil
from scipy.spatial.distance import cdist


def _pitch_classes():
    """Map from pitch class (str) to semitone (int)."""
    pitch_classes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    semitones = [0, 2, 4, 5, 7, 9, 11]
    return dict([(c, s) for c, s in zip(pitch_classes, semitones)])

# Maps pitch classes (strings) to semitone indexes (ints).
PITCH_CLASSES = _pitch_classes()


def pitch_class_to_semitone(pitch_class):
    """Convert a pitch class to semitone.

    Parameters
    ----------
    pitch_class: str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'

    Returns
    -------
    semitone: int
        Semitone value of the pitch class.

    Raises
    ------
    InvalidFormatException
    """
    semitone = 0
    for idx, char in enumerate(pitch_class):
        if char == '#' and idx > 0:
            semitone += 1
        elif char == 'b' and idx > 0:
            semitone -= 1
        elif idx == 0:
            semitone = PITCH_CLASSES.get(char)
        else:
            raise InvalidFormatException(
                "Pitch class improperly formed: %s" % pitch_class)
    return semitone % 12


def collect_single_notes(basedir):
    # en = "Einzelnoten"
    en = "*"
    filepaths = []
    for n in range(6):
        subpattern = "*/" + "*/" * n + "*.wav"
        fpath = os.path.join(basedir, subpattern)
        print fpath
        filepaths += glob.glob(fpath)

    return filepaths


def file_to_instrument_code(filename):
    return re.split('[\W_]+', futil.filebase(filename))[0]


def file_to_instrument_code2(filename):
    return re.split('[\W_]+', filename.split('/')[-2])[0]


_reduced_set = {
    "AFL": ['AFLmV', 'AFLmV0leg',],
    "FL1": ['FL1RR', 'FL1RRstac', 'FL1mV',
            'FL1mV0leg', 'FL1mV0po', 'FL1mVleg'],
    "FL2": ['FL2RR', 'FL2RRst', 'FL2mV', 'FL2mV0leg', 'FL2mVleg'],
    "PFL": ['PFL0st', 'PFLmV', 'PFLmV0leg', 'PFLsta'],
    "MA": ['MAtrem'],
    "VA": ['A'],
    "AKG": ['AkG'],
    'HO': ['Ho'],
    'Tmp': ['TmpME'],
    'SXS': ['SXSstac'],
    "BT": ['Bt'],
    "BP": ['Bp'],
    "CI": ['Ci'],
    "Tr": ['TR'],
    "TrC": ['TRC', 'TrCRR'],
    "WT": ['WTB'],
    "VC": ['VCmVcres2']}


def group_by_instrument(file_list):
    base_set = dict()
    for f in file_list:
        key = file_to_instrument_code(f)
        if not key in base_set:
            base_set[key] = list()
        base_set[key].append(f)

    vsl_set = dict()
    for base_key in base_set:
        for real_key, redux in _reduced_set.items():
            if base_key in redux:
                base_key = real_key
                break
        if not base_key in vsl_set:
            vsl_set[base_key] = list()
        vsl_set[base_key] += base_set[base_key]
    return vsl_set


_instrument_codes = ['FL1', 'HO', 'TrC', 'VC', 'VI', 'BP', 'BT', 'VA', 'CI',
                     'Tr', 'WT', 'TP', 'PFL', 'KLB', 'AFL', 'KB', 'TU', 'FA',
                     'EG', 'SXS', 'AKG', 'PT', 'PO', 'OB']


def has_pitch(f):
    return False if re.match('.*([A-G]#?[0-9])',
                             futil.filebase(f)) is None else True


def contains(s, invert=False):
    return lambda x: x.count(s) > 0 if invert is False else x.count(s) == 0


def filter_abberations(files):
    subset = filter(has_pitch, files)
    tags = ['GLISSANDI', "Percuss", "CHORD", 'Gong', 'glis', 'run']
    for t in tags:
        subset = filter(contains(t, True), subset)

    return subset


def note_name_to_number(note_name):
    octave = int(note_name[-1])
    pitch_class = pitch_class_to_semitone(note_name[:-1])
    return octave*12 + pitch_class


def file_to_note_number(filename):
    matches = re.findall('.*([A-G]#?[0-9])', futil.filebase(filename))
    if len(matches) == 0:
        return None
    return note_name_to_number(matches[0].strip(".wav"))


def remove_outliers(filelist, pitch_histograms,
                    threshold=0.4, note_min=16, note_max=84):
    note_nums = np.array([file_to_note_number(f) for f in filelist])
    idxs = [(n, np.equal(note_nums, n)) for n in range(note_min, note_max + 1)]
    valid_files = []
    for note_num, idx in idxs:
        hist = pitch_histograms[idx]
        subfiles = np.asarray(filelist)[idx]
        assert hist.shape[0] > 500
        target = hist.mean(axis=0)[np.newaxis, :]
        dist = cdist(hist, target, 'cosine').flatten()
        for f in subfiles[dist <= threshold]:
            valid_files.append(f)

    return valid_files
