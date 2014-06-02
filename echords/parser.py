import argparse
import json
import time
import glob
import numpy as np
import pychords.guitar as G
import pychords.lib as C
import mir_eval
import os.path as path
import marl.fileutils as futils


CHORD_KEY = "chords = new Array()"
VARIATIONS = '.variations'
NO_FRETS = ','.join(["X"] * 6)
TAB_CHORDS = 'tab_chords'
DISTANCE = 'distance'


def parse_chord_js(line):
    chords = dict()
    js_lines = line.split(CHORD_KEY)[-1].split(";")
    for l in js_lines:
        if VARIATIONS in l:
            exec(l.replace(VARIATIONS, ''))
    for k in chords:
        chords[k] = chords[k].split(" ")
    return chords


def load_echords(html_file):
    fh = open(html_file)
    for line in fh:
        if CHORD_KEY in line:
            return parse_chord_js(line)


def align_echords_with_lab(html_file, lab_file):
    tab_data = load_echords(html_file)
    # Assume the tab doesn't include the no-chord.
    tab_data[C.NO_CHORD] = [NO_FRETS]
    tab_chords = tab_data.keys()
    tab_frets = [G.decode(tab_data[l][0]) for l in tab_chords]
    chord_map = dict()
    for lab_chord in set(mir_eval.io.load_intervals(lab_file)[1]):
        dist = chroma_distance(lab_chord, tab_frets)
        sidx = dist.argsort()
        idx = None
        for i in sidx:
            if lab_chord.startswith(tab_chords[i][0]):
                idx = i
                break
        if idx is None:
            idx = sidx[0]
        chord_map[lab_chord] = dict(
            chord_label=tab_chords[idx],
            fret_label=tab_data[tab_chords[idx]][0],
            distance=dist.min())
    chord_map[TAB_CHORDS] = tab_chords
    return chord_map


def chroma_distance(chord_label, frets):
    chord_chroma = C.chord_to_chroma(chord_label)
    fret_chroma = np.array([G.frets_to_chroma(x) for x in frets])
    return np.abs(chord_chroma[np.newaxis, :] - fret_chroma).sum(axis=1)


def translate_labels(labels, chord_map, mismatch_threshold=2):
    return [C.X_CHORD if chord_map[l][DISTANCE] >= mismatch_threshold
            else chord_map[l]['fret_label'] for l in labels]


def main(args):
    html_dir = path.join(args.html_directory, "*.html")
    lab_dir = path.join(args.lab_directory, "*.lab")
    html_files = dict([(futils.filebase(f), f)
                       for f in glob.glob(html_dir)])
    lab_files = dict([(futils.filebase(f), f)
                      for f in glob.glob(lab_dir)])
    futils.create_directory(args.output_directory)
    for key in html_files:
        if not key in lab_files:
            continue
        chord_map = align_echords_with_lab(html_files[key], lab_files[key])
        intervals, labels = mir_eval.io.load_intervals(lab_files[key])
        fret_labels = translate_labels(
            labels, chord_map, args.mismatch_threshold)
        output_file = path.join(args.output_directory, "%s.tab" % key)
        with open(output_file, 'w') as fp:
            json.dump(dict(intervals=intervals.tolist(),
                           labels=fret_labels), fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a set of queries and write the results as JSON.")
    parser.add_argument(
        "html_directory",
        metavar="html_directory", type=str,
        help="Directory containing tab html files.")
    parser.add_argument(
        "lab_directory",
        metavar="lab_directory", type=str,
        help="Directory containing lab files.")
    parser.add_argument(
        "output_directory",
        metavar="output_directory", type=str,
        help="Directory containing output tab files.")
    parser.add_argument(
        "--mismatch_threshold",
        metavar="mismatch_threshold", type=int, default=2,
        help="Directory containing output tab files.")
    main(parser.parse_args())
