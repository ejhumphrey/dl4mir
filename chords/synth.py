import argparse
import numpy as np
import glob
import json
import os
import marl
import random
import time
import marl.fileutils as futil

import dl4mir.common.util as util
# from multiprocessing import Pool

NUM_CPUS = 12


def normalize(x_n):
    return x_n / np.abs(x_n).max()


def hwr(x):
    return x * (x >= 0)


def flange(x_n, fs, gain=0.5, depth=0.001, delay=0.005, freq=5.0):
    N = len(x_n)
    n_delay = fs * delay
    n_depth = fs * depth
    offset = np.round(n_depth * np.sin(2*np.pi*np.arange(N)*freq/fs) + n_delay)
    dline = (np.arange(N) - offset).astype(int)
    return x_n + gain * x_n[hwr(dline)] * (dline > 0)


def load_many(filenames, samplerate, channels=1, normalize_signal=False):
    """Load several audio signals into a list.

    Parameters
    ----------
    filenames : list
        Audio filepaths to load
    samplerate : scalar
        Samplerate of the given signals.
    channels : int, default=1
        Number of channels for the signals.
    normalize : bool, default=True
        If True, scale each signal by its absolute-maximum value.

    Returns
    -------
    signals : list
        The loaded signals; each with shape=(num_samples, num_channels)
    """
    signals = []
    for f in filenames:
        x_n, fs = marl.audio.read(f, samplerate=samplerate, channels=channels)
        if normalize_signal:
            x_n = normalize(x_n)
        signals.append(x_n)
    return signals


def combine(signals, num_samples=None):
    """Sum a set of signals.

    Parameters
    ----------
    signals : list of np.ndarrays
        The input signals to combine.
    num_samples : int, default None
        Maximum

    Returns
    -------
    y_out : np.ndarray
        The eveloped output signal.
    """
    if num_samples is None:
        num_samples = max([len(x_n) for x_n in signals])

    y_out = np.zeros([num_samples, x_n.shape[1]])
    for x_n in signals:
        y_out[:len(x_n)] += x_n

    return y_out


def envelope(x_n, samplerate, attack=0.01, decay=0.005, release=0.05,
             peak_level=1.0, sustain_level=1.0):
    """Apply an ADSR envelope to a 1d signal.

    Parameters
    ----------
    x_n : np.ndarray, ndim=1
        Input signal to window.
    samplerate : scalar
        Samplerate of the given signals.
    attack : scalar
        Rise time of the envelope.
    decay : scalar
        Fall time of the envelope, immediately following the attack.
    release : scalar
        Offset time of the envelope, applied to the end of the signal.
    peak_level : scalar
        Amplitude at the highest point of the attack.
    sustain_level : scalar
        Amplitude during the sustained portion of the signal.

    Returns
    -------
    y_out : np.ndarray
        The eveloped output signal.
    """
    duration = len(x_n) / float(samplerate)
    attack_env = np.linspace(0, peak_level, int(attack * samplerate))
    decay_env = np.linspace(peak_level, sustain_level, int(decay * samplerate))
    sustain = max([0, duration - sum([attack, decay, release])])
    sustain_env = np.ones(int(sustain * samplerate)) * sustain_level
    release_env = np.linspace(sustain_level, 0, int(release * samplerate))
    env = np.concatenate([attack_env, decay_env, sustain_env, release_env])
    L = min([len(x_n), len(env)])
    return x_n[:L] * env[:L][:, np.newaxis]


def sequence_signals(signals, intervals, samplerate=44100,
                     duration=None, env_args=None):
    """Window and place a set of signals into a buffer.

    Parameters
    ----------
    signals : list of np.ndarrays
        Audio signals to place in time.
    intervals : np.ndarray
        Start and end times for each signal.
    samplerate : scalar
        Samplerate of the given signals.
    duration : scalar, default=None
        Total duration, must be >= intervals.max(); otherwise, defaults to
        intervals.max().
    env_args: dict
        Keyword arguments for the evelope function.

    Returns
    -------
    y_out : np.ndarray
        The combined output signal.
    """
    assert len(intervals) == len(signals)
    env_args = dict() if env_args is None else env_args
    durations = np.abs(np.diff(intervals, axis=1))
    if duration is None:
        duration = intervals.max()
    else:
        assert duration >= intervals.max()
    num_samples = int(samplerate * duration)
    num_channels = None
    for x_n in signals:
        if hasattr(x_n, 'shape'):
            num_channels = x_n.shape[1]
            break
    if num_channels is None:
        raise ValueError("All signals are empty? wtf?")
    output_buffer = np.zeros([num_samples, num_channels])
    for start, dur, x_n in zip(intervals[:, 0], durations, signals):
        if x_n is None:
            continue
        num_samples = int(dur * samplerate)
        x_n = envelope(x_n[:num_samples], samplerate, **env_args)
        idx = int(start * samplerate)
        x_len = min([num_samples, len(x_n)])
        output_buffer[idx:idx + x_len] += x_n

    return output_buffer


def random_symbolic_chord_sequence(chord_set, num_chords, repeat_prob=0.5):
    """
    Parameters
    ----------
    chord_set : dict
        Chord labels pointing to different note number voicings.
    num_chords : int
        Number of chord voicings to sample.

    Returns
    -------
    chord_labels : list
        Chord label names for each chord.
    note_numbers : list
        List of midi note numbers to sound in each chord.
    """
    chord_labels = []
    note_numbers = []
    last_chord = random.choice(chord_set.keys())
    for _ in range(num_chords):
        if np.random.binomial(1, p=repeat_prob):
            label = last_chord
        else:
            label = random.choice(chord_set.keys())
        chord_labels += [label]
        note_numbers += [random.choice(chord_set[chord_labels[-1]])]
        last_chord = chord_labels[-1]
    return chord_labels, note_numbers


def random_polyphonic_sequence(poly_probs, pitch_probs, num_collections):
    """
    Parameters
    ----------


    Returns
    -------
    chord_labels : list
        Chord label names for each chord.
    note_numbers : list
        List of midi note numbers to sound in each chord.
    """
    note_numbers = []
    for _ in range(num_collections):
        num_notes = util.categorical_sample(poly_probs)
        pitches = set()
        while len(pitches) < num_notes:
            pitches.add(util.categorical_sample(pitch_probs))
        note_numbers.append(list(pitches))
    return note_numbers


def random_drum_signal(drum_set, duration=30, samplerate=44100, channels=1):
    """
    Parameters
    ----------
    drum_set : dict
        Collection of drum clips and timing metadata. Requires the following
        keys: ['audio_file', 'beat_times', 'duration']
    duration : scalar
        Desired length of the signal, in seconds.
    samplerate : scalar
        Samplerate for the signal.
    channels : int
        Number of channels.

    Process
    -------
        apply random level / amplitude envelope (in log-space)

    Returns
    -------
    drum_signal : np.ndarray, shape=(num_samples, num_channels)
    beat_times : np.ndarray, ndim=1
    """
    key = random.choice(drum_set.keys())
    drum_data = drum_set[key]
    base_duration = drum_data['duration']
    num_repeats = int(max([np.round(duration / base_duration), 1]))
    signals = load_many([drum_data['audio_file']], samplerate=samplerate,
                        channels=channels, normalize_signal=True) * num_repeats
    beat_times = [np.asarray(drum_data['beat_times']) + base_duration * n
                  for n in range(num_repeats)]
    intervals = np.asarray([(n*base_duration, (n+1)*base_duration)
                            for n in range(num_repeats)])
    y_n = sequence_signals(signals, intervals, samplerate)
    return y_n, np.concatenate(beat_times)


def random_vsl_files_for_notes(instrument_set, note_numbers):
    return [random.choice(instrument_set[n]) for n in note_numbers]


def random_note_signal(intervals, instrument_set, notes, amplitudes=None,
                       samplerate=44100, env_args=None):
    """
    Parameters
    ----------
    intervals : np.ndarray
        Start times for the chords.
    instrument_set : dict
        Note numbers pointing to a list of relevant files.
    chord_set : dict
        Chord labels pointing to a set of note-number voicings.
    samplerate : scalar
        Samplerate for the signal.
    channels : int
        Number of channels for the signal.

    Process
    -------
        Generate a symbolic chord sequence -> labels, note_numbers
        beat_times -> intervals
            subsample / merge?
        for each note_collection, pick instrument files
        load signals for each collection
            scale each signal
        combine each chord
        sequence chords to intervals
        apply random level / amplitude envelope (in log-space)

    Returns
    -------
    chord_signal : np.ndarray
        Synthesized audio signal.
    labels : list
        List of string chord labels.
    """

    # signals = load_many(filenames, samplerate, normalize)
    # return combine(signals)

    signals = []
    amplitudes = np.ones(len(intervals)) if amplitudes is None else amplitudes
    for nts, amp in zip(notes, amplitudes):
        if not nts:
            signals.append(None)
            continue
        instrument_files = random_vsl_files_for_notes(instrument_set, nts)
        note_signals = load_many(instrument_files, samplerate=samplerate,
                                 channels=1, normalize_signal=True)
        signals += [amp*combine(note_signals)]

    return sequence_signals(signals, intervals, samplerate, env_args)


def random_noise_signal(audio_files, intervals, amplitudes=None,
                        samplerate=44100, env_args=None):
    """Noise Signal

    Parameters
    ----------
        intervals, audio_files

    Process
    -------
        for each interval, random activation
        load signals for each activation
        scale signals
        sequence to intervals
        apply random level / amplitude envelope (in log-space)

    Returns
    -------
    noise_signal
    """
    files = [random.choice(audio_files) for _ in range(len(intervals))]
    signals = load_many(files, samplerate, channels=1, normalize_signal=True)
    amplitudes = np.ones(len(intervals)) if amplitudes is None else amplitudes
    return sequence_signals([a*x for a, x in zip(amplitudes, signals)],
                            intervals, samplerate, env_args)


def start_times_to_intervals(start_times, end_duration=None):
    """Convert a set of start times to adjacent, non-overlapping intervals.

    Parameters
    ----------
    start_times : array_like
        Set of start times; must be increasing.
    end_duration : scalar>0, default=None
        Desired duration for the final interval; if None, the mean difference
        of the start times is used.

    Returns
    -------
    intervals : np.ndarray, shape=(len(start_times), 2)
        Start and end times of the intervals.
    """
    durations = np.abs(np.diff(start_times))
    if end_duration is None:
        end_duration = durations.mean() + start_times.max()
    intervals = []
    for t, d in zip(start_times, durations):
        intervals += [(t, t+d)]
    return np.asarray(intervals + [(start_times[-1], end_duration)])


def make_one_chord(drum_set, chord_set, instrument_set, noise_files,
                   weights=None, duration=30, samplerate=44100,
                   min_beat_period=0.5):
    drum_signal, beat_times = random_drum_signal(
        drum_set, samplerate=samplerate, duration=duration)
    if np.abs(np.diff(beat_times)).mean() < min_beat_period:
        beat_times = beat_times[::2]
    intervals = start_times_to_intervals(beat_times)

    # -- Generate chords --
    chord_labels, notes = random_symbolic_chord_sequence(
        chord_set, len(intervals))
    chord_signal = random_note_signal(
        intervals, instrument_set, notes, samplerate=samplerate)

    # -- Generate noise --
    # Subsample intervals
    idx = np.random.binomial(1, 0.5, size=len(intervals)).astype(bool)
    noise_signal = random_noise_signal(
        noise_files, intervals[idx], samplerate=samplerate)

    weights = np.ones(3) / 3.0 if weights is None else weights
    signals = [drum_signal, chord_signal, noise_signal]
    y_out = combine([w * x for w, x in zip(weights, signals)])
    return y_out, chord_labels, intervals


def make_one_note_sequence(drum_set, instrument_set, noise_files, max_notes=8,
                           mix_weights=None, duration=30, samplerate=44100,
                           min_beat_period=0.5):
    drum_signal, beat_times = random_drum_signal(
        drum_set, samplerate=samplerate, duration=duration)
    if np.abs(np.diff(beat_times)).mean() < min_beat_period:
        beat_times = beat_times[::2]
    intervals = start_times_to_intervals(beat_times)

    # -- Generate note sequence --
    pitch_probs = np.zeros(120)
    for note_num in instrument_set:
        pitch_probs[int(note_num)] = len(instrument_set[note_num])
    pitch_probs /= float(pitch_probs.sum())

    notes = random_polyphonic_sequence(
        np.ones(max_notes + 1) / float(max_notes + 1),
        pitch_probs, len(intervals))

    note_signal = random_note_signal(
        intervals, instrument_set, notes, samplerate=samplerate)

    # -- Generate noise --
    # Subsample intervals
    idx = np.random.binomial(1, 0.5, size=len(intervals)).astype(bool)
    noise_signal = random_noise_signal(
        noise_files, intervals[idx], samplerate=samplerate)

    mix_weights = np.ones(3) / 3.0 if mix_weights is None else mix_weights
    signals = [drum_signal, note_signal, noise_signal]
    y_out = combine([w * x for w, x in zip(mix_weights, signals)])
    return y_out, notes, intervals


def render_chords(n, drum_set, chord_set, instrument_set, noise_files,
                  duration, audio_dir, lab_dir, samplerate=44100.0):
    weights = [np.random.uniform(0.25, 1.0),
               np.random.uniform(0.25, 0.5),
               np.random.uniform(0.5, 0.75)]
    try:
        y_out, chord_labels, intervals = make_one_chord(
            drum_set, chord_set,
            instrument_set, noise_files,
            weights, duration, samplerate)
    except ValueError:
        print "Died trying to render '%d', revisit..."
        return
    y_out *= np.random.uniform(0.25, 1.0) / np.abs(y_out).max()
    apath = "%s/%04d.mp3" % (audio_dir, n)
    marl.audio.write(apath, y_out, samplerate)
    with open("%s/%04d.json" % (lab_dir, n), 'w') as fp:
        json.dump(
            dict(labels=chord_labels, intervals=intervals.tolist()),
            fp, indent=2)
    print "[%s] Finished %12d" % (time.asctime(), n)


def render_polypitch(idx, drum_set, instrument_set, noise_files, duration,
                     audio_dir, lab_dir, samplerate=44100.0):
    mix_weights = [np.random.uniform(0.25, 1.0),
                   np.random.uniform(0.25, 0.5),
                   np.random.uniform(0.5, 0.75)]
    y_out, note_numbers, intervals = make_one_note_sequence(
        drum_set=drum_set, instrument_set=instrument_set, max_notes=8,
        noise_files=noise_files, mix_weights=mix_weights, duration=duration,
        samplerate=samplerate)

    y_out *= np.random.uniform(0.25, 1.0) / np.abs(y_out).max()
    apath = "%s/%04d.mp3" % (audio_dir, idx)
    marl.audio.write(apath, y_out, samplerate)
    with open("%s/%04d.json" % (lab_dir, idx), 'w') as fp:
        json.dump(
            dict(note_numbers=note_numbers, intervals=intervals.tolist()),
            fp, indent=2)
    print "[%s] Finished %12d" % (time.asctime(), idx)


def main(args):
    audio_dir = futil.create_directory(
        os.path.join(args.output_directory, "audio"))
    lab_dir = futil.create_directory(
        os.path.join(args.output_directory, "labs"))

    drum_set = json.load(open(args.drum_set))
    instrument_set = dict(
        [(int(k), v) for k, v in json.load(open(args.instrument_set)).items()])

    noise_files = glob.glob(os.path.join(args.noise_dir, "*.wav"))

    duration = 180
    samplerate = 44100.0

    for idx in range(args.num_files):
        render_polypitch(idx, drum_set, instrument_set, noise_files, duration,
                         audio_dir, lab_dir, samplerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("drum_set",
                        metavar="drum_set", type=str,
                        help="Path to a JSON file of drum track info.")
    parser.add_argument("instrument_set",
                        metavar="instrument_set", type=str,
                        help="Path to a JSON file of note numbers and files.")
    parser.add_argument("chord_voicings",
                        metavar="chord_voicings", type=str,
                        help="Path to a JSON file of chord labels and notes.")
    parser.add_argument("noise_dir",
                        metavar="noise_dir", type=str,
                        help="Path to a set of background noise files.")
    parser.add_argument("num_files",
                        metavar="num_files", type=int,
                        help="Path to a JSON file of .")

    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the training results.")
    main(parser.parse_args())
