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
import pretty_midi

from multiprocessing import Pool

NUM_CPUS = 12


def normalize(x_n):
    return x_n / np.abs(x_n).max()



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
    attack_env = np.linspace(0, peak_level, int(attack * samplerate))
    decay_env = np.linspace(peak_level, sustain_level, int(decay * samplerate))
    release_env = np.linspace(sustain_level, 0, int(release * samplerate))

    sustain_samples = len(x_n) - sum([len(_) for _ in [attack_env,
                                                       decay_env,
                                                       release_env]])
    sustain_samples = max([sustain_samples, 0])
    sustain_env = np.ones(sustain_samples) * sustain_level

    env = np.concatenate([attack_env, decay_env, sustain_env, release_env])
    L = len(x_n)
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


def synthesize_midi_data(data, instrument_set, samplerate):
    num_samples = time_to_samples(data.get_end_time(), samplerate)
    output_signal = np.zeros([num_samples, 1])
    for i in data.instruments:
        if i.is_drum:
            continue
        for count, n in enumerate(i.notes):
            if not n.pitch in instrument_set:
                continue
            f = random_vsl_files_for_notes(instrument_set, [n.pitch])[0]
            i0 = time_to_samples(n.start, samplerate)
            i1 = time_to_samples(n.end, samplerate)
            y_n = load_note(f, samplerate, num_samples=i1-i0)
            output_signal[i0:i1] += (y_n*velocity_to_gain(n.velocity))
        print "Finished %s" % i

    return output_signal


def time_to_samples(v, samplerate):
    return int(np.round(v * samplerate))


def load_note(f, samplerate, num_samples=None, **env_args):
    x_n = marl.audio.read(f, samplerate=samplerate, channels=1)[0]
    if not num_samples is None:
        x_out = np.zeros([num_samples, x_n.shape[1]])
        x_len = min([len(x_n), num_samples])
        x_out[:x_len, :] = x_n[:x_len, :]
    else:
        x_out = x_n
    return envelope(x_out, samplerate, **env_args)


def velocity_to_gain(veloc, db_range=60):
    r = 10.0 ** (db_range / 20.0)
    b = 127.0 / (126.0 * np.sqrt(r)) - (1.0 / 126.0)
    m = (1 - b) / 127.0
    return (m * veloc + b) ** 2


def random_vsl_files_for_notes(instrument_set, note_numbers):
    return [random.choice(instrument_set[n]) for n in note_numbers]


def process_one(args):
    midi_file, output_directory, instrument_set, samplerate = args
    data = pretty_midi.PrettyMIDI(midi_file)
    y_n = synthesize_midi_data(data, instrument_set, samplerate)
    output_file = os.path.join(output_directory,
                               "%s.mp3" % futil.filebase(midi_file))
    marl.audio.write(output_file, y_n, samplerate)
    print "[%s] Finished %s" % (time.asctime(), midi_file)


def spool_args(midi_files, output_directory, instrument_set,
               samplerate=22050.0):
    for f in midi_files:
        yield (f, output_directory, instrument_set, samplerate)


def main(args):
    output_dir = futil.create_directory(args.output_directory)

    instrument_set = dict(
        [(int(k), v) for k, v in json.load(open(args.instrument_set)).items()])
    midi_files = glob.glob(os.path.join(args.midi_directory, "*.mid"))
    pool = Pool(processes=NUM_CPUS)
    pool.map_async(
        func=process_one,
        iterable=spool_args(midi_files, output_dir, instrument_set))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("midi_directory",
                        metavar="midi_directory", type=str,
                        help="Path to a JSON file of drum track info.")
    parser.add_argument("instrument_set",
                        metavar="instrument_set", type=str,
                        help="Path to a JSON file of note numbers and files.")
    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the training results.")
    main(parser.parse_args())
