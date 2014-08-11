import argparse
import numpy as np
import marl
import random


def normalize(x_n):
    return x_n / np.abs(x_n).max()


def load_many(filenames, samplerate, channels=1, normalize=False):
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
        if normalize:
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
    output_buffer = np.zeros([num_samples, signals[0].shape[1]])
    for start, dur, x_n in zip(intervals[:, 0], durations, signals):
        num_samples = int(dur * samplerate)
        x_n = envelope(x_n[:num_samples], samplerate, **env_args)
        idx = int(start * samplerate)
        x_len = min([num_samples, len(x_n)])
        output_buffer[idx:idx + x_len] += x_n

    return output_buffer


def random_chord(instrument_set, notes, samplerate=44100, normalize=False):
    """
    Parameters
    ----------
    instrument_set : dict
        Note numbers pointing to a list of relevant files.
    notes : array_like
        Set of note numbers to choose from.
    samplerate : scalar
        Samplerate for the signal.
    """
    filenames = [random.choice(instrument_set[n]) for n in notes]
    signals = load_many(filenames, samplerate, normalize)
    return combine(signals)


def random_chord_sequence(intervals, instrument_set, chord_set,
                          samplerate=44100, env_args=None):
    signals = []
    chord_labels = []
    for _ in range(len(intervals)):
        chord_labels += [random.choice(chord_set.keys())]
        notes = random.choice(chord_set[chord_labels[-1]])
        signals += [random_chord(instrument_set, notes, samplerate)]

    y_out = sequence_signals(signals, intervals, samplerate, env_args)
    return y_out, chord_labels


def random_audio_sequence(audio_files, intervals,
                          samplerate=44100, env_args=None):
    files = [random.choice(audio_files) for _ in range(len(intervals))]
    signals = load_many(files, samplerate, channels=1)
    return sequence_signals(signals, intervals, samplerate, env_args)


# Mixing weights ...
def random_signal(drum_set, instrument_set, chord_set,
                  voice_files):
    duration = timing_data['duration']
    x_n, fs = marl.audio.read(drum_file, samplerate=44100, channels=1)
    chord_times = []
    voice_times = []
    for t in timing_data['beat_times'][1:-1]:
        if np.random.binomial(1, p=0.5):
            chord_times.append(t)
        if np.random.binomial(1, p=0.5):
            voice_times.append(t)

    y_n, intervals, chord_labels = random_chord_sequence(
        intervals, instrument_set, chord_set, samplerate=fs)
    v_n = random_audio_sequence(voice_files, intervals, samplerate=fs)
    y_n = y_n + v_n + x_n.squeeze()
    y_n /= np.abs(y_n).max()
    return y_n, intervals, chord_labels


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
        end_duration = durations.mean()
    intervals = []
    for t, d in zip(start_times, durations):
        intervals += [(t, t+d)]
    return np.asarray(intervals + [(start_times[-1], end_duration)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("drum_set",
                        metavar="drum_dir", type=str,
                        help="Path to a JSON file of .")
    parser.add_argument("instrument_set",
                        metavar="drum_dir", type=str,
                        help="Path to a JSON file of .")
    parser.add_argument("vox_set",
                        metavar="drum_dir", type=str,
                        help="Path to a JSON file of .")

    # Outputs
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Path to save the training results.")
    # main(parser.parse_args())
