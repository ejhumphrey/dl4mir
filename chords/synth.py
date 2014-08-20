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


def random_symbolic_chord_sequence(chord_set, num_chords):
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
    for _ in range(num_chords):
        chord_labels += [random.choice(chord_set.keys())]
        note_numbers += [random.choice(chord_set[chord_labels[-1]])]
    return chord_labels, note_numbers


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
    num_repeats = max([np.round(duration / base_duration), 1])
    signals = load_many([drum_data['audio_file']], samplerate=samplerate,
                        channels=channels, normalize=True) * num_repeats
    beat_times = [np.asarray(drum_data['beat_times']) + base_duration * n
                  for n in num_repeats]
    intervals = np.asarray([(n, n+base_duration) for n in num_repeats])
    y_n = sequence_signals(signals, intervals, samplerate)
    return y_n, np.concatenate(beat_times)


def random_chord_signal(intervals, instrument_set, notes,
                        samplerate=44100, normalize=False):
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

    def random_vsl_files_for_notes(instrument_set, note_numbers):
        return [random.choice(instrument_set[n]) for n in note_numbers]

    signals = load_many(filenames, samplerate, normalize)
    return combine(signals)

    signals = []
    chord_labels = []
    for _ in range(len(intervals)):
        chord_labels += [random.choice(chord_set.keys())]
        notes = random.choice(chord_set[chord_labels[-1]])
        signals += [random_chord(instrument_set, notes, samplerate)]

    y_out = sequence_signals(signals, intervals, samplerate, env_args)
    return y_out, chord_labels


def random_noise_signal():
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
    pass


def random_audio_sequence(audio_files, intervals,
                          samplerate=44100, env_args=None):
    files = [random.choice(audio_files) for _ in range(len(intervals))]
    signals = load_many(files, samplerate, channels=1)
    return sequence_signals(signals, intervals, samplerate, env_args)


# Mixing weights ...
def generate_signal(drum_file, instrument_set, chord_set,
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
