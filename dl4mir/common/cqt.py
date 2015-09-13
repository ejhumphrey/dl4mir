""""""

from claudio.fileio import FramedAudioReader
import numpy as np


def constantq_kernel(q, freq_min, octaves, samplerate, bins_per_octave):
    """Generate a constant-Q kernel for applying as the Hadamard (element-wise)
    product in the complex Fourier domain.

    Parameters
    ----------
    q : scalar
        Bandwidth (quality) of each channel.
    freq_min : scalar
        Lowest analysis frequency desired. Note that this may be forced to
        change if the samplerate cannot support it.
    octaves : int
        Number of octaves to analyze over the freq_min.
    samplerate : scalar
        Samplerate in Hertz of the signals to be processed.
    bins_per_octave : int
        Number of channels (bins) for each octave.

    Returns
    -------
    kernel : np.ndarray
        2D complex-valued matrix of CQT coefficients.
    """
    freq_max = np.power(2.0, octaves) * freq_min
    oct_low = 0
    oct_high = np.log2(freq_max / float(freq_min))
    num_steps = int((oct_high) * bins_per_octave + 0.5) + 1
    f_basis = np.logspace(
        start=oct_low, stop=oct_high, num=num_steps, base=2) * freq_min

    a_basis = []
    N_max = -1
    # Log-space produces an N+1 length vector; drop the last value.
    for f_k in f_basis[:-1]:
        N_k = np.round(q * samplerate /
                       (f_k * (np.power(2.0, 1.0 / bins_per_octave) - 1)))
        a_k = np.sqrt(np.hanning(int(N_k))) / N_k * \
            np.exp(2j * np.pi * f_k / samplerate * np.arange(int(N_k)))
        a_basis += [a_k]
        if N_k > N_max:
            N_max = N_k

    K = len(a_basis)
    N_log2 = int(np.power(2, np.ceil(np.log2(N_max))))
    a_matrix = np.zeros([K, N_log2], dtype=np.complex)
    for k in range(K):
        N_k = len(a_basis[k])
        n_0 = int((N_log2 - N_k) / 2.0)
        a_matrix[k, n_0:n_0 + N_k] = a_basis[k]
    return np.fft.rfft(a_matrix.real, axis=1)


def cqt(filepath, q=1.0, freq_min=27.5, octaves=7, bins_per_octave=36,
        framerate=20.0, samplerate=11025.0, channels=1, bytedepth=2,
        overlap=None, stride=None, time_points=None, alignment='center',
        offset=0):
    """Compute the Constant-Q Transform of an audio file.

    Parameters
    ----------
    filepath: str
        Audio file to process.
    q: scalar, default=1.0
        Quality of the subband chanels.
    freq_min: scalar, default=27.5
        Minimum frequency of the analysis.
    octaves: scalar, default=7
        Number of octaves to compute.
    bins_per_octave: scalar, default=36
        Number of channels per octave.
    framerate: scalar, default=20.0
        Ratio of analysis frames per second, in Hertz.
    samplerate: scalar, default=11025.0
        Samplerate for the input audio; will raise an error if not high enough.
        TODO(ejhumphrey): It *might* be worth abstracting this away from the
        user, but it is nice to have the control.
    channels: int, default=1
        Number of channels for the input audio.
    bytedepth: int, default=2
        Number of bytes for the encoding. Everything is much more stable at 2.
    overlap: scalar, default=None
        Alternate method of controlling the number of frames per second, as the
        percent overlap between frames; as the transform is non-invertible, its
        use is discouraged but maintained as a courtesy.
    stride: int, default=None
        Alternate method of controlling the number of frames per second, as the
        number of samples between adjacent frames; as the transform is
        non-invertible, its use is discouraged but maintained as a courtesy.
    time_points: array_like, default=None
        Alternate method of controlling the alignment of analysis frames, where
        the frame times are provided as an array of values in seconds. Note
        that this takes precedent over all other alignment options.
    alignment: str, default='center'
        Justification for a frame around an index, one of 'left', 'center', or
        'right'.
    offset: scalar, default=0
        Number of samples to offset each frame around an index.

    Returns
    -------
    time_points: np.ndarray
        Time points of the analysis frames.
    cqt_spectra: np.ndarray, shape=(num_channels, num_frames, num_bins)
        Constant-Q coefficients for the audio signal.
    """
    freq_min_top_octave = freq_min * 2 ** (octaves - 1)
    freq_max = freq_min * 2 ** (octaves)
    if freq_max < (samplerate / 2.0):
        raise ValueError("Samplerate must be greater than {0} for the given "
                         "parameters.".format(freq_max * 2))

    kernel = constantq_kernel(
        q=q, freq_min=freq_min_top_octave,
        octaves=1,
        samplerate=samplerate,
        bins_per_octave=bins_per_octave)
    framesize = 2 * (kernel.shape[1] - 1)

    base_reader = FramedAudioReader(
        filepath, framesize=framesize, samplerate=samplerate,
        channels=channels, bytedepth=bytedepth, overlap=overlap, stride=stride,
        framerate=framerate, time_points=time_points, alignment=alignment,
        offset=offset)

    time_points = base_reader.time_points

    def generate_readers(count):
        this_reader, n = base_reader, 1
        yield this_reader
        while n < count:
            this_reader = FramedAudioReader(
                filepath=this_reader.wavefile,
                samplerate=this_reader.samplerate / 2.0,
                framesize=framesize,
                time_points=time_points,
                alignment=alignment,
                channels=channels)
            yield this_reader
            n += 1

    def fx(X):
        return np.array(
            [np.abs(np.dot(kernel, np.fft.rfft(x, axis=0))) for x in X])

    X = [fx(reader) for reader in generate_readers(octaves)]
    # Note that readers are generated backwards; so, reverse the result.
    X.reverse()
    # Truncate to the shortest duration octave. These *should* be the same
    # dimension, but this check ensures that they are.
    n_len = min([X_i.shape[0] for X_i in X])
    cqt_spectra = np.concatenate([X_i[:n_len, :] for X_i in X], axis=1)
    return reader.time_points, cqt_spectra.transpose(2, 0, 1)
