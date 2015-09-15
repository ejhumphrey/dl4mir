import numpy as np
import claudio
from dl4mir.common import fileutil as F

from dl4mir.common import audio_files_to_cqt_arrays as CQT


def test_base_call():
    textlist = F.TempFile('.txt')
    audio = F.TempFile('.wav')
    samplerate = 8000.0
    sine = np.sin(2 * np.pi * 440 / samplerate * np.arange(5 * samplerate))
    claudio.write(audio.path, sine, samplerate)

    F.dump_textlist([audio.path], textlist.path)
    output_dir = F.TempDir()
    assert CQT.main(textlist.path, output_dir.path)
