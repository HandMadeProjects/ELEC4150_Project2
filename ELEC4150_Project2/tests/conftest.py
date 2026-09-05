"""
Shared pytest fixtures for ELEC4150 Project 2 test suite.
"""

import numpy as np
import pytest
import tempfile
import os
import soundfile as sf


@pytest.fixture
def stereo_audio():
    """Synthetic 1-second stereo audio at 44100 Hz."""
    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)   # 440 Hz sine
    right = 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz sine
    audio = np.stack([left, right], axis=1)     # shape (N, 2)
    return audio, fs


@pytest.fixture
def mono_audio():
    """Synthetic 1-second mono audio at 44100 Hz."""
    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    return signal, fs


@pytest.fixture
def temp_wav_file(stereo_audio):
    """Write synthetic stereo audio to a temporary WAV file and return the path."""
    audio, fs = stereo_audio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, fs)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def centred_stereo():
    """
    Stereo audio where L == R (perfectly centre-panned).
    Vocal extraction should return signal == mean(L, R) == L == R.
    """
    fs = 44100
    t = np.linspace(0, 1.0, fs, endpoint=False)
    mono = 0.6 * np.sin(2 * np.pi * 440 * t)
    audio = np.stack([mono, mono], axis=1)
    return audio, fs
