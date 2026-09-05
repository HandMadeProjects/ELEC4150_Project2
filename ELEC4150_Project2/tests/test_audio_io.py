"""
Tests for part1/audio_io.py — load_audio and validate_audio.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from part1.audio_io import load_audio, validate_audio, save_audio, skip_leading_silence


class TestLoadAudio:

    def test_load_returns_array_and_int(self, temp_wav_file):
        """load_audio should return (ndarray, int)."""
        audio, fs = load_audio(temp_wav_file)
        assert isinstance(audio, np.ndarray)
        assert isinstance(fs, int)

    def test_load_stereo_shape(self, temp_wav_file):
        """Stereo WAV should give shape (N, 2)."""
        audio, fs = load_audio(temp_wav_file)
        assert audio.ndim == 2
        assert audio.shape[1] == 2

    def test_load_correct_sample_rate(self, temp_wav_file):
        """Sample rate read back should match what was written (44100)."""
        _, fs = load_audio(temp_wav_file)
        assert fs == 44100

    def test_load_nonexistent_file_raises(self):
        """Loading a missing file should raise an error."""
        with pytest.raises(Exception):
            load_audio("nonexistent_file_xyz.wav")


class TestValidateAudio:

    def test_valid_stereo_passes(self, stereo_audio):
        """validate_audio should return True for valid stereo input."""
        audio, fs = stereo_audio
        result = validate_audio(audio, fs)
        assert result is True

    def test_valid_mono_passes(self, mono_audio):
        """validate_audio should return True for valid mono input."""
        audio, fs = mono_audio
        result = validate_audio(audio, fs)
        assert result is True

    def test_empty_audio_raises(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="no samples"):
            validate_audio(np.array([]), 44100)

    def test_invalid_fs_raises(self, stereo_audio):
        """Zero/negative sample rate should raise ValueError."""
        audio, _ = stereo_audio
        with pytest.raises(ValueError, match="Invalid sampling rate"):
            validate_audio(audio, 0)

    def test_nan_audio_raises(self, stereo_audio):
        """Audio containing NaN should raise ValueError."""
        audio, fs = stereo_audio
        audio_bad = audio.copy()
        audio_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="invalid values"):
            validate_audio(audio_bad, fs)

    def test_inf_audio_raises(self, stereo_audio):
        """Audio containing Inf should raise ValueError."""
        audio, fs = stereo_audio
        audio_bad = audio.copy()
        audio_bad[10, 1] = np.inf
        with pytest.raises(ValueError, match="invalid values"):
            validate_audio(audio_bad, fs)


class TestSaveAudio:

    def test_save_and_reload(self, tmp_path, mono_audio):
        """save_audio should write WAV readable by load_audio."""
        audio, fs = mono_audio
        out_path = str(tmp_path / "test_out.wav")
        saved = save_audio(out_path, audio, fs)
        assert os.path.exists(saved)

        reloaded, reload_fs = load_audio(saved)
        assert reload_fs == fs
        assert len(reloaded) == len(audio)
        np.testing.assert_allclose(reloaded, audio, atol=1e-3)

    def test_save_normalizes_clipping(self, tmp_path):
        """save_audio should normalize peak if > 1.0 without crashing."""
        fs = 44100
        loud_audio = np.array([2.5, -3.0, 1.2], dtype=np.float32)
        out_path = str(tmp_path / "loud.wav")
        save_audio(out_path, loud_audio, fs)
        reloaded, _ = load_audio(out_path)
        assert np.max(np.abs(reloaded)) <= 1.0


class TestSkipLeadingSilence:

    def test_drops_leading_zeros(self):
        fs = 8000
        silence = np.zeros(fs, dtype=np.float64)
        tone = 0.4 * np.ones(fs, dtype=np.float64)
        audio = np.concatenate([silence, tone])
        trimmed = skip_leading_silence(audio, fs, threshold=0.01, pre_roll_s=0.0)
        assert len(trimmed) == len(tone)
        np.testing.assert_allclose(trimmed[:100], tone[:100], atol=1e-12)

    def test_stereo_shape_preserved(self):
        fs = 8000
        silence = np.zeros((fs, 2))
        active = 0.3 * np.ones((fs, 2))
        audio = np.vstack([silence, active])
        trimmed = skip_leading_silence(audio, fs, threshold=0.01, pre_roll_s=0.0)
        assert trimmed.ndim == 2
        assert trimmed.shape[1] == 2
        assert len(trimmed) == fs
