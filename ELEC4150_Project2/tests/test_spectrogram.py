"""
Tests for part1/spectrogram.py — calculate_spectrogram.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from part1.spectrogram import calculate_spectrogram


class TestCalculateSpectrogram:

    def test_returns_three_arrays(self, mono_audio):
        """Function should return (frequencies, times, magnitude_db)."""
        audio, fs = mono_audio
        result = calculate_spectrogram(audio, fs)
        assert len(result) == 3

    def test_frequencies_shape(self, mono_audio):
        """frequencies array should have shape (nperseg//2 + 1,) = (1025,)."""
        audio, fs = mono_audio
        frequencies, _, _ = calculate_spectrogram(audio, fs)
        # nperseg=2048 → 1025 frequency bins
        assert frequencies.shape == (1025,)

    def test_frequencies_start_at_zero(self, mono_audio):
        """First frequency bin must be 0 Hz (DC component)."""
        audio, fs = mono_audio
        frequencies, _, _ = calculate_spectrogram(audio, fs)
        assert frequencies[0] == pytest.approx(0.0)

    def test_frequencies_end_at_nyquist(self, mono_audio):
        """Last frequency bin must be Nyquist = fs/2."""
        audio, fs = mono_audio
        frequencies, _, _ = calculate_spectrogram(audio, fs)
        assert frequencies[-1] == pytest.approx(fs / 2, rel=1e-3)

    def test_times_array_positive(self, mono_audio):
        """All time values must be >= 0."""
        audio, fs = mono_audio
        _, times, _ = calculate_spectrogram(audio, fs)
        assert np.all(times >= 0)

    def test_magnitude_db_is_2d(self, mono_audio):
        """Magnitude array should be 2-D (freq_bins × time_frames)."""
        audio, fs = mono_audio
        _, _, magnitude_db = calculate_spectrogram(audio, fs)
        assert magnitude_db.ndim == 2

    def test_magnitude_db_shape_consistent(self, mono_audio):
        """magnitude_db rows == len(frequencies), cols == len(times)."""
        audio, fs = mono_audio
        frequencies, times, magnitude_db = calculate_spectrogram(audio, fs)
        assert magnitude_db.shape[0] == len(frequencies)
        assert magnitude_db.shape[1] == len(times)

    def test_magnitude_db_is_finite(self, mono_audio):
        """Magnitude array must not contain NaN (Inf is allowed for true zeros)."""
        audio, fs = mono_audio
        _, _, magnitude_db = calculate_spectrogram(audio, fs)
        assert not np.any(np.isnan(magnitude_db))

    def test_magnitude_db_values_reasonable(self, mono_audio):
        """dB values should be in a sane range: -300 to +100 dB."""
        audio, fs = mono_audio
        _, _, magnitude_db = calculate_spectrogram(audio, fs)
        assert np.all(magnitude_db > -350)
        assert np.all(magnitude_db < 100)

    def test_60k_sample_rate(self):
        """Spectrogram should work correctly at the project's 60 kHz rate."""
        fs = 60000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        frequencies, times, magnitude_db = calculate_spectrogram(audio, fs)
        assert frequencies[-1] == pytest.approx(fs / 2, rel=1e-3)
        assert magnitude_db.ndim == 2
