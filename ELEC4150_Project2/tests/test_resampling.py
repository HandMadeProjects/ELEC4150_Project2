"""
Tests for part1/resampling.py — resample_audio.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from part1.resampling import resample_audio


class TestResampleAudio:

    def test_output_length_upsample(self, mono_audio):
        """Upsampling from 44100 to 60000 Hz — output length should scale correctly."""
        audio, fs_in = mono_audio
        fs_out = 60000
        resampled = resample_audio(audio, fs_in, fs_out)
        expected_len = int(len(audio) * fs_out / fs_in)
        # Allow ±2 samples tolerance (polyphase rounding)
        assert abs(len(resampled) - expected_len) <= 2

    def test_output_length_downsample(self, mono_audio):
        """Downsampling from 44100 to 22050 Hz — output length should halve."""
        audio, fs_in = mono_audio
        fs_out = 22050
        resampled = resample_audio(audio, fs_in, fs_out)
        expected_len = int(len(audio) * fs_out / fs_in)
        assert abs(len(resampled) - expected_len) <= 2

    def test_same_rate_unchanged_length(self, mono_audio):
        """Resampling to the same rate should return the same length."""
        audio, fs = mono_audio
        resampled = resample_audio(audio, fs, fs)
        assert len(resampled) == len(audio)

    def test_output_is_1d_for_1d_input(self, mono_audio):
        """1-D input should produce 1-D output."""
        audio, fs = mono_audio
        resampled = resample_audio(audio, fs, 60000)
        assert resampled.ndim == 1

    def test_output_values_finite(self, mono_audio):
        """Resampled output must not contain NaN or Inf."""
        audio, fs = mono_audio
        resampled = resample_audio(audio, fs, 60000)
        assert np.isfinite(resampled).all()

    def test_target_60k(self, stereo_audio):
        """Project requirement: resample each channel to 60 kHz."""
        audio, fs_in = stereo_audio
        left = audio[:, 0]
        resampled = resample_audio(left, fs_in, 60000)
        expected_len = int(len(left) * 60000 / fs_in)
        assert abs(len(resampled) - expected_len) <= 2

    def test_amplitude_roughly_preserved(self, mono_audio):
        """Peak amplitude of a pure sine should be roughly preserved after resampling."""
        audio, fs = mono_audio
        resampled = resample_audio(audio, fs, 60000)
        # Allow 10% amplitude drift
        assert abs(np.max(np.abs(resampled)) - np.max(np.abs(audio))) < 0.1
