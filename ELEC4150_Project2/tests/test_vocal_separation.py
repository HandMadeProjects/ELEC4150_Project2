"""
Tests for part1/vocal_separation.py — separate_vocal_stereo.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from part1.vocal_separation import (
    separate_vocal_stereo,
    separate_vocal_and_karaoke,
    separate_karaoke_wiener,
    measure_vocal_attenuation_db,
)


class TestSeparateVocalStereo:

    def test_returns_two_arrays(self, stereo_audio):
        """Function should return a (vocal, instrumental) tuple."""
        audio, _ = stereo_audio
        result = separate_vocal_stereo(audio)
        assert len(result) == 2

    def test_vocal_is_mean_of_channels(self, stereo_audio):
        """Vocal = (L + R) / 2 before normalisation."""
        audio, _ = stereo_audio
        left = audio[:, 0]
        right = audio[:, 1]
        expected_vocal_unnorm = (left + right) / 2.0
        vocal, _ = separate_vocal_stereo(audio)

        # If normalisation was applied, scale expected too
        peak = np.max(np.abs(expected_vocal_unnorm))
        if peak > 1:
            expected_vocal_unnorm = expected_vocal_unnorm / peak

        np.testing.assert_allclose(vocal, expected_vocal_unnorm, atol=1e-9)

    def test_instrumental_is_difference(self, stereo_audio):
        """Instrumental = (L - R) / 2 before normalisation."""
        audio, _ = stereo_audio
        left = audio[:, 0]
        right = audio[:, 1]
        expected_inst_unnorm = (left - right) / 2.0
        _, instrumental = separate_vocal_stereo(audio)

        peak = np.max(np.abs(expected_inst_unnorm))
        if peak > 1:
            expected_inst_unnorm = expected_inst_unnorm / peak

        np.testing.assert_allclose(instrumental, expected_inst_unnorm, atol=1e-9)

    def test_centred_audio_vocal_equals_channel(self, centred_stereo):
        """When L == R, vocal == L == R (pure centre)."""
        audio, _ = centred_stereo
        vocal, _ = separate_vocal_stereo(audio)
        np.testing.assert_allclose(vocal, audio[:, 0], atol=1e-9)

    def test_centred_audio_instrumental_is_zero(self, centred_stereo):
        """When L == R, instrumental (difference) should be all zeros."""
        audio, _ = centred_stereo
        _, instrumental = separate_vocal_stereo(audio)
        np.testing.assert_allclose(instrumental, 0.0, atol=1e-9)

    def test_mono_input_raises_value_error(self, mono_audio):
        """1-D (mono) input must raise ValueError."""
        audio, _ = mono_audio
        with pytest.raises(ValueError, match="stereo"):
            separate_vocal_stereo(audio)

    def test_single_channel_2d_raises(self):
        """Shape (N, 1) input must raise ValueError."""
        audio = np.random.randn(44100, 1)
        with pytest.raises(ValueError, match="stereo"):
            separate_vocal_stereo(audio)

    def test_output_length_matches_input(self, stereo_audio):
        """Output arrays must have the same length as the input."""
        audio, _ = stereo_audio
        vocal, instrumental = separate_vocal_stereo(audio)
        assert len(vocal) == len(audio)
        assert len(instrumental) == len(audio)

    def test_peak_amplitude_within_bounds(self, stereo_audio):
        """After normalisation, peak amplitude must be <= 1.0."""
        audio, _ = stereo_audio
        vocal, instrumental = separate_vocal_stereo(audio)
        assert np.max(np.abs(vocal)) <= 1.0 + 1e-9
        assert np.max(np.abs(instrumental)) <= 1.0 + 1e-9

    def test_output_is_finite(self, stereo_audio):
        """Both outputs must be finite (no NaN / Inf)."""
        audio, _ = stereo_audio
        vocal, instrumental = separate_vocal_stereo(audio)
        assert np.isfinite(vocal).all()
        assert np.isfinite(instrumental).all()


def _band_energy(sig, fs, f0, bw=25.0):
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    return float(np.sum(np.abs(spec[mask]) ** 2))


def _centred_mix(fs=16000, duration=1.5):
    """Centre 1 kHz vocal + left 400 Hz + right 700 Hz + shared 80 Hz bass."""
    t = np.arange(int(fs * duration)) / fs
    vocal = 0.30 * np.sin(2 * np.pi * 1000 * t)
    bass = 0.25 * np.sin(2 * np.pi * 80 * t)
    left_inst = 0.22 * np.sin(2 * np.pi * 400 * t)
    right_inst = 0.22 * np.sin(2 * np.pi * 700 * t)
    audio = np.stack(
        [vocal + bass + left_inst, vocal + bass + right_inst],
        axis=1,
    )
    return audio, fs, vocal


class TestSeparateVocalAndKaraoke:
    def test_mono_input_raises(self, mono_audio):
        audio, fs = mono_audio
        with pytest.raises(ValueError, match="stereo"):
            separate_vocal_and_karaoke(audio, fs)

    def test_output_shapes_and_peaks(self, stereo_audio):
        audio, fs = stereo_audio
        vocal, karaoke, atten = separate_vocal_and_karaoke(audio, fs)
        assert vocal.shape == (len(audio),)
        assert karaoke.shape == audio.shape
        assert np.max(np.abs(vocal)) <= 1.0 + 1e-9
        assert np.max(np.abs(karaoke)) <= 1.0 + 1e-9
        assert np.isfinite(vocal).all()
        assert np.isfinite(karaoke).all()
        assert np.isfinite(atten)

    def test_centred_tone_removed_from_karaoke(self):
        """Part 1.3: a centre-panned 1 kHz 'vocal' is attenuated ≥ 40 dB."""
        audio, fs, _ = _centred_mix()
        _, karaoke, atten_db = separate_vocal_and_karaoke(audio, fs)
        k_mono = np.mean(karaoke, axis=1)
        orig_mono = np.mean(audio, axis=1)
        ratio_db = 10 * np.log10(
            (_band_energy(k_mono, fs, 1000) + 1e-20)
            / (_band_energy(orig_mono, fs, 1000) + 1e-20)
        )
        assert ratio_db <= -35.0, f"1 kHz residual {ratio_db:.1f} dB"
        assert atten_db >= 39.99, f"reported attenuation {atten_db:.1f} dB"

    def test_panned_instruments_kept_in_karaoke(self):
        """Part 1.3: hard-panned instruments stay in the karaoke track."""
        audio, fs, _ = _centred_mix()
        _, karaoke, _ = separate_vocal_and_karaoke(audio, fs)
        left_keep = 10 * np.log10(
            (_band_energy(karaoke[:, 0], fs, 400) + 1e-20)
            / (_band_energy(audio[:, 0], fs, 400) + 1e-20)
        )
        right_keep = 10 * np.log10(
            (_band_energy(karaoke[:, 1], fs, 700) + 1e-20)
            / (_band_energy(audio[:, 1], fs, 700) + 1e-20)
        )
        assert left_keep >= -3.0, f"left 400 Hz lost {left_keep:.1f} dB"
        assert right_keep >= -3.0, f"right 700 Hz lost {right_keep:.1f} dB"

    def test_bass_kept_in_karaoke(self):
        """Part 1.3: sub-vocal bass is protected."""
        audio, fs, _ = _centred_mix()
        _, karaoke, _ = separate_vocal_and_karaoke(audio, fs)
        k_mono = np.mean(karaoke, axis=1)
        orig_mono = np.mean(audio, axis=1)
        bass_keep = 10 * np.log10(
            (_band_energy(k_mono, fs, 80, bw=15) + 1e-20)
            / (_band_energy(orig_mono, fs, 80, bw=15) + 1e-20)
        )
        assert bass_keep >= -3.0, f"bass lost {bass_keep:.1f} dB"

    def test_extracted_vocal_is_the_centre_tone(self):
        """Part 1.2: isolated vocal is the 1 kHz centre, not the panned instruments."""
        audio, fs, _ = _centred_mix()
        vocal, _, _ = separate_vocal_and_karaoke(audio, fs)
        vocal_vs_left = 10 * np.log10(
            (_band_energy(vocal, fs, 1000) + 1e-20)
            / (_band_energy(vocal, fs, 400) + 1e-20)
        )
        vocal_vs_bass = 10 * np.log10(
            (_band_energy(vocal, fs, 1000) + 1e-20)
            / (_band_energy(vocal, fs, 80, bw=15) + 1e-20)
        )
        assert vocal_vs_left >= 20.0, f"vocal/400 Hz only {vocal_vs_left:.1f} dB"
        assert vocal_vs_bass >= 12.0, f"vocal/bass only {vocal_vs_bass:.1f} dB"

    def test_measure_matches_separation(self):
        audio, fs, _ = _centred_mix()
        _, karaoke, atten = separate_vocal_and_karaoke(audio, fs)
        measured = measure_vocal_attenuation_db(audio, karaoke, fs)
        assert abs(measured - atten) < 1.0

    def test_wiener_wrapper_returns_mono_karaoke(self):
        audio, fs, _ = _centred_mix()
        vocal, karaoke = separate_karaoke_wiener(audio, fs)
        assert vocal.ndim == 1
        assert karaoke.ndim == 1
        assert len(vocal) == len(audio)
        assert len(karaoke) == len(audio)
