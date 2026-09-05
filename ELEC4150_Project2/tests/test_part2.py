import pytest
import numpy as np
from scipy.signal import resample_poly

from part2.compression import compress_vocal, measure_bitrate, compression_snr_db
from part2.modulation import modulate_ssb, measure_bandwidth
from part2.demodulation import demodulate_ssb, evaluate_demodulation
from part2.noise import add_awgn, calculate_snr
from part2.filtering import (
    apply_anti_aliasing_lpf,
    apply_rf_bandpass,
    apply_wiener_denoise,
    apply_countermeasure_pipeline,
)


def test_compression_64kbps():
    """Verify compression achieves exactly 64 kbps (8 bits x 8 kHz)."""
    fs = 60000
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    compressed_bytes, decoded, fs_out, bitrate = compress_vocal(
        signal, fs, target_bitrate=64000
    )

    assert abs(bitrate - 64000) < 1.0, f"Expected 64 kbps, got {bitrate} bps"
    assert len(compressed_bytes) == int(duration * 8000)
    assert fs_out == fs
    assert len(decoded) == len(signal)


def test_compression_audio_quality():
    """Verify μ-law decode preserves signal fidelity with positive SNR."""
    fs = 60000
    t = np.arange(int(fs * 1.0)) / fs
    signal = 0.6 * np.sin(2 * np.pi * 1000 * t)

    _, decoded, _, _ = compress_vocal(signal, fs, target_bitrate=64000)
    snr = compression_snr_db(signal, decoded)
    assert snr > 12.0, f"Expected compression SNR > 12 dB, got {snr:.2f} dB"


def test_modulation_carrier_and_bandwidth():
    """Verify SSB-AM carrier is at 250 kHz and bandwidth <= 4 kHz."""
    fs_rf = 600000
    fc = 250000
    max_bw = 4000
    duration = 1.0
    t = np.arange(int(fs_rf * duration)) / fs_rf
    # 2 kHz test tone
    baseband = 0.7 * np.sin(2 * np.pi * 2000 * t)

    modulated, t_rf, carrier = modulate_ssb(baseband, fs=fs_rf, fc=fc, mode="usb")

    bw_hz, freqs, psd_db = measure_bandwidth(modulated, fs=fs_rf, fc=fc, threshold_db=-40)
    peak_freq = freqs[np.argmax(psd_db)]

    assert abs(peak_freq - fc) <= max_bw, f"Carrier not near {fc} Hz: got {peak_freq}"
    assert bw_hz <= max_bw + 100, f"Bandwidth {bw_hz} Hz exceeded limit {max_bw} Hz"


def test_awgn_noise_snr():
    """Verify AWGN generates target 10 dB SNR accurately."""
    sig = np.random.randn(50000)
    target_snr = 10.0
    noisy, noise, measured_snr = add_awgn(sig, snr_db=target_snr, seed=42)

    assert abs(measured_snr - target_snr) < 0.2, f"Expected {target_snr} dB, got {measured_snr} dB"


def test_countermeasure_improvement():
    """Verify RF bandpass countermeasure delivers significant SNR gain (> 8 dB)."""
    fs_rf = 600000
    fc = 250000
    bw = 4000
    t = np.arange(60000) / fs_rf
    sig = 0.5 * np.sin(2 * np.pi * 252000 * t)

    noisy, _, snr_init = add_awgn(sig, snr_db=10.0, seed=42)
    cleaned, snr_before, snr_after, gain = apply_countermeasure_pipeline(
        noisy, sig, fs_rf=fs_rf, fc=fc, bw=bw, mode="usb"
    )

    assert snr_after > 18.0, f"Expected post-countermeasure SNR > 18 dB, got {snr_after:.2f} dB"
    assert gain > 8.0, f"Expected gain > 8 dB, got {gain:.2f} dB"


def test_demodulation_recovery():
    """Verify coherent demodulation recovers transmitted baseband signal."""
    fs_rf = 600000
    fc = 250000
    bw = 4000
    t = np.arange(int(fs_rf * 0.5)) / fs_rf
    baseband = 0.6 * np.sin(2 * np.pi * 1500 * t)

    modulated, _, _ = modulate_ssb(baseband, fs=fs_rf, fc=fc, mode="usb")
    recovered = demodulate_ssb(modulated, fs=fs_rf, fc=fc, mode="usb", bw=bw)

    snr = evaluate_demodulation(baseband, recovered)
    assert snr > 20.0, f"Expected recovery SNR > 20 dB, got {snr:.2f} dB"


def test_anti_aliasing_lpf_removes_high_frequency():
    """4 kHz anti-aliasing LPF must suppress content above the RF bandwidth limit."""
    fs = 60000
    t = np.arange(int(fs * 0.25)) / fs
    high = np.sin(2 * np.pi * 8000 * t)
    filtered = apply_anti_aliasing_lpf(high, fs, cutoff=4000)

    freqs = np.fft.rfftfreq(len(high), 1 / fs)
    orig_e = np.sum(np.abs(np.fft.rfft(high))[freqs > 6000] ** 2)
    filt_e = np.sum(np.abs(np.fft.rfft(filtered))[freqs > 6000] ** 2)
    atten_db = 10 * np.log10((orig_e + 1e-20) / (filt_e + 1e-20))
    assert atten_db > 40.0, f"High-frequency residual only {atten_db:.1f} dB down"
