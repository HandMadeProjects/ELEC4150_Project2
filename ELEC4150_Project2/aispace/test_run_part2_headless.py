import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    AUDIO_FILE,
    TARGET_FS,
    BIT_RATE,
    CARRIER_FREQ,
    MAX_BANDWIDTH,
    CHANNEL_SNR_DB,
    RANDOM_SEED,
)
from part1.audio_io import load_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
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


def run_part2_tests():
    print("=" * 60)
    print("ELEC4150/8150 — PART 2 HEADLESS VERIFICATION TEST")
    print("=" * 60)

    output_dir = os.path.join(PROJECT_ROOT, "outputs", "part-2")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 0: Load and prepare vocal from Part 1
    # -------------------------------------------------------------
    print("\n[Step 0] Loading audio and extracting vocal...")
    audio_path = os.path.join(PROJECT_ROOT, AUDIO_FILE)
    audio, orig_fs = load_audio(audio_path)
    audio = skip_leading_silence(audio, orig_fs)

    # Use 8 seconds of actual audio (after leading silence)
    test_duration = 8.0
    audio_trimmed = audio[: int(orig_fs * test_duration)]
    
    left_60k = resample_audio(audio_trimmed[:, 0], orig_fs, TARGET_FS)
    right_60k = resample_audio(audio_trimmed[:, 1], orig_fs, TARGET_FS)
    audio_60k = np.stack([left_60k, right_60k], axis=1)
    
    vocal_60k, _, _ = separate_vocal_and_karaoke(audio_60k, TARGET_FS)
    print(f"Extracted vocal duration: {len(vocal_60k)/TARGET_FS:.2f} s at {TARGET_FS} Hz")

    passed_checks = 0
    total_checks = 0

    # -------------------------------------------------------------
    # Step 1: Compression (1 mark)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 1: Compression to 64 kbps")
    print("-" * 50)
    total_checks += 1
    
    compressed_bytes, decoded_vocal, fs_out, actual_bitrate = compress_vocal(
        vocal_60k, TARGET_FS, target_bitrate=BIT_RATE
    )
    decoded_vocal = apply_anti_aliasing_lpf(
        decoded_vocal, TARGET_FS, cutoff=MAX_BANDWIDTH
    )
    print(f"Target Bitrate : {BIT_RATE} bps (64 kbps)")
    print(f"Actual Bitrate : {actual_bitrate:.1f} bps ({actual_bitrate/1000:.1f} kbps)")
    comp_snr = compression_snr_db(vocal_60k, decoded_vocal)
    print(f"Compression SNR: {comp_snr:.2f} dB")

    if abs(actual_bitrate - BIT_RATE) < 1.0:
        print(">> Check 1: Compression bitrate == 64 kbps: PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 1: Compression bitrate == 64 kbps: FAIL ✗ ({actual_bitrate} bps)")

    # -------------------------------------------------------------
    # Step 2: Modulation (2 marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 2: Modulation at fc=250 kHz, BW <= 4 kHz")
    print("-" * 50)
    total_checks += 2

    # Upsample decoded baseband to RF simulation sampling rate (600 kHz)
    # Nyquist for 250 kHz carrier + 4 kHz is 508 kHz, so 600 kHz is valid.
    FS_RF = 600_000
    vocal_rf = resample_poly(decoded_vocal, FS_RF // TARGET_FS, 1)
    print(f"RF Simulation fs: {FS_RF} Hz (upsampled from {TARGET_FS} Hz)")

    modulated_rf, t_rf, carrier = modulate_ssb(
        vocal_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb"
    )

    # Measure Bandwidth
    bw_hz, freqs_rf, psd_db = measure_bandwidth(
        modulated_rf, fs=FS_RF, fc=CARRIER_FREQ, threshold_db=-40
    )
    print(f"Carrier Frequency : {CARRIER_FREQ} Hz (250 kHz)")
    print(f"Measured Bandwidth: {bw_hz:.1f} Hz (Limit: <= {MAX_BANDWIDTH} Hz)")

    # Check carrier frequency
    peak_freq = freqs_rf[np.argmax(psd_db)]
    if abs(peak_freq - CARRIER_FREQ) <= MAX_BANDWIDTH:
        print(f">> Check 2a: Carrier centered near 250 kHz ({peak_freq:.0f} Hz): PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 2a: Carrier near 250 kHz: FAIL ✗ ({peak_freq} Hz)")

    # Check bandwidth limit
    if bw_hz <= MAX_BANDWIDTH + 100:  # small margin for FFT bin quantization
        print(f">> Check 2b: Bandwidth <= 4 kHz ({bw_hz:.1f} Hz): PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 2b: Bandwidth <= 4 kHz: FAIL ✗ ({bw_hz:.1f} Hz)")

    # Plot and save modulated spectrum
    fig_mod, ax_mod = plt.subplots(figsize=(10, 4))
    mask = (freqs_rf >= CARRIER_FREQ - 10000) & (freqs_rf <= CARRIER_FREQ + 15000)
    ax_mod.plot(freqs_rf[mask] / 1000.0, psd_db[mask], color="crimson", linewidth=1.2)
    ax_mod.axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--", label=f"Carrier ({CARRIER_FREQ/1000} kHz)")
    ax_mod.axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":", label=f"Max BW limit (+{MAX_BANDWIDTH/1000} kHz)")
    ax_mod.set_title(f"Modulated RF Spectrum (SSB-AM USB) — Occupied BW = {bw_hz:.0f} Hz (Limit: <= 4000 Hz)")
    ax_mod.set_xlabel("Frequency (kHz)")
    ax_mod.set_ylabel("Normalized PSD (dB)")
    ax_mod.set_ylim(-60, 5)
    ax_mod.grid(True, alpha=0.5)
    ax_mod.legend()
    fig_mod.tight_layout()
    fig_mod_path = os.path.join(output_dir, "modulated_spectrum.png")
    fig_mod.savefig(fig_mod_path, dpi=150)
    plt.close(fig_mod)
    print(f"Saved: {fig_mod_path}")

    # -------------------------------------------------------------
    # Step 2c: Clean demodulation (validate the link before noise)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 3a: Clean demodulation (no noise)")
    print("-" * 50)
    total_checks += 1
    demod_clean = demodulate_ssb(
        modulated_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=MAX_BANDWIDTH
    )
    clean_60k = resample_poly(demod_clean, 1, FS_RF // TARGET_FS)[: len(vocal_60k)]
    peak_ref = np.max(np.abs(decoded_vocal))
    peak_clean = np.max(np.abs(clean_60k))
    if peak_clean > 0:
        clean_60k = clean_60k * (peak_ref / peak_clean)
    clean_snr = evaluate_demodulation(decoded_vocal, clean_60k)
    print(f"Clean-link recovered SNR: {clean_snr:.2f} dB")
    wavfile.write(
        os.path.join(output_dir, "clean_transmitted_vocal.wav"),
        TARGET_FS,
        (np.clip(clean_60k, -1, 1) * 32767).astype(np.int16),
    )
    if clean_snr > 15.0:
        print(">> Check 2c: Clean demodulation validates transmission: PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 2c: Clean demodulation: FAIL ✗ ({clean_snr:.2f} dB)")

    # -------------------------------------------------------------
    # Step 3: Noise Addition (SNR = 10 dB)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 4a: Add AWGN at SNR = 10 dB")
    print("-" * 50)
    total_checks += 1

    noisy_rf, noise, actual_snr = add_awgn(
        modulated_rf, snr_db=CHANNEL_SNR_DB, seed=RANDOM_SEED
    )
    print(f"Target AWGN SNR: {CHANNEL_SNR_DB} dB")
    print(f"Actual AWGN SNR: {actual_snr:.2f} dB")

    if abs(actual_snr - CHANNEL_SNR_DB) < 0.5:
        print(">> Check 3: AWGN at SNR = 10 dB: PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 3: AWGN at SNR = 10 dB: FAIL ✗ ({actual_snr:.2f} dB)")

    # -------------------------------------------------------------
    # Step 4: Noise Countermeasure & Evaluation (3 marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 4b: Countermeasure & Performance Evaluation")
    print("-" * 50)
    total_checks += 1

    cleaned_rf, snr_before, snr_after, improvement_db = apply_countermeasure_pipeline(
        noisy_rf, modulated_rf, fs_rf=FS_RF, fc=CARRIER_FREQ, bw=MAX_BANDWIDTH, mode="usb"
    )
    print(f"RF SNR Before Countermeasure : {snr_before:.2f} dB")
    print(f"RF SNR After Countermeasure  : {snr_after:.2f} dB")
    print(f"RF SNR Improvement           : +{improvement_db:.2f} dB")

    if improvement_db > 5.0:
        print(f">> Check 4: Countermeasure SNR improvement (+{improvement_db:.2f} dB > +5 dB): PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 4: Countermeasure improvement: FAIL ✗ ({improvement_db:.2f} dB)")

    # Plot Countermeasure Comparison
    fig_cm, axes_cm = plt.subplots(2, 1, figsize=(11, 6))
    peak_idx = int(np.argmax(np.abs(modulated_rf)))
    rf_half = 800
    rf_start = max(0, peak_idx - rf_half)
    t_slice = slice(rf_start, rf_start + 2 * rf_half)
    axes_cm[0].plot(t_rf[t_slice] * 1000, noisy_rf[t_slice], color="red", alpha=0.7, label="Noisy RF (SNR=10dB)")
    axes_cm[0].plot(t_rf[t_slice] * 1000, cleaned_rf[t_slice], color="green", alpha=0.8, label="Cleaned RF (BPF Countermeasure)")
    axes_cm[0].plot(t_rf[t_slice] * 1000, modulated_rf[t_slice], color="black", linestyle="--", alpha=0.5, label="Clean RF Reference")
    axes_cm[0].set_title(f"Countermeasure Time Domain — RF SNR: {snr_before:.1f} dB -> {snr_after:.1f} dB (+{improvement_db:.1f} dB improvement)")
    axes_cm[0].set_xlabel("Time (ms)")
    axes_cm[0].set_ylabel("Amplitude")
    axes_cm[0].legend(loc="upper right")
    axes_cm[0].grid(True, alpha=0.5)

    # Spectrum comparison
    _, _, psd_noisy = measure_bandwidth(noisy_rf, fs=FS_RF, fc=CARRIER_FREQ)
    _, _, psd_clean_rf = measure_bandwidth(cleaned_rf, fs=FS_RF, fc=CARRIER_FREQ)
    axes_cm[1].plot(freqs_rf[mask] / 1000.0, psd_noisy[mask], color="red", alpha=0.5, label="Noisy RF Spectrum")
    axes_cm[1].plot(freqs_rf[mask] / 1000.0, psd_clean_rf[mask], color="green", alpha=0.8, label="Filtered RF Spectrum")
    axes_cm[1].axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--")
    axes_cm[1].axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":")
    axes_cm[1].set_title("RF Spectrum Before vs After Bandpass Countermeasure")
    axes_cm[1].set_xlabel("Frequency (kHz)")
    axes_cm[1].set_ylabel("PSD (dB)")
    axes_cm[1].set_ylim(-60, 5)
    axes_cm[1].legend(loc="upper right")
    axes_cm[1].grid(True, alpha=0.5)

    fig_cm.tight_layout()
    fig_cm_path = os.path.join(output_dir, "noise_countermeasure_comparison.png")
    fig_cm.savefig(fig_cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"Saved: {fig_cm_path}")

    # -------------------------------------------------------------
    # Step 5: Demodulation & Audio Recovery (1 mark)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 3: Demodulation & Audio Recovery")
    print("-" * 50)
    total_checks += 1

    # Demodulate cleaned RF
    demod_rf = demodulate_ssb(cleaned_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=MAX_BANDWIDTH)
    
    # Downsample back to TARGET_FS (60 kHz)
    recovered_raw_60k = resample_poly(demod_rf, 1, FS_RF // TARGET_FS)[: len(vocal_60k)]

    # Baseband Wiener denoising (Stage 2 countermeasure) at TARGET_FS
    recovered_60k = apply_wiener_denoise(recovered_raw_60k, fs=TARGET_FS, bw_signal=MAX_BANDWIDTH)
    
    # Match peak scale for direct waveform visualization
    peak_ref = np.max(np.abs(decoded_vocal))
    peak_rec = np.max(np.abs(recovered_60k))
    if peak_rec > 0:
        recovered_60k = recovered_60k * (peak_ref / peak_rec)

    # Demodulation quality
    demod_snr = evaluate_demodulation(decoded_vocal, recovered_60k)
    print(f"Recovered Audio SNR relative to pre-mod vocal: {demod_snr:.2f} dB")

    # Demodulate unmitigated noisy RF for audio comparison
    demod_raw_noisy = demodulate_ssb(noisy_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=30_000)
    recovered_noisy_60k = resample_poly(demod_raw_noisy, 1, FS_RF // TARGET_FS)[: len(vocal_60k)]
    peak_noisy = np.max(np.abs(recovered_noisy_60k))
    if peak_noisy > 0:
        recovered_noisy_60k = recovered_noisy_60k * (peak_ref / peak_noisy)
    noisy_snr = evaluate_demodulation(decoded_vocal, recovered_noisy_60k)
    print(f"Noisy demod SNR (no countermeasure): {noisy_snr:.2f} dB")
    print(f"Audio SNR gain from countermeasure : +{demod_snr - noisy_snr:.2f} dB")

    # Save recovered audio files
    recovered_wav_path = os.path.join(output_dir, "recovered_vocal.wav")
    noisy_wav_path = os.path.join(output_dir, "noisy_recovered_vocal.wav")
    wavfile.write(recovered_wav_path, TARGET_FS, (recovered_60k * 32767).astype(np.int16))
    wavfile.write(noisy_wav_path, TARGET_FS, (recovered_noisy_60k * 32767).astype(np.int16))
    print(f"Saved: {recovered_wav_path}")
    print(f"Saved: {noisy_wav_path}")

    if len(recovered_60k) == len(vocal_60k) and demod_snr > 0:
        print(">> Check 5: Demodulation and audio recovery: PASS ✓")
        passed_checks += 1
    else:
        print(">> Check 5: Demodulation and audio recovery: FAIL ✗")

    # Plot Recovered Audio Comparison
    fig_rec, ax_rec = plt.subplots(figsize=(11, 4))
    plot_start = int(0.25 * TARGET_FS)
    plot_n = min(12000, len(vocal_60k) - plot_start)
    t_audio = np.arange(plot_n) / TARGET_FS
    sl = slice(plot_start, plot_start + plot_n)
    ax_rec.plot(t_audio * 1000, decoded_vocal[sl], label="Original Vocal (Decoded from 64 kbps)", color="blue", alpha=0.7)
    ax_rec.plot(t_audio * 1000, recovered_noisy_60k[sl], label=f"Noisy demod ({noisy_snr:.1f} dB)", color="red", alpha=0.4)
    ax_rec.plot(t_audio * 1000, recovered_60k[sl], label=f"After countermeasure ({demod_snr:.1f} dB)", color="green", alpha=0.85)
    ax_rec.set_title("Demodulated vocal — noise influence vs countermeasure")
    ax_rec.set_xlabel("Time (ms)")
    ax_rec.set_ylabel("Amplitude")
    ax_rec.legend(loc="upper right")
    ax_rec.grid(True, alpha=0.5)
    fig_rec.tight_layout()
    fig_rec_path = os.path.join(output_dir, "recovered_vocal_waveform.png")
    fig_rec.savefig(fig_rec_path, dpi=150)
    plt.close(fig_rec)
    print(f"Saved: {fig_rec_path}")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PART 2 VERIFICATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    print("=" * 60)
    return passed_checks == total_checks


if __name__ == "__main__":
    success = run_part2_tests()
    sys.exit(0 if success else 1)
