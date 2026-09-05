"""
ELEC4150/8150 Project 2 — Part 2 Live Runner
============================================
Spec order:
  1. Compress Part 1 vocal to 64 kbps
  2. SSB-AM onto 250 kHz, occupied BW <= 4 kHz (show spectrum shift)
  3. Demodulate the CLEAN RF and play it (validate the link)
  4. Add AWGN at SNR = 10 dB; show it on RF and on the demodulated audio
  5. Countermeasures (RF bandpass + baseband Wiener); verify before/after

Usage:
  python run_part2.py
  python run_part2.py --duration 30
  python run_part2.py --audio audio_1.wav --duration 20
"""

import os
import argparse
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import scipy.io.wavfile as wavfile

from config import (
    AUDIO_FILE,
    TARGET_FS,
    AUDIO_MAX_DURATION,
    BIT_RATE,
    CARRIER_FREQ,
    MAX_BANDWIDTH,
    CHANNEL_SNR_DB,
    RANDOM_SEED,
)
from part1.audio_io import load_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
from part2.compression import compress_vocal, compression_snr_db
from part2.modulation import modulate_ssb, measure_bandwidth
from part2.demodulation import demodulate_ssb, evaluate_demodulation
from part2.noise import add_awgn
from part2.filtering import (
    apply_anti_aliasing_lpf,
    apply_countermeasure_pipeline,
    apply_wiener_denoise,
)

OUTPUTS_DIR = os.path.join("outputs", "part-2")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
FS_RF = 600_000

parser = argparse.ArgumentParser(description="Run ELEC4150 Project 2 Part 2")
parser.add_argument(
    "--audio",
    default=AUDIO_FILE,
    help=f"Audio file to process (default: {AUDIO_FILE})",
)
parser.add_argument(
    "--duration",
    type=float,
    default=25.0,
    help="Seconds of song to process after leading silence (default: 25, max 30)",
)
args, _ = parser.parse_known_args()
DEMO_DURATION = min(float(args.duration), float(AUDIO_MAX_DURATION))


def _psd_db(x, fs):
    n = len(x)
    window = np.hanning(n)
    spec = np.fft.rfft(x * window)
    psd = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd_db = 10.0 * np.log10(psd / (np.max(psd) + 1e-20) + 1e-20)
    return freqs, psd_db


def _match_peak(sig, ref):
    peak_ref = np.max(np.abs(ref))
    peak = np.max(np.abs(sig))
    if peak > 0 and peak_ref > 0:
        return sig * (peak_ref / peak)
    return sig


def _demod_to_60k(rf, n_out, bw=MAX_BANDWIDTH):
    demod = demodulate_ssb(
        rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=bw
    )
    audio = resample_poly(demod, 1, FS_RF // TARGET_FS)[:n_out]
    return audio


def _play(sig, seconds, label):
    n = int(min(seconds, len(sig) / TARGET_FS) * TARGET_FS)
    print(f">> Playing: {label} ({n / TARGET_FS:.1f} s)")
    try:
        sd.play(np.asarray(sig[:n], dtype=np.float32), TARGET_FS)
        sd.wait()
    except Exception as exc:
        print(f"Audio playback skipped: {exc}")


def _save_wav(path, sig):
    peak = np.max(np.abs(sig))
    scaled = sig / peak if peak > 1.0 else sig
    wavfile.write(path, TARGET_FS, (np.clip(scaled, -1, 1) * 32767).astype(np.int16))
    print(f"Saved: {path}")


print("=" * 65)
print("ELEC4150 PROJECT 2 — PART 2: TRANSMITTING SIGNAL OVER RF")
print(f"Audio : {args.audio}")
print(f"Demo  : {DEMO_DURATION:.0f} s of song (after leading silence)")
print("=" * 65)

# ------------------------------------------------------------
# Load Part 1 vocal
# ------------------------------------------------------------
print("\n[Prep] Extracting Part 1 vocal...")
audio, orig_fs = load_audio(args.audio)
audio = skip_leading_silence(audio, orig_fs)
audio = audio[: int(orig_fs * DEMO_DURATION)]
left_60k = resample_audio(audio[:, 0], orig_fs, TARGET_FS)
right_60k = resample_audio(audio[:, 1], orig_fs, TARGET_FS)
vocal, _, _ = separate_vocal_and_karaoke(
    np.stack([left_60k, right_60k], axis=1), TARGET_FS
)
print(f"Vocal : {len(vocal) / TARGET_FS:.2f} s at {TARGET_FS} Hz")

# ------------------------------------------------------------
# 1. Compress to 64 kbps
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[1] Compress vocal to 64 kbps  (ITU-T G.711 μ-law, 8 bit × 8 kHz)")
print("-" * 65)
compressed_bytes, decoded_vocal, _, actual_bitrate = compress_vocal(
    vocal, TARGET_FS, target_bitrate=BIT_RATE
)
decoded_vocal = apply_anti_aliasing_lpf(
    decoded_vocal, TARGET_FS, cutoff=MAX_BANDWIDTH
)
comp_snr = compression_snr_db(vocal, decoded_vocal)
print(f"Target bitrate : {BIT_RATE} bps (64 kbps)")
print(f"Actual bitrate : {actual_bitrate:.1f} bps ({actual_bitrate / 1000:.1f} kbps)")
print(f"Compression SNR: {comp_snr:.2f} dB")
if abs(actual_bitrate - BIT_RATE) < 1.0:
    print("64 kbps requirement: PASS")
else:
    print("64 kbps requirement: FAIL")

# ------------------------------------------------------------
# 2. Modulate onto 250 kHz, BW <= 4 kHz
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[2] SSB-AM USB modulation  (fc = 250 kHz, BW <= 4 kHz)")
print("-" * 65)
vocal_rf = resample_poly(decoded_vocal, FS_RF // TARGET_FS, 1)
modulated_rf, t_rf, _ = modulate_ssb(
    vocal_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb"
)
bw_hz, freqs_rf, psd_rf = measure_bandwidth(
    modulated_rf, fs=FS_RF, fc=CARRIER_FREQ, threshold_db=-40
)
print(f"Carrier          : {CARRIER_FREQ} Hz")
print(f"Occupied BW      : {bw_hz:.1f} Hz  (limit {MAX_BANDWIDTH} Hz)")
print("Spectrum move    : baseband 0–4 kHz  →  USB 250–254 kHz")
if bw_hz <= MAX_BANDWIDTH + 100:
    print("<= 4 kHz bandwidth: PASS")
else:
    print("<= 4 kHz bandwidth: FAIL")

# ------------------------------------------------------------
# 3. Clean demodulation — validate the link before noise
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[3] Demodulate CLEAN RF and play  (validate successful transmission)")
print("-" * 65)
clean_60k = _match_peak(_demod_to_60k(modulated_rf, len(vocal)), decoded_vocal)
clean_snr = evaluate_demodulation(decoded_vocal, clean_60k)
print(f"Clean-link recovered SNR: {clean_snr:.2f} dB")
_save_wav(os.path.join(OUTPUTS_DIR, "clean_transmitted_vocal.wav"), clean_60k)
_play(clean_60k, min(10.0, DEMO_DURATION), "clean demodulated vocal (no noise)")

# ------------------------------------------------------------
# 4. AWGN at 10 dB — influence on transmitted and demodulated
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[4] AWGN channel at SNR = 10 dB")
print("-" * 65)
noisy_rf, _, actual_snr = add_awgn(
    modulated_rf, snr_db=CHANNEL_SNR_DB, seed=RANDOM_SEED
)
print(f"Measured RF SNR : {actual_snr:.2f} dB  (target {CHANNEL_SNR_DB} dB)")

# Wide LPF = receiver with no channel-matched filtering (noise floods baseband)
noisy_60k = _match_peak(
    _demod_to_60k(noisy_rf, len(vocal), bw=30_000), decoded_vocal
)
noisy_snr = evaluate_demodulation(decoded_vocal, noisy_60k)
print(f"Demodulated SNR with noise, no RF selectivity: {noisy_snr:.2f} dB")
_save_wav(os.path.join(OUTPUTS_DIR, "noisy_recovered_vocal.wav"), noisy_60k)

# ------------------------------------------------------------
# 5. Countermeasures and verification
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[5] Countermeasures: RF bandpass + baseband Wiener")
print("-" * 65)
cleaned_rf, snr_before, snr_after, improvement_db = apply_countermeasure_pipeline(
    noisy_rf, modulated_rf, fs_rf=FS_RF, fc=CARRIER_FREQ, bw=MAX_BANDWIDTH, mode="usb"
)
print(f"RF SNR before BPF : {snr_before:.2f} dB")
print(f"RF SNR after  BPF : {snr_after:.2f} dB")
print(f"RF SNR improvement: +{improvement_db:.2f} dB")

recovered_raw = _demod_to_60k(cleaned_rf, len(vocal))
recovered_60k = apply_wiener_denoise(
    recovered_raw, fs=TARGET_FS, bw_signal=MAX_BANDWIDTH
)
recovered_60k = _match_peak(recovered_60k, decoded_vocal)
recovered_snr = evaluate_demodulation(decoded_vocal, recovered_60k)
print(f"Demodulated SNR after countermeasure: {recovered_snr:.2f} dB")
print(
    f"Audio SNR gain vs noisy demod       : "
    f"+{recovered_snr - noisy_snr:.2f} dB"
)
_save_wav(os.path.join(OUTPUTS_DIR, "recovered_vocal.wav"), recovered_60k)

# ------------------------------------------------------------
# Figures — one panel per spec requirement
# ------------------------------------------------------------
print("\n[Plots] Baseband vs RF, noise on RF, countermeasure, recovered audio")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

bb_f, bb_db = _psd_db(decoded_vocal, TARGET_FS)
axes[0, 0].plot(bb_f / 1000.0, bb_db, color="#3182ce", linewidth=1.0)
axes[0, 0].axvline(MAX_BANDWIDTH / 1000.0, color="gray", linestyle=":", label="4 kHz cap")
axes[0, 0].set_xlim(0, 10)
axes[0, 0].set_ylim(-80, 5)
axes[0, 0].set_title("Before modulation: compressed vocal (baseband)")
axes[0, 0].set_xlabel("Frequency (kHz)")
axes[0, 0].set_ylabel("PSD (dB)")
axes[0, 0].legend(loc="upper right")
axes[0, 0].grid(True, alpha=0.4)

rf_mask = (freqs_rf >= CARRIER_FREQ - 10000) & (freqs_rf <= CARRIER_FREQ + 15000)
axes[0, 1].plot(freqs_rf[rf_mask] / 1000.0, psd_rf[rf_mask], color="crimson", linewidth=1.1)
axes[0, 1].axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--", label="fc = 250 kHz")
axes[0, 1].axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":", label="+4 kHz")
axes[0, 1].set_ylim(-60, 5)
axes[0, 1].set_title(f"After SSB-AM USB: RF spectrum  (BW = {bw_hz:.0f} Hz)")
axes[0, 1].set_xlabel("Frequency (kHz)")
axes[0, 1].set_ylabel("PSD (dB)")
axes[0, 1].legend(loc="upper right")
axes[0, 1].grid(True, alpha=0.4)

_, _, psd_noisy = measure_bandwidth(noisy_rf, fs=FS_RF, fc=CARRIER_FREQ)
_, _, psd_clean_rf = measure_bandwidth(cleaned_rf, fs=FS_RF, fc=CARRIER_FREQ)
axes[1, 0].plot(freqs_rf[rf_mask] / 1000.0, psd_noisy[rf_mask], color="red", alpha=0.55, label="Noisy RF (10 dB SNR)")
axes[1, 0].plot(freqs_rf[rf_mask] / 1000.0, psd_clean_rf[rf_mask], color="green", linewidth=1.1, label=f"After BPF (+{improvement_db:.1f} dB)")
axes[1, 0].axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--")
axes[1, 0].axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":")
axes[1, 0].set_ylim(-60, 5)
axes[1, 0].set_title("Noise on transmitted RF, then bandpass countermeasure")
axes[1, 0].set_xlabel("Frequency (kHz)")
axes[1, 0].set_ylabel("PSD (dB)")
axes[1, 0].legend(loc="upper right")
axes[1, 0].grid(True, alpha=0.4)

plot_start = int(0.25 * TARGET_FS)
plot_n = min(12000, len(vocal) - plot_start)
t_ms = np.arange(plot_n) / TARGET_FS * 1000.0
sl = slice(plot_start, plot_start + plot_n)
axes[1, 1].plot(t_ms, decoded_vocal[sl], color="blue", alpha=0.8, label="64 kbps vocal")
axes[1, 1].plot(t_ms, noisy_60k[sl], color="red", alpha=0.45, label=f"Noisy demod ({noisy_snr:.1f} dB)")
axes[1, 1].plot(t_ms, recovered_60k[sl], color="green", alpha=0.9, label=f"After countermeasure ({recovered_snr:.1f} dB)")
axes[1, 1].set_title("Noise on demodulated audio vs countermeasure")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].legend(loc="upper right")
axes[1, 1].grid(True, alpha=0.4)

fig.tight_layout()
summary_path = os.path.join(OUTPUTS_DIR, "part2_summary_plots.png")
fig.savefig(summary_path, dpi=150)
print(f"Saved: {summary_path}")

fig_mod, ax_mod = plt.subplots(figsize=(10, 4))
ax_mod.plot(freqs_rf[rf_mask] / 1000.0, psd_rf[rf_mask], color="crimson", linewidth=1.2, label="SSB-AM USB")
ax_mod.axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--", label="fc = 250 kHz")
ax_mod.axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":", label="+4 kHz limit")
ax_mod.set_title(f"Modulated RF spectrum — vocal moved to 250 kHz USB  (BW = {bw_hz:.0f} Hz ≤ 4 kHz)")
ax_mod.set_xlabel("Frequency (kHz)")
ax_mod.set_ylabel("PSD (dB)")
ax_mod.set_ylim(-60, 5)
ax_mod.legend(loc="upper right")
ax_mod.grid(True, alpha=0.4)
fig_mod.tight_layout()
fig_mod.savefig(os.path.join(OUTPUTS_DIR, "modulated_spectrum.png"), dpi=150)
print(f"Saved: {os.path.join(OUTPUTS_DIR, 'modulated_spectrum.png')}")

fig_cm, axes_cm = plt.subplots(2, 1, figsize=(11, 6))
peak_idx = int(np.argmax(np.abs(modulated_rf)))
rf_half = 800
rf_start = max(0, peak_idx - rf_half)
t_slice = slice(rf_start, rf_start + 2 * rf_half)
axes_cm[0].plot(t_rf[t_slice] * 1000, noisy_rf[t_slice], color="red", alpha=0.7, label="Noisy RF (10 dB)")
axes_cm[0].plot(t_rf[t_slice] * 1000, cleaned_rf[t_slice], color="green", alpha=0.85, label="After RF BPF")
axes_cm[0].plot(t_rf[t_slice] * 1000, modulated_rf[t_slice], color="black", linestyle="--", alpha=0.5, label="Clean RF")
axes_cm[0].set_title(f"Transmitted RF: noise vs countermeasure  ({snr_before:.1f} → {snr_after:.1f} dB)")
axes_cm[0].set_xlabel("Time (ms)")
axes_cm[0].set_ylabel("Amplitude")
axes_cm[0].legend(loc="upper right")
axes_cm[0].grid(True, alpha=0.4)
axes_cm[1].plot(freqs_rf[rf_mask] / 1000.0, psd_noisy[rf_mask], color="red", alpha=0.5, label="Noisy RF")
axes_cm[1].plot(freqs_rf[rf_mask] / 1000.0, psd_clean_rf[rf_mask], color="green", linewidth=1.1, label="After BPF")
axes_cm[1].axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--")
axes_cm[1].axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":")
axes_cm[1].set_title("RF spectrum before vs after bandpass countermeasure")
axes_cm[1].set_xlabel("Frequency (kHz)")
axes_cm[1].set_ylabel("PSD (dB)")
axes_cm[1].set_ylim(-60, 5)
axes_cm[1].legend(loc="upper right")
axes_cm[1].grid(True, alpha=0.4)
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUTS_DIR, "noise_countermeasure_comparison.png"), dpi=150)
print(f"Saved: {os.path.join(OUTPUTS_DIR, 'noise_countermeasure_comparison.png')}")

fig_rec, ax_rec = plt.subplots(figsize=(11, 4))
ax_rec.plot(t_ms, decoded_vocal[sl], color="blue", alpha=0.8, label="64 kbps vocal")
ax_rec.plot(t_ms, noisy_60k[sl], color="red", alpha=0.4, label=f"Noisy demod ({noisy_snr:.1f} dB)")
ax_rec.plot(t_ms, recovered_60k[sl], color="green", alpha=0.9, label=f"Recovered ({recovered_snr:.1f} dB)")
ax_rec.set_title("Demodulated vocal — noise influence vs countermeasure")
ax_rec.set_xlabel("Time (ms)")
ax_rec.set_ylabel("Amplitude")
ax_rec.legend(loc="upper right")
ax_rec.grid(True, alpha=0.4)
fig_rec.tight_layout()
fig_rec.savefig(os.path.join(OUTPUTS_DIR, "recovered_vocal_waveform.png"), dpi=150)
print(f"Saved: {os.path.join(OUTPUTS_DIR, 'recovered_vocal_waveform.png')}")

plt.show(block=False)

print("\n" + "-" * 65)
print("Listening check: noisy demod (noise influence), then recovered")
print("-" * 65)
_play(noisy_60k, min(8.0, DEMO_DURATION), "noisy demodulated vocal (AWGN, no countermeasure)")
_play(recovered_60k, DEMO_DURATION, "recovered vocal after countermeasure")

plt.show()
print("\nPart 2 complete.")
