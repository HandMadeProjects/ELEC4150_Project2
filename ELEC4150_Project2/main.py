"""
ELEC4150/8150 Project 2 — Part 1 Main Entry Point
==================================================
Vocal extraction pipeline:
  1. Load & validate stereo audio
  2. Resample to 60 kHz
  3. Extract vocal and instrumental components
  4. Display time-domain waveform
  5. Real-time spectrogram with simultaneous audio playback
  6. Static spectrum comparison (before vs after extraction)
  7. Save all figures to outputs/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interactive backend — change to "Agg" for headless
import matplotlib.pyplot as plt

from config import (
    TARGET_FS,
    AUDIO_MAX_DURATION,
    AUDIO_FILE,
    BIT_RATE,
    CARRIER_FREQ,
    MAX_BANDWIDTH,
    CHANNEL_SNR_DB,
    SPEED_OF_SOUND,
    MIC_RADIUS,
    RANDOM_SEED,
)

from part1.audio_io import load_audio, validate_audio, save_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
from part1.spectrogram import calculate_spectrogram
from part1.realtime_spectrogram import play_with_realtime_spectrogram
from part1.plotting import plot_waveform

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

from part3.microphone_model import (
    create_circular_array,
    generate_random_source,
    calculate_distances,
    apply_inverse_square_attenuation,
    simulate_microphone_array,
)
from part3.delay import (
    fractional_delay,
    calculate_propagation_delays,
    apply_propagation_delays,
)
from part3.localisation import (
    estimate_tdoa_gcc_phat,
    compute_pairwise_tdoas,
    localise_source_nls,
    steered_response_power_map,
    evaluate_localisation_accuracy,
)
from scipy.signal import resample_poly
import scipy.io.wavfile as wavfile

# ============================================================
# PATHS
# ============================================================

AUDIO_FILE_PATH = AUDIO_FILE
OUTPUTS_DIR = os.path.join("outputs", "part-1")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD & VALIDATE AUDIO
# ============================================================

print("=" * 50)
print("STEP 1 — Load Audio")
print("=" * 50)

audio, original_fs = load_audio(AUDIO_FILE_PATH)
audio = skip_leading_silence(audio, original_fs)

print(f"Original sampling rate : {original_fs} Hz")
print(f"Original samples       : {len(audio)}")
print(f"Original duration      : {len(audio) / original_fs:.2f} s")

validate_audio(audio, original_fs)


# ============================================================
# STEP 2: TRIM TO AUDIO_MAX_DURATION SECONDS
# ============================================================

print("\n" + "=" * 50)
print("STEP 2 — Trim Audio")
print("=" * 50)

max_samples = int(AUDIO_MAX_DURATION * original_fs)
audio = audio[:max_samples]

left  = audio[:, 0]
right = audio[:, 1]

print(f"Trimmed to {len(left) / original_fs:.2f} seconds")


# ============================================================
# STEP 3: RESAMPLE BOTH CHANNELS TO TARGET_FS (60 kHz)
# ============================================================

print("\n" + "=" * 50)
print("STEP 3 — Resample to 60 kHz")
print("=" * 50)

left_60k  = resample_audio(left,  original_fs, TARGET_FS)
right_60k = resample_audio(right, original_fs, TARGET_FS)

audio_60k = np.stack([left_60k, right_60k], axis=1)

print(f"Before : {original_fs} Hz")
print(f"After  : {TARGET_FS} Hz")

if TARGET_FS == 60000:
    print("60 kHz requirement: PASS ✓")
else:
    print("60 kHz requirement: FAIL ✗")

save_audio(os.path.join(OUTPUTS_DIR, "full_song_60k.wav"), audio_60k, TARGET_FS)
print("Saved: outputs/part-1/full_song_60k.wav")


# ============================================================
# STEP 4: VOCAL EXTRACTION  (simple centre channel)
# ============================================================

print("\n" + "=" * 50)
print("STEP 4 — Vocal Extraction (Centre + harmonic singing band)")
print("=" * 50)

vocal, karaoke, attenuation_db = separate_vocal_and_karaoke(audio_60k, TARGET_FS)
karaoke_mono = np.mean(karaoke, axis=1)

print("Centre-panned harmonic vocal extracted (bass/drums removed).")
save_audio(os.path.join(OUTPUTS_DIR, "extracted_vocal.wav"), vocal, TARGET_FS)
print("Saved: outputs/part-1/extracted_vocal.wav")


# ============================================================
# STEP 4b: KARAOKE EXTRACTION
#
# STFT masks: keep bass, drums and side-panned instruments;
# apply a 40 dB floor to centre-panned harmonic bins in the
# singing band (160–5000 Hz). Reconstruct as stereo.
# ============================================================

print("\n" + "=" * 50)
print("STEP 4b — Karaoke Extraction (vocal deleted, tune kept)")
print("=" * 50)

print(f"Vocal attenuation in karaoke track: {attenuation_db:.1f} dB")

if attenuation_db >= 40:
    print("≥ 40 dB requirement: PASS ✓")
else:
    print(f"≥ 40 dB requirement: FAIL ✗  (got {attenuation_db:.1f} dB)")

save_audio(os.path.join(OUTPUTS_DIR, "karaoke_instrumental.wav"), karaoke, TARGET_FS)
print("Saved: outputs/part-1/karaoke_instrumental.wav")


# ============================================================
# STEP 5: PLOT TIME-DOMAIN WAVEFORMS
# ============================================================

print("\n" + "=" * 50)
print("STEP 5 — Time-Domain Waveforms")
print("=" * 50)

# --- Vocal waveform ---
vocal_time = np.arange(len(vocal)) / TARGET_FS

fig_vocal, ax_vocal = plt.subplots(figsize=(12, 4))
ax_vocal.plot(vocal_time, vocal, linewidth=0.6)
ax_vocal.set_title("Extracted Vocal Component — Time Domain")
ax_vocal.set_xlabel("Time (s)")
ax_vocal.set_ylabel("Amplitude")
ax_vocal.set_xlim(0, vocal_time[-1])
ax_vocal.grid(True)
fig_vocal.tight_layout()
fig_vocal.savefig(os.path.join(OUTPUTS_DIR, "vocal_waveform.png"), dpi=150)
print("Saved: outputs/vocal_waveform.png")

# --- Karaoke waveform ---
inst_time = np.arange(len(karaoke_mono)) / TARGET_FS

fig_inst, ax_inst = plt.subplots(figsize=(12, 4))
ax_inst.plot(inst_time, karaoke_mono, linewidth=0.6, color="steelblue")
ax_inst.set_title(
    f"Karaoke Track — {attenuation_db:.1f} dB vocal attenuation"
)
ax_inst.set_xlabel("Time (s)")
ax_inst.set_ylabel("Amplitude")
ax_inst.set_xlim(0, inst_time[-1])
ax_inst.grid(True)
fig_inst.tight_layout()
fig_inst.savefig(os.path.join(OUTPUTS_DIR, "karaoke_waveform.png"), dpi=150)
print("Saved: outputs/part-1/karaoke_waveform.png")


# ============================================================
# STEP 6: SPECTRUM COMPARISON (before vs after)
# ============================================================

print("\n" + "=" * 50)
print("STEP 6 — Spectrum Comparison")
print("=" * 50)

original_mono = (left_60k + right_60k) / 2.0
N = min(len(original_mono), len(vocal), len(karaoke_mono))
frequencies = np.fft.rfftfreq(N, d=1 / TARGET_FS)

original_fft  = np.fft.rfft(original_mono[:N])
vocal_fft     = np.fft.rfft(vocal[:N])
karaoke_fft   = np.fft.rfft(karaoke_mono[:N])

original_db   = 20 * np.log10(np.abs(original_fft)  + 1e-12)
vocal_db      = 20 * np.log10(np.abs(vocal_fft)     + 1e-12)
karaoke_db    = 20 * np.log10(np.abs(karaoke_fft)   + 1e-12)

fig_spec, ax_spec = plt.subplots(figsize=(12, 6))
ax_spec.plot(frequencies, original_db,  label="Original (mono)",        alpha=0.7)
ax_spec.plot(frequencies, vocal_db,     label="Extracted Vocal",         linewidth=1.2)
ax_spec.plot(frequencies, karaoke_db,   label=f"Karaoke ({attenuation_db:.0f} dB atten.)",
             linewidth=1.2, color="steelblue")
ax_spec.set_title("Spectrum — Original vs Extracted Vocal vs Karaoke")
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("Magnitude (dB)")
ax_spec.set_xlim(0, 10000)
ax_spec.grid(True)
ax_spec.legend()
fig_spec.tight_layout()
fig_spec.savefig(os.path.join(OUTPUTS_DIR, "spectrum_comparison.png"), dpi=150)
print("Saved: outputs/part-1/spectrum_comparison.png")


# ============================================================
# STEP 7: STATIC VOCAL SPECTROGRAM
# ============================================================

print("\n" + "=" * 50)
print("STEP 7 — Static Vocal Spectrogram")
print("=" * 50)

freqs, times, mag_db = calculate_spectrogram(vocal, TARGET_FS)

fig_sg, ax_sg = plt.subplots(figsize=(12, 6))
img = ax_sg.pcolormesh(
    times, freqs, mag_db,
    shading="auto",
    cmap="magma",
    vmin=-80, vmax=0
)
ax_sg.set_title("Vocal Component — Spectrogram")
ax_sg.set_xlabel("Time (s)")
ax_sg.set_ylabel("Frequency (Hz)")
ax_sg.set_ylim(0, 10000)
fig_sg.colorbar(img, ax=ax_sg, label="Magnitude (dB)")
fig_sg.tight_layout()
fig_sg.savefig(os.path.join(OUTPUTS_DIR, "vocal_spectrogram.png"), dpi=150)
print("Saved: outputs/part-1/vocal_spectrogram.png")


# ============================================================
# STEP 8: REAL-TIME SPECTROGRAM + AUDIO PLAYBACK (Part 1)
# ============================================================
print("\n" + "=" * 50)
print("STEP 8 — Real-Time Audio + Spectrogram (Part 1)")
print("=" * 50)

# Optional interactive playback
try:
    play_with_realtime_spectrogram(vocal, TARGET_FS, duration=min(10, len(vocal)/TARGET_FS))
except Exception as e:
    print(f"Interactive audio playback skipped: {e}")


# ============================================================
# PART 2: RF TRANSMISSION PIPELINE
# ============================================================
OUTPUTS_DIR_P2 = os.path.join("outputs", "part-2")
os.makedirs(OUTPUTS_DIR_P2, exist_ok=True)

print("\n" + "=" * 50)
print("PART 2: RF TRANSMISSION PIPELINE")
print("=" * 50)

# ------------------------------------------------------------
# STEP 9: COMPRESSION TO 64 KBPS
# ------------------------------------------------------------
print("\n--- Step 9: Compression to 64 kbps ---")
compressed_bytes, decoded_vocal, _, actual_bitrate = compress_vocal(
    vocal, TARGET_FS, target_bitrate=BIT_RATE
)
decoded_vocal = apply_anti_aliasing_lpf(
    decoded_vocal, TARGET_FS, cutoff=MAX_BANDWIDTH
)
comp_snr = compression_snr_db(vocal, decoded_vocal)
print(f"Target Bitrate : {BIT_RATE} bps (64 kbps)")
print(f"Actual Bitrate : {actual_bitrate:.1f} bps ({actual_bitrate/1000:.1f} kbps)")
print(f"Compression SNR: {comp_snr:.2f} dB")

# ------------------------------------------------------------
# STEP 10: MODULATION (fc = 250 kHz, BW <= 4 kHz)
# ------------------------------------------------------------
print("\n--- Step 10: Modulation (SSB-AM USB) ---")
FS_RF = 600_000
vocal_rf = resample_poly(decoded_vocal, FS_RF // TARGET_FS, 1)
modulated_rf, t_rf, carrier = modulate_ssb(
    vocal_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb"
)
bw_hz, freqs_rf, psd_db = measure_bandwidth(
    modulated_rf, fs=FS_RF, fc=CARRIER_FREQ, threshold_db=-40
)
print(f"Carrier Frequency : {CARRIER_FREQ} Hz (250 kHz)")
print(f"Occupied Bandwidth: {bw_hz:.1f} Hz (Limit: <= {MAX_BANDWIDTH} Hz)")

# Plot Modulated Spectrum (baseband vs RF)
mask_rf = (freqs_rf >= CARRIER_FREQ - 10000) & (freqs_rf <= CARRIER_FREQ + 15000)
fig_mod, ax_mod = plt.subplots(figsize=(10, 4))
ax_mod.plot(freqs_rf[mask_rf] / 1000.0, psd_db[mask_rf], color="crimson", linewidth=1.2)
ax_mod.axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--", label=f"Carrier ({CARRIER_FREQ/1000} kHz)")
ax_mod.axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":", label=f"Max BW limit (+{MAX_BANDWIDTH/1000} kHz)")
ax_mod.set_title(f"Part 2: Modulated RF Spectrum (SSB-AM USB) — Occupied BW = {bw_hz:.0f} Hz (<= 4 kHz)")
ax_mod.set_xlabel("Frequency (kHz)")
ax_mod.set_ylabel("Normalized PSD (dB)")
ax_mod.set_ylim(-60, 5)
ax_mod.grid(True, alpha=0.5)
ax_mod.legend()
fig_mod.tight_layout()
fig_mod.savefig(os.path.join(OUTPUTS_DIR_P2, "modulated_spectrum.png"), dpi=150)
print("Saved: outputs/part-2/modulated_spectrum.png")

# Clean demodulation — validate the link before adding noise
print("\n--- Step 10b: Clean demodulation (validate transmission) ---")
demod_clean = demodulate_ssb(modulated_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=MAX_BANDWIDTH)
clean_60k = resample_poly(demod_clean, 1, FS_RF // TARGET_FS)[: len(vocal)]
peak_ref = np.max(np.abs(decoded_vocal))
if np.max(np.abs(clean_60k)) > 0:
    clean_60k = clean_60k * (peak_ref / np.max(np.abs(clean_60k)))
clean_snr = evaluate_demodulation(decoded_vocal, clean_60k)
print(f"Clean-link recovered SNR: {clean_snr:.2f} dB")

# ------------------------------------------------------------
# STEP 11: AWGN CHANNEL (SNR = 10 dB)
# ------------------------------------------------------------
print("\n--- Step 11: AWGN Noise Channel (SNR = 10 dB) ---")
noisy_rf, noise, actual_channel_snr = add_awgn(
    modulated_rf, snr_db=CHANNEL_SNR_DB, seed=RANDOM_SEED
)
print(f"Channel SNR: {actual_channel_snr:.2f} dB (Target: {CHANNEL_SNR_DB} dB)")

# ------------------------------------------------------------
# STEP 12: NOISE COUNTERMEASURE & EVALUATION
# ------------------------------------------------------------
print("\n--- Step 12: Countermeasure & Performance Evaluation ---")
cleaned_rf, snr_before, snr_after, improvement_db = apply_countermeasure_pipeline(
    noisy_rf, modulated_rf, fs_rf=FS_RF, fc=CARRIER_FREQ, bw=MAX_BANDWIDTH, mode="usb"
)
print(f"RF SNR Before Countermeasure : {snr_before:.2f} dB")
print(f"RF SNR After Countermeasure  : {snr_after:.2f} dB")
print(f"RF SNR Improvement           : +{improvement_db:.2f} dB")

# Plot Countermeasure Comparison
fig_cm, axes_cm = plt.subplots(2, 1, figsize=(11, 6))
peak_idx = int(np.argmax(np.abs(modulated_rf)))
rf_start = max(0, peak_idx - 800)
t_slice = slice(rf_start, rf_start + 1600)
axes_cm[0].plot(t_rf[t_slice] * 1000, noisy_rf[t_slice], color="red", alpha=0.7, label="Noisy RF (SNR=10dB)")
axes_cm[0].plot(t_rf[t_slice] * 1000, cleaned_rf[t_slice], color="green", alpha=0.8, label="Cleaned RF (BPF Countermeasure)")
axes_cm[0].plot(t_rf[t_slice] * 1000, modulated_rf[t_slice], color="black", linestyle="--", alpha=0.5, label="Clean RF Reference")
axes_cm[0].set_title(f"Part 2: Countermeasure Time Domain — RF SNR: {snr_before:.1f} dB -> {snr_after:.1f} dB (+{improvement_db:.1f} dB improvement)")
axes_cm[0].set_xlabel("Time (ms)")
axes_cm[0].set_ylabel("Amplitude")
axes_cm[0].legend(loc="upper right")
axes_cm[0].grid(True, alpha=0.5)

_, _, psd_noisy = measure_bandwidth(noisy_rf, fs=FS_RF, fc=CARRIER_FREQ)
_, _, psd_clean_rf = measure_bandwidth(cleaned_rf, fs=FS_RF, fc=CARRIER_FREQ)
axes_cm[1].plot(freqs_rf[mask_rf] / 1000.0, psd_noisy[mask_rf], color="red", alpha=0.5, label="Noisy RF Spectrum")
axes_cm[1].plot(freqs_rf[mask_rf] / 1000.0, psd_clean_rf[mask_rf], color="green", alpha=0.8, label="Filtered RF Spectrum")
axes_cm[1].axvline(CARRIER_FREQ / 1000.0, color="black", linestyle="--")
axes_cm[1].axvline((CARRIER_FREQ + MAX_BANDWIDTH) / 1000.0, color="gray", linestyle=":")
axes_cm[1].set_title("Part 2: RF Spectrum Before vs After Bandpass Countermeasure")
axes_cm[1].set_xlabel("Frequency (kHz)")
axes_cm[1].set_ylabel("PSD (dB)")
axes_cm[1].set_ylim(-60, 5)
axes_cm[1].legend(loc="upper right")
axes_cm[1].grid(True, alpha=0.5)

fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUTS_DIR_P2, "noise_countermeasure_comparison.png"), dpi=150)
print("Saved: outputs/part-2/noise_countermeasure_comparison.png")

# ------------------------------------------------------------
# STEP 13: DEMODULATION & AUDIO RECOVERY
# ------------------------------------------------------------
print("\n--- Step 13: Demodulation & Audio Recovery ---")
demod_rf = demodulate_ssb(cleaned_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=MAX_BANDWIDTH)
recovered_raw_60k = resample_poly(demod_rf, 1, FS_RF // TARGET_FS)[: len(vocal)]
recovered_60k = apply_wiener_denoise(recovered_raw_60k, fs=TARGET_FS, bw_signal=MAX_BANDWIDTH)

peak_ref = np.max(np.abs(decoded_vocal))
peak_rec = np.max(np.abs(recovered_60k))
if peak_rec > 0:
    recovered_60k = recovered_60k * (peak_ref / peak_rec)

demod_snr = evaluate_demodulation(decoded_vocal, recovered_60k)
print(f"Recovered Audio SNR (with countermeasure): {demod_snr:.2f} dB")

demod_noisy = demodulate_ssb(noisy_rf, fs=FS_RF, fc=CARRIER_FREQ, mode="usb", bw=30_000)
recovered_noisy_60k = resample_poly(demod_noisy, 1, FS_RF // TARGET_FS)[: len(vocal)]
if np.max(np.abs(recovered_noisy_60k)) > 0:
    recovered_noisy_60k = recovered_noisy_60k * (peak_ref / np.max(np.abs(recovered_noisy_60k)))
noisy_snr = evaluate_demodulation(decoded_vocal, recovered_noisy_60k)
print(f"Recovered Audio SNR (no countermeasure)  : {noisy_snr:.2f} dB")

# Save Recovered Audio Files
recovered_wav_path = os.path.join(OUTPUTS_DIR_P2, "recovered_vocal.wav")
noisy_wav_path = os.path.join(OUTPUTS_DIR_P2, "noisy_recovered_vocal.wav")
wavfile.write(recovered_wav_path, TARGET_FS, (recovered_60k * 32767).astype(np.int16))
wavfile.write(noisy_wav_path, TARGET_FS, (recovered_noisy_60k * 32767).astype(np.int16))
print(f"Saved: {recovered_wav_path}")
print(f"Saved: {noisy_wav_path}")

# Plot Recovered Waveform
fig_rec, ax_rec = plt.subplots(figsize=(11, 4))
plot_start = int(0.25 * TARGET_FS)
plot_n = min(12000, len(vocal) - plot_start)
t_audio = np.arange(plot_n) / TARGET_FS
sl = slice(plot_start, plot_start + plot_n)
ax_rec.plot(t_audio * 1000, decoded_vocal[sl], label="Original Vocal (Decoded from 64 kbps)", color="blue", alpha=0.7)
ax_rec.plot(t_audio * 1000, recovered_noisy_60k[sl], label=f"Noisy demod ({noisy_snr:.1f} dB)", color="red", alpha=0.4)
ax_rec.plot(t_audio * 1000, recovered_60k[sl], label=f"After countermeasure ({demod_snr:.1f} dB)", color="green", alpha=0.85)
ax_rec.set_title("Part 2: Demodulated vocal — noise influence vs countermeasure")
ax_rec.set_xlabel("Time (ms)")
ax_rec.set_ylabel("Amplitude")
ax_rec.legend(loc="upper right")
ax_rec.grid(True, alpha=0.5)
fig_rec.tight_layout()
fig_rec.savefig(os.path.join(OUTPUTS_DIR_P2, "recovered_vocal_waveform.png"), dpi=150)
print("Saved: outputs/part-2/recovered_vocal_waveform.png")


# ============================================================
# PART 3: MICROPHONE ARRAY & SOURCE LOCALISATION
# ============================================================
OUTPUTS_DIR_P3 = os.path.join("outputs", "part-3")
os.makedirs(OUTPUTS_DIR_P3, exist_ok=True)

print("\n" + "=" * 50)
print("PART 3: MICROPHONE ARRAY & SOURCE LOCALISATION")
print("=" * 50)

# ------------------------------------------------------------
# STEP 14: MICROPHONE ARRAY GEOMETRY & SOUND PROPAGATION
# ------------------------------------------------------------
print("\n--- Step 14: Circular Microphone Array Model ---")
num_mics = 8
mic_coords = create_circular_array(num_mics=num_mics, radius=MIC_RADIUS)
source_pos = generate_random_source(r_min=2.5, r_max=3.5, seed=RANDOM_SEED)
distances = calculate_distances(source_pos, mic_coords)

print(f"Array: {num_mics} microphones on circle of radius {MIC_RADIUS} m")
print(f"True Source Position: ({source_pos[0]:.3f}, {source_pos[1]:.3f}) m")
for m in range(num_mics):
    print(f"  Mic {m}: ({mic_coords[m,0]:.2f}, {mic_coords[m,1]:.2f}) -> distance = {distances[m]:.3f} m")

# Simulate multi-channel reception (attenuation + sub-sample propagation delays)
mic_signals, _, delays_applied = simulate_microphone_array(
    vocal, TARGET_FS, mic_coords, source_pos, c=SPEED_OF_SOUND, relative_delays=True
)

# Plot Received Waveforms
fig_wf, ax_wf = plt.subplots(figsize=(11, 5))
t_plot_ms = np.arange(600) / TARGET_FS * 1000.0
for m in range(num_mics):
    ax_wf.plot(
        t_plot_ms,
        mic_signals[m, :600],
        label=f"Mic {m} (d={distances[m]:.2f}m, τ={delays_applied[m]*1000:.1f}ms)",
        alpha=0.75,
        linewidth=1.0,
    )
ax_wf.set_title("Part 3: Received Signals at 8-Mic Array (Inverse-Square Attenuation & Propagation Delays)")
ax_wf.set_xlabel("Time (ms)")
ax_wf.set_ylabel("Amplitude")
ax_wf.legend(loc="upper right", ncol=2, fontsize=8)
ax_wf.grid(True, alpha=0.4)
fig_wf.tight_layout()
fig_wf.savefig(os.path.join(OUTPUTS_DIR_P3, "microphone_signals_waveform.png"), dpi=150)
print("Saved: outputs/part-3/microphone_signals_waveform.png")

# ------------------------------------------------------------
# STEP 15: TDOA ESTIMATION (GCC-PHAT) & SOURCE LOCALISATION
# ------------------------------------------------------------
print("\n--- Step 15: GCC-PHAT TDOA & Source Localisation ---")
measured_tdoas = compute_pairwise_tdoas(
    mic_signals, TARGET_FS, ref_mic=0, c=SPEED_OF_SOUND, mic_coords=mic_coords
)
estimated_pos, _ = localise_source_nls(
    mic_coords, measured_tdoas, c=SPEED_OF_SOUND, ref_mic=0
)
err_m, err_cm, err_pct = evaluate_localisation_accuracy(source_pos, estimated_pos)

print(f"True Source      : ({source_pos[0]:.4f}, {source_pos[1]:.4f}) m")
print(f"Estimated Source : ({estimated_pos[0]:.4f}, {estimated_pos[1]:.4f}) m")
print(f"Localisation Error: {err_cm:.3f} cm ({err_m*1000:.1f} mm)")

# Plot Array Geometry & Source Localisation
fig_geo, ax_geo = plt.subplots(figsize=(8, 8))
circ = plt.Circle((0, 0), MIC_RADIUS, color="gray", fill=False, linestyle="--", label=f"Mic Array Ring (R={MIC_RADIUS}m)")
ax_geo.add_patch(circ)
ax_geo.scatter(mic_coords[:, 0], mic_coords[:, 1], color="red", s=80, zorder=5, label="Microphones (M=8)")
for m in range(num_mics):
    ax_geo.annotate(f"M{m}", (mic_coords[m, 0] * 1.15, mic_coords[m, 1] * 1.15), fontsize=9, fontweight="bold", ha="center")

ax_geo.scatter(source_pos[0], source_pos[1], color="blue", marker="*", s=200, zorder=6, label=f"True Source ({source_pos[0]:.2f}, {source_pos[1]:.2f})")
ax_geo.scatter(estimated_pos[0], estimated_pos[1], color="lime", marker="x", s=150, linewidth=2.5, zorder=7, label=f"Estimated ({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f})\nError: {err_cm:.2f} cm")
for m in range(num_mics):
    ax_geo.plot([mic_coords[m, 0], source_pos[0]], [mic_coords[m, 1], source_pos[1]], color="gray", alpha=0.2, linestyle=":")

ax_geo.set_title(f"Part 3: 2D Microphone Array Geometry & Source Localisation\nError = {err_cm:.2f} cm ({err_m*1000:.1f} mm)")
ax_geo.set_xlabel("X Position (meters)")
ax_geo.set_ylabel("Y Position (meters)")
ax_geo.set_xlim(-4.5, 4.5)
ax_geo.set_ylim(-4.5, 4.5)
ax_geo.set_aspect("equal")
ax_geo.grid(True, alpha=0.4)
ax_geo.legend(loc="upper left")
fig_geo.tight_layout()
fig_geo.savefig(os.path.join(OUTPUTS_DIR_P3, "array_geometry_and_source.png"), dpi=150)
print("Saved: outputs/part-3/array_geometry_and_source.png")

# Plot GCC-PHAT Peaks
fig_gcc, axes_gcc = plt.subplots(4, 2, figsize=(12, 8))
axes_gcc = axes_gcc.ravel()
delays_theo = calculate_propagation_delays(distances, c=SPEED_OF_SOUND, relative_to_min=True)
for m in range(num_mics):
    _, cc, lags_sec = estimate_tdoa_gcc_phat(mic_signals[m], mic_signals[0], TARGET_FS)
    mask_lags = np.abs(lags_sec) <= 0.010
    axes_gcc[m].plot(lags_sec[mask_lags] * 1000, cc[mask_lags], color="steelblue", linewidth=1.1)
    axes_gcc[m].axvline(delays_theo[m] * 1000, color="red", linestyle="--", label=f"True: {delays_theo[m]*1000:.2f}ms")
    axes_gcc[m].axvline(measured_tdoas[m] * 1000, color="green", linestyle=":", label=f"GCC: {measured_tdoas[m]*1000:.2f}ms")
    axes_gcc[m].set_title(f"Mic {m} vs Mic 0 (TDOA = {measured_tdoas[m]*1000:.2f} ms)", fontsize=9)
    axes_gcc[m].set_xlabel("Lag (ms)", fontsize=8)
    axes_gcc[m].legend(fontsize=7)
    axes_gcc[m].grid(True, alpha=0.3)
fig_gcc.suptitle("Part 3: GCC-PHAT Cross-Correlation Functions (Sharp TDOA Peaks)", fontsize=11)
fig_gcc.tight_layout()
fig_gcc.savefig(os.path.join(OUTPUTS_DIR_P3, "tdoa_gcc_phat_peaks.png"), dpi=150)
print("Saved: outputs/part-3/tdoa_gcc_phat_peaks.png")


# ============================================================
# DISPLAY ALL FIGURES
# ============================================================
print("\n" + "=" * 50)
print("All Part 1, Part 2 & Part 3 steps complete.")
print("=" * 50)
plt.show()