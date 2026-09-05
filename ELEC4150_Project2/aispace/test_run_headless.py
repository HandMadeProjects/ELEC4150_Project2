"""
Headless test runner for Part 1 pipeline.
Saves all figures to outputs/ without opening any windows or playing audio.
Run with: conda run -n baikarms python aispace/test_run_headless.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves files, no pop-ups
import matplotlib.pyplot as plt

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import TARGET_FS, AUDIO_MAX_DURATION, AUDIO_FILE
from part1.audio_io import load_audio, validate_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
from part1.spectrogram import calculate_spectrogram

OUTPUTS_DIR = os.path.join(ROOT, "outputs", "part-1")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

results = []

def record(label, passed, detail=""):
    results.append((label, passed, detail))
    mark = PASS if passed else FAIL
    print(f"{mark}  {label}" + (f"  [{detail}]" if detail else ""))


print("=" * 60)
print("ELEC4150 Project 2 — Part 1 Headless Test Run")
print("=" * 60)


# ----------------------------------------------------------
# STEP 1: Load audio
# ----------------------------------------------------------
print("\n[STEP 1] Load Audio")
print(f"[AUDIO] {AUDIO_FILE}")
audio, fs = load_audio(AUDIO_FILE)
audio = skip_leading_silence(audio, fs)
record("Audio loaded",          audio is not None and len(audio) > 0)
record("Stereo format (N,2)",   audio.ndim == 2 and audio.shape[1] == 2)
record("Sample rate detected",  fs > 0,  f"{fs} Hz")


# ----------------------------------------------------------
# STEP 2: Validate
# ----------------------------------------------------------
print("\n[STEP 2] Validate Audio")
valid = validate_audio(audio, fs)
record("validate_audio() passes", valid is True)


# ----------------------------------------------------------
# STEP 3: Trim
# ----------------------------------------------------------
print("\n[STEP 3] Trim to 25 s")
max_samples = int(AUDIO_MAX_DURATION * fs)
audio = audio[:max_samples]
left, right = audio[:, 0], audio[:, 1]
duration = len(left) / fs
record("Trim applied", duration <= AUDIO_MAX_DURATION, f"{duration:.2f} s")


# ----------------------------------------------------------
# STEP 4: Resample to 60 kHz
# ----------------------------------------------------------
print(f"\n[STEP 4] Resample {fs} Hz → {TARGET_FS} Hz")
left_60k  = resample_audio(left,  fs, TARGET_FS)
right_60k = resample_audio(right, fs, TARGET_FS)
audio_60k = np.stack([left_60k, right_60k], axis=1)

expected = int(len(left) * TARGET_FS / fs)
record("60 kHz requirement met",  TARGET_FS == 60000)
record("Resampled length correct", abs(len(left_60k) - expected) <= 2,
       f"got {len(left_60k)}, expected ~{expected}")


# ----------------------------------------------------------
# STEP 5: Vocal / Instrumental Separation
# ----------------------------------------------------------
print("\n[STEP 5] Vocal Separation (centre + harmonic singing band)")
vocal, karaoke, attenuation_db = separate_vocal_and_karaoke(audio_60k, TARGET_FS)
karaoke_mono = np.mean(karaoke, axis=1)
record("Vocal extracted",         len(vocal) == len(left_60k))
record("Vocal peak ≤ 1.0",        np.max(np.abs(vocal)) <= 1.0 + 1e-9,
       f"peak={np.max(np.abs(vocal)):.4f}")


# ----------------------------------------------------------
# STEP 6: Karaoke extraction — bass/drums kept, vocals −40 dB
# ----------------------------------------------------------
print("\n[STEP 6] Karaoke Extraction (vocal deleted, tune kept)")
original_mono_60k = (left_60k + right_60k) / 2.0
record("Karaoke extracted",        len(karaoke_mono) == len(left_60k))
record("Karaoke peak ≤ 1.0",       np.max(np.abs(karaoke)) <= 1.0 + 1e-9)
record("≥ 40 dB vocal attenuation (RUBRIC)",
       attenuation_db >= 40,
       f"{attenuation_db:.1f} dB")


# ----------------------------------------------------------
# STEP 7: Spectrogram calculation
# ----------------------------------------------------------
print("\n[STEP 7] Spectrogram")
f_sg, t_sg, mag_db = calculate_spectrogram(vocal, TARGET_FS)
record("Spectrogram shape valid", mag_db.ndim == 2)
record("Freq bins correct",       f_sg.shape == (1025,))
record("Nyquist correct",         abs(f_sg[-1] - TARGET_FS/2) < 1)


# ----------------------------------------------------------
# PLOT 1: Vocal waveform
# ----------------------------------------------------------
print("\n[PLOTS] Generating figures...")
t_ax = np.arange(len(vocal)) / TARGET_FS

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_ax, vocal, linewidth=0.5, color="#e05c5c")
ax.set_title("Extracted Vocal Component — Time Domain  [60 kHz]")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_xlim(0, t_ax[-1]); ax.grid(True, alpha=0.4)
fig.tight_layout()
p = os.path.join(OUTPUTS_DIR, "vocal_waveform.png")
fig.savefig(p, dpi=150); plt.close(fig)
record("vocal_waveform.png saved", os.path.exists(p))


# ----------------------------------------------------------
# PLOT 2: Karaoke waveform
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.arange(len(karaoke_mono)) / TARGET_FS, karaoke_mono,
        linewidth=0.5, color="#5c9de0")
ax.set_title(
    f"Karaoke Track — {attenuation_db:.1f} dB vocal attenuation"
)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.4)
fig.tight_layout()
p = os.path.join(OUTPUTS_DIR, "karaoke_waveform.png")
fig.savefig(p, dpi=150); plt.close(fig)
record("karaoke_waveform.png saved", os.path.exists(p))


# ----------------------------------------------------------
# PLOT 3: Spectrum comparison
# ----------------------------------------------------------
N_sp = min(len(vocal), len(karaoke_mono))
freqs_plot = np.fft.rfftfreq(N_sp, d=1 / TARGET_FS)
o_db_plot = 20 * np.log10(np.abs(np.fft.rfft(original_mono_60k[:N_sp])) + 1e-12)
v_db_plot = 20 * np.log10(np.abs(np.fft.rfft(vocal[:N_sp]))             + 1e-12)
k_db_plot = 20 * np.log10(np.abs(np.fft.rfft(karaoke_mono[:N_sp]))      + 1e-12)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(freqs_plot, o_db_plot, label="Original (mono)",              alpha=0.7, linewidth=0.8)
ax.plot(freqs_plot, v_db_plot, label="Extracted Vocal",              alpha=0.9, linewidth=0.8)
ax.plot(freqs_plot, k_db_plot, label=f"Karaoke ({attenuation_db:.0f} dB atten.)",
        alpha=0.9, linewidth=0.8, color="steelblue")
ax.set_title("Spectrum Comparison — Original vs Vocal vs Karaoke")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
ax.set_xlim(0, 10000); ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
p = os.path.join(OUTPUTS_DIR, "spectrum_comparison.png")
fig.savefig(p, dpi=150); plt.close(fig)
record("spectrum_comparison.png saved", os.path.exists(p))


# ----------------------------------------------------------
# PLOT 4: Vocal spectrogram
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
img = ax.pcolormesh(t_sg, f_sg, mag_db, shading="auto",
                    cmap="magma", vmin=-80, vmax=0)
ax.set_title("Vocal Component — Spectrogram  [60 kHz]")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
ax.set_ylim(0, 10000)
fig.colorbar(img, ax=ax, label="Magnitude (dB)")
fig.tight_layout()
p = os.path.join(OUTPUTS_DIR, "vocal_spectrogram.png")
fig.savefig(p, dpi=150); plt.close(fig)
record("vocal_spectrogram.png saved", os.path.exists(p))


# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
for label, ok, detail in results:
    mark = "✓" if ok else "✗"
    d = f"  [{detail}]" if detail else ""
    print(f"  [{mark}] {label}{d}")
print(f"\n  {passed}/{total} checks passed")
if passed == total:
    print("  ALL PASS ✓")
else:
    print(f"  {total-passed} FAILED ✗")
print("=" * 60)
print(f"\nOutput figures saved to: {OUTPUTS_DIR}")
