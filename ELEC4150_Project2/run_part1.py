"""
ELEC4150/8150 Project 2 — Part 1 Live Runner
============================================
Implements and demonstrates Part 1 aims:
  - Part 1.1: Convert audio to 60 kHz, save full_song_60k.wav, and play audio + live moving DJ spectrogram
  - Part 1.2: Extract center-panned vocals only from song file, save extracted_vocal.wav & waveforms
  - Part 1.3: Karaoke track: delete vocals and keep tune only, save karaoke_instrumental.wav & attenuation

Usage:
  python run_part1.py                   # Plays 60 kHz full song + live DJ spectrogram (Part 1.1 default)
  python run_part1.py --track vocal     # Plays extracted vocal + live DJ spectrogram (Part 1.2)
  python run_part1.py --track karaoke   # Plays karaoke tune + live DJ spectrogram (Part 1.3)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from config import TARGET_FS, AUDIO_MAX_DURATION, AUDIO_FILE
from part1.audio_io import load_audio, validate_audio, save_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
from part1.realtime_spectrogram import play_with_realtime_spectrogram

OUTPUTS_DIR = os.path.join("outputs", "part-1")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run ELEC4150 Project 2 Part 1")
parser.add_argument(
    "--audio",
    default=AUDIO_FILE,
    help=f"Audio file to process (default: {AUDIO_FILE})",
)
parser.add_argument(
    "--track",
    choices=["full", "vocal", "karaoke"],
    default="full",
    help="Which track to play in the live moving DJ spectrogram (default: full [Part 1.1])",
)
parser.add_argument(
    "--duration",
    type=float,
    default=20.0,
    help="Duration in seconds for live audio playback & spectrogram (default: 20s)",
)
args, _ = parser.parse_known_args()

print("=" * 65)
print("ELEC4150 PROJECT 2 — PART 1: AUDIO PROCESSING PIPELINE")
print(f"Audio Source File : {args.audio}")
print(f"Selected Playback : {args.track.upper()} TRACK (Part 1.{'1' if args.track == 'full' else '2' if args.track == 'vocal' else '3'})")
print("=" * 65)

# ------------------------------------------------------------
# PART 1.1: AUDIO LOADING & RESAMPLING TO 60 kHz
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[PART 1.1] Convert Audio to 60 kHz & Full Song Spectrogram")
print("-" * 65)

print(f">> Loading and validating audio ({args.audio})...")
audio, orig_fs = load_audio(args.audio)
audio = skip_leading_silence(audio, orig_fs)
validate_audio(audio, orig_fs)

# Trim to maximum project duration
max_samples = int(min(AUDIO_MAX_DURATION, len(audio) / orig_fs) * orig_fs)
audio = audio[:max_samples]

print(f">> Resampling stereo audio from {orig_fs} Hz to {TARGET_FS} Hz...")
left_60k = resample_audio(audio[:, 0], orig_fs, TARGET_FS)
right_60k = resample_audio(audio[:, 1], orig_fs, TARGET_FS)
audio_60k = np.stack([left_60k, right_60k], axis=1)

# Save 60 kHz full song
full_song_path = os.path.join(OUTPUTS_DIR, "full_song_60k.wav")
save_audio(full_song_path, audio_60k, TARGET_FS)
print(f">> Saved 60 kHz full song: {full_song_path}")

# ------------------------------------------------------------
# PART 1.2: ONLY EXTRACT VOCALS FROM SONG FILE
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[PART 1.2] Only Extract Vocals from Song File")
print("-" * 65)

print(">> Isolating centre-panned harmonic vocals (singing band, drums/bass removed)...")
vocal, karaoke, atten_db = separate_vocal_and_karaoke(audio_60k, TARGET_FS)
karaoke_mono = np.mean(karaoke, axis=1)

# Save extracted vocal
vocal_path = os.path.join(OUTPUTS_DIR, "extracted_vocal.wav")
save_audio(vocal_path, vocal, TARGET_FS)
print(f">> Saved extracted vocal: {vocal_path}")

# ------------------------------------------------------------
# PART 1.3: KARAOKE TRACK (DELETE VOCALS & KEEP TUNE ONLY)
# ------------------------------------------------------------
print("\n" + "-" * 65)
print("[PART 1.3] Karaoke: Delete Vocals & Keep Tune Only")
print("-" * 65)

print(">> Removing vocals, keeping bass, drums and stereo instruments...")
karaoke_path = os.path.join(OUTPUTS_DIR, "karaoke_instrumental.wav")
save_audio(karaoke_path, karaoke, TARGET_FS)
print(f">> Saved karaoke instrumental: {karaoke_path}")
print(f">> Vocal attenuation achieved: {atten_db:.1f} dB")
if atten_db >= 40:
    print(">> >= 40 dB requirement: PASS")
else:
    print(f">> >= 40 dB requirement: FAIL  (got {atten_db:.1f} dB)")

# ------------------------------------------------------------
# STATIC ANALYSIS PLOTS: WAVEFORMS & SPECTRUM COMPARISON
# ------------------------------------------------------------
print("\n>> Generating summary waveform & frequency spectrum plots...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
t_sec = np.arange(len(vocal)) / TARGET_FS

# Decimate time signals for clean, crisp rendering
plot_step = max(1, len(vocal) // 10000)
mono_orig = (left_60k + right_60k) / 2.0

# 1. Full song / vocal waveform
axes[0].plot(t_sec[::plot_step], mono_orig[::plot_step], color="#718096", alpha=0.5, label="60 kHz Song (Part 1.1)")
axes[0].plot(t_sec[::plot_step], vocal[::plot_step], color="#3182ce", linewidth=0.8, label="Extracted Vocal (Part 1.2)")
axes[0].set_title("Part 1.1 & 1.2: Waveforms (Full Song vs Extracted Vocal)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.4)

# 2. Karaoke track waveform
axes[1].plot(t_sec[::plot_step], karaoke_mono[::plot_step], color="#319795", linewidth=0.8)
axes[1].set_title(f"Part 1.3: Karaoke Track Waveform — Vocals Deleted ({atten_db:.1f} dB Attenuation)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.4)

# 3. Frequency spectrum comparison up to 10 kHz
n_spec = min(len(mono_orig), len(vocal), len(karaoke_mono))
freqs = np.fft.rfftfreq(n_spec, 1 / TARGET_FS)
f_mask = freqs <= 10000
freqs_sub = freqs[f_mask]
f_step = max(1, len(freqs_sub) // 2000)

orig_db = 20 * np.log10(np.abs(np.fft.rfft(mono_orig[:n_spec]))[f_mask] + 1e-12)
vocal_db = 20 * np.log10(np.abs(np.fft.rfft(vocal[:n_spec]))[f_mask] + 1e-12)
karaoke_db = 20 * np.log10(np.abs(np.fft.rfft(karaoke_mono[:n_spec]))[f_mask] + 1e-12)

axes[2].plot(freqs_sub[::f_step], orig_db[::f_step], label="Part 1.1: Full Song Mono", color="#718096", alpha=0.6)
axes[2].plot(freqs_sub[::f_step], vocal_db[::f_step], label="Part 1.2: Extracted Vocal", color="#3182ce", linewidth=1.2)
axes[2].plot(freqs_sub[::f_step], karaoke_db[::f_step], label="Part 1.3: Karaoke Tune", color="#319795", linewidth=1.2)
axes[2].set_title("Frequency Spectrum Comparison (0–10 kHz)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude (dB)")
axes[2].set_xlim(0, 10000)
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.4)

fig.tight_layout()
summary_plot_path = os.path.join(OUTPUTS_DIR, "part1_summary_plots.png")
fig.savefig(summary_plot_path, dpi=150)
plt.close(fig)
print(f">> Saved summary analysis plot: {summary_plot_path}")

# ------------------------------------------------------------
# LIVE PLAYBACK WITH MOVING DJ SPECTROGRAM
# ------------------------------------------------------------
# Select audio track for real-time player
if args.track == "vocal":
    play_audio = vocal
    track_name = "Part 1.2: Extracted Vocal"
elif args.track == "karaoke":
    play_audio = karaoke_mono
    track_name = "Part 1.3: Karaoke Tune (Vocals Deleted)"
else:
    play_audio = mono_orig
    track_name = "Part 1.1: 60 kHz Full Song"

print("\n" + "=" * 65)
print(f"LAUNCHING REAL-TIME AUDIO PLAYBACK + MOVING DJ SPECTROGRAM")
print(f"Active Track : {track_name}")
print("=" * 65)

play_duration = min(args.duration, len(play_audio) / float(TARGET_FS))
play_with_realtime_spectrogram(play_audio, TARGET_FS, duration=play_duration)

print("\nPart 1 complete.")
