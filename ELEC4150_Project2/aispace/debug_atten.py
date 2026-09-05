import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")
os.chdir(r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")

from part1.audio_io import load_audio
from part1.resampling import resample_audio
from part1.vocal_separation import separate_karaoke_wiener, measure_vocal_attenuation_db
from scipy.signal import stft

AUDIO_FILE = "audio_2-edsheeran_perfect.wav"
N_FFT = 2048; HOP = 512; NOVERLAP = N_FFT - HOP
TARGET_FS = 60000

# --- Load & check stereo ---
audio, fs = load_audio(AUDIO_FILE)
print(f"Loaded: {AUDIO_FILE}")
print(f"  Sample rate : {fs} Hz")
print(f"  Duration    : {len(audio)/fs:.2f} s")
print(f"  Shape       : {audio.shape}")

L0 = audio[:,0]; R0 = audio[:,1]
corr = float(np.corrcoef(L0, R0)[0,1])
diff_power = float(np.mean((L0-R0)**2))
sum_power  = float(np.mean((L0+R0)**2))
side_ratio = diff_power / (sum_power + 1e-10)
print(f"\nStereo analysis:")
print(f"  L-R correlation         : {corr:.4f}")
print(f"  Side / Centre power     : {side_ratio:.4f}  ({10*np.log10(side_ratio+1e-10):.1f} dB)")
if corr > 0.99:
    print("  => Near-MONO  (ideal for centre-channel cancellation)")
elif corr > 0.90:
    print("  => Lightly stereo  (good for cancellation)")
elif corr > 0.80:
    print("  => Moderately stereo")
else:
    print("  => Strongly stereo  (harder to cancel)")

# --- Resample & karaoke ---
audio = audio[:int(30*fs)]
left_60k  = resample_audio(audio[:,0], fs, TARGET_FS)
right_60k = resample_audio(audio[:,1], fs, TARGET_FS)
audio_60k = np.stack([left_60k, right_60k], axis=1)

print(f"\nRunning karaoke extraction...")
_, karaoke = separate_karaoke_wiener(audio_60k, TARGET_FS)
overall = measure_vocal_attenuation_db(audio_60k, karaoke, TARGET_FS)
print(f"Overall STFT-domain attenuation: {overall:.1f} dB")
if overall >= 40:
    print("  => >= 40 dB PASS ✓")
else:
    print(f"  => FAIL — {overall:.1f} dB only")

# --- Per-frequency attenuation ---
f_ax, _, Lf = stft(audio_60k[:,0], fs=TARGET_FS, window='hann', nperseg=N_FFT, noverlap=NOVERLAP)
_,    _, Rf = stft(audio_60k[:,1], fs=TARGET_FS, window='hann', nperseg=N_FFT, noverlap=NOVERLAP)
V = (Lf+Rf)/2; S = (Lf-Rf)/2
V_pow = np.abs(V)**2; S_pow = np.abs(S)**2
eps = 1e-10
vr = V_pow/(S_pow+eps)
beta2_mask = (S_pow**2) / (V_pow**2 + S_pow**2 + eps)
hard_zero  = (vr > 8).astype(float)
mask = (1.0 - hard_zero) * beta2_mask

V_freq        = np.mean(V_pow, axis=1)
V_masked_freq = np.mean(np.abs(mask * V)**2, axis=1)
per_freq_atten = 10*np.log10(V_freq / (V_masked_freq + 1e-30))

vocal_band = (f_ax >= 200) & (f_ax <= 3500)
pct_40 = np.mean(per_freq_atten[vocal_band] >= 40) * 100
print(f"\nVocal band (200-3500 Hz):")
print(f"  Bins >= 40 dB     : {pct_40:.1f}%")
print(f"  Min attenuation   : {per_freq_atten[vocal_band].min():.1f} dB")
print(f"  Max attenuation   : {per_freq_atten[vocal_band].max():.1f} dB")
print(f"  Mean attenuation  : {per_freq_atten[vocal_band].mean():.1f} dB")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(f_ax/1000, per_freq_atten, linewidth=0.9, color="#e05c5c", label="Per-frequency attenuation")
ax.axhline(40, color="lime", linewidth=1.5, linestyle="--", label="40 dB rubric target")
ax.axhline(overall, color="gold", linewidth=1.5, linestyle="--",
           label=f"Overall STFT-domain: {overall:.1f} dB")
ax.fill_between(f_ax/1000, per_freq_atten, 40,
                where=(per_freq_atten >= 40),
                alpha=0.25, color="lime", label=f"Bins ≥40 dB ({pct_40:.0f}%)")
ax.axvspan(0.2, 3.5, alpha=0.07, color="yellow", label="Vocal band (200–3500 Hz)")
ax.set_xlim(0, 10); ax.set_ylim(-10, 80)
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Attenuation (dB)")
ax.set_title(
    f"Per-frequency Vocal Attenuation — Ed Sheeran: Perfect\n"
    f"Overall: {overall:.1f} dB  |  Vocal-band ≥40 dB: {pct_40:.0f}%  |  "
    f"L-R corr: {corr:.3f}"
)
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2\outputs\part-1\vocal_attenuation_audio2.png"
fig.savefig(out, dpi=150)
print(f"\nSaved plot: {out}")
