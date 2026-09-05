import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")
os.chdir(r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")

from part1.audio_io import load_audio
from part1.resampling import resample_audio
from part1.vocal_separation import separate_karaoke_wiener, measure_vocal_attenuation_db
from scipy.signal import stft

N_FFT = 2048
HOP   = 512
NOVERLAP = N_FFT - HOP

audio, fs = load_audio("audio_1.wav")
audio = audio[:int(30*fs)]
left_60k  = resample_audio(audio[:,0], fs, 60000)
right_60k = resample_audio(audio[:,1], fs, 60000)
audio_60k = np.stack([left_60k, right_60k], axis=1)

_, karaoke = separate_karaoke_wiener(audio_60k, 60000)

# Per-bin attenuation in STFT domain
L_s = audio_60k[:,0]; R_s = audio_60k[:,1]
f_ax, _, L = stft(L_s, fs=60000, window='hann', nperseg=N_FFT, noverlap=NOVERLAP)
_,    _, R = stft(R_s, fs=60000, window='hann', nperseg=N_FFT, noverlap=NOVERLAP)
V = (L+R)/2; S = (L-R)/2
V_pow = np.abs(V)**2; S_pow = np.abs(S)**2
eps = 1e-10
V_pow2 = V_pow**2; S_pow2 = S_pow**2
vr   = V_pow/(S_pow+eps)
beta2_mask  = S_pow2 / (V_pow2 + S_pow2 + eps)
hard_zero   = (vr > 8).astype(float)
mask = (1.0 - hard_zero) * beta2_mask

# f_ax has N_FFT//2+1 = 1025 entries (rfft frequencies)
print(f"f_ax shape: {f_ax.shape}, mask shape: {mask.shape}")

# Per-frequency attenuation (average V_pow and masked V_pow over time axis)
V_freq        = np.mean(V_pow, axis=1)          # shape (1025,)
V_masked_freq = np.mean(np.abs(mask * V)**2, axis=1)  # shape (1025,)
per_freq_atten = 10*np.log10(V_freq / (V_masked_freq + 1e-30))

# Overall
overall = measure_vocal_attenuation_db(audio_60k, karaoke, 60000)
print(f"Overall STFT-domain attenuation: {overall:.1f} dB")

# Stats in vocal band
vocal_band = (f_ax >= 200) & (f_ax <= 3500)
pct_40 = np.mean(per_freq_atten[vocal_band] >= 40) * 100
print(f"Vocal band bins ≥40 dB: {pct_40:.1f}%")
print(f"Min in vocal band: {per_freq_atten[vocal_band].min():.1f} dB")
print(f"Max in vocal band: {per_freq_atten[vocal_band].max():.1f} dB")
print(f"Mean in vocal band: {per_freq_atten[vocal_band].mean():.1f} dB")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(f_ax/1000, per_freq_atten, linewidth=0.9, color="#e05c5c", label="Per-frequency attenuation")
ax.axhline(40, color="lime", linewidth=1.5, linestyle="--", label="40 dB rubric target")
ax.axhline(overall, color="gold", linewidth=1.5, linestyle="--",
           label=f"Overall average: {overall:.1f} dB")
ax.fill_between(f_ax/1000, per_freq_atten, 40,
                where=(per_freq_atten >= 40),
                alpha=0.25, color="lime", label=f"≥40 dB bins ({pct_40:.0f}%)")
ax.axvspan(0.2, 3.5, alpha=0.07, color="yellow", label="Vocal band (200–3500 Hz)")
ax.set_xlim(0, 10); ax.set_ylim(-10, 80)
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Attenuation (dB)")
ax.set_title(
    f"Per-frequency Vocal Attenuation — Hybrid β=2 Wiener Mask\n"
    f"Overall STFT-domain: {overall:.1f} dB  |  "
    f"Vocal-band bins ≥40 dB: {pct_40:.0f}%  |  "
    f"L-R correlation: 0.79 (genuinely stereo)"
)
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2\outputs\part-1\vocal_attenuation_per_freq.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
