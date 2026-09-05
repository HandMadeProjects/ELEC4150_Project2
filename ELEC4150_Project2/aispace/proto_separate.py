"""Quick quality check of the production separator on real songs."""
import os
import sys
import time
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from part1.audio_io import load_audio
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke

TARGET_FS = 60000
OUT = os.path.join("outputs", "part-1", "proto")
os.makedirs(OUT, exist_ok=True)


def band_ratio_db(sig, ref, fs, f_lo, f_hi):
    def e(x):
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
        m = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.mean(np.abs(spec[m]) ** 2))
    return 10 * np.log10((e(sig) + 1e-20) / (e(ref) + 1e-20))


def rms(x):
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return float(np.sqrt(np.mean(x ** 2)))


def load60(path, seconds=10.0, start=6.0):
    audio, fs = load_audio(path)
    a0 = int(start * fs)
    a1 = int((start + seconds) * fs)
    audio = audio[a0:a1]
    left = resample_audio(audio[:, 0], fs, TARGET_FS)
    right = resample_audio(audio[:, 1], fs, TARGET_FS)
    return np.stack([left, right], axis=1)


def eval_one(tag, path):
    print(f"\n======== {tag} ========")
    audio = load60(path)
    orig = (audio[:, 0] + audio[:, 1]) / 2.0
    t0 = time.time()
    vocal, karaoke, atten = separate_vocal_and_karaoke(audio, TARGET_FS)
    kmono = np.mean(karaoke, axis=1)
    print(f"  time={time.time()-t0:.1f}s  vocal-bin atten={atten:.1f} dB")
    print(f"  RMS orig={rms(orig):.4f} vocal={rms(vocal):.4f} karaoke={rms(kmono):.4f}")
    for lo, hi, lab in [(0, 150, "bass"), (200, 4000, "voice"), (8000, 16000, "air")]:
        print(
            f"  {lab:5s} vocal {band_ratio_db(vocal, orig, TARGET_FS, lo, hi):6.1f} dB   "
            f"karaoke {band_ratio_db(kmono, orig, TARGET_FS, lo, hi):6.1f} dB"
        )
    sf.write(os.path.join(OUT, f"{tag}_vocal.wav"), vocal, TARGET_FS)
    sf.write(os.path.join(OUT, f"{tag}_karaoke.wav"), karaoke, TARGET_FS)


if __name__ == "__main__":
    eval_one("gangnam", "audio_3-GANGNAM STYLE.wav")
    eval_one("perfect", "audio_2-edsheeran_perfect.wav")
    eval_one("audio1", "audio_1.wav")
