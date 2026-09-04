import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# STEP 1: LOAD AUDIO
# ============================================================

filename = "audio_1.wav"

audio, original_fs = sf.read(filename)

print("Original sampling rate:", original_fs)
print("Original samples:", len(audio))


# ============================================================
# STEP 2: CHECK AUDIO FORMAT
# ============================================================

if audio.ndim != 2 or audio.shape[1] != 2:
    raise ValueError(
        "Vocal extraction requires a stereo WAV file."
    )

left = audio[:, 0]
right = audio[:, 1]

print("Stereo audio detected.")
print("Left channel samples:", len(left))
print("Right channel samples:", len(right))


# ============================================================
# STEP 3: TRIM AUDIO TO 25 SECONDS
# ============================================================

duration = 25

max_samples = int(duration * original_fs)

left = left[:max_samples]
right = right[:max_samples]

print("Trimmed duration:", len(left) / original_fs, "seconds")


# ============================================================
# STEP 4: RESAMPLE TO 60 kHz
# ============================================================

target_fs = 60000

left_60k = resample_poly(
    left,
    target_fs,
    original_fs
)

right_60k = resample_poly(
    right,
    target_fs,
    original_fs
)

print("\nSampling Rate Conversion")
print("------------------------")
print(f"Before: {original_fs} Hz")
print(f"After : {target_fs} Hz")

if target_fs == 60000:
    print("60 kHz requirement: PASS")
else:
    print("60 kHz requirement: FAIL")


# ============================================================
# STEP 5: VOCAL EXTRACTION
# ============================================================

print("\nExtracting vocal component...")

# Centre-channel extraction
#
# Vocals that are approximately centred in a stereo recording
# are reinforced by adding the left and right channels.

vocal = (left_60k + right_60k) / 2


# ============================================================
# STEP 6: NORMALISE VOCAL SIGNAL
# ============================================================

max_value = np.max(np.abs(vocal))

if max_value > 0:
    vocal = vocal / max_value * 0.95


print("Vocal extraction completed.")


# ============================================================
# STEP 7: CREATE TIME AXIS
# ============================================================

vocal_time = np.arange(len(vocal)) / target_fs


# ============================================================
# STEP 8: PLOT EXTRACTED VOCAL SIGNAL
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    vocal_time,
    vocal,
    linewidth=0.6
)

plt.title("Extracted Vocal Component - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.xlim(0, len(vocal) / target_fs)

plt.tight_layout()


# ============================================================
# STEP 9: PLOT VOCAL SPECTROGRAM
# ============================================================

plt.figure(figsize=(12, 6))

plt.specgram(
    vocal,
    NFFT=2048,
    Fs=target_fs,
    noverlap=1024
)

plt.title("Extracted Vocal Component - Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.colorbar(label="Power (dB)")

plt.ylim(0, 10000)

plt.tight_layout()


# ============================================================
# STEP 10: CALCULATE ORIGINAL AND VOCAL SPECTRUM
# ============================================================

# Create mono version of original stereo audio
original_mono = (left_60k + right_60k) / 2

N = len(original_mono)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / target_fs
)

original_fft = np.fft.rfft(original_mono)
vocal_fft = np.fft.rfft(vocal)

original_magnitude = np.abs(original_fft)
vocal_magnitude = np.abs(vocal_fft)

# Convert to dB
original_db = 20 * np.log10(
    original_magnitude + 1e-12
)

vocal_db = 20 * np.log10(
    vocal_magnitude + 1e-12
)


# ============================================================
# STEP 11: PLOT SPECTRUM COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frequencies,
    original_db,
    label="Original"
)

plt.plot(
    frequencies,
    vocal_db,
    label="Extracted Vocal"
)

plt.title("Spectrum Before and After Vocal Extraction")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, 10000)

plt.grid(True)
plt.legend()

plt.tight_layout()


# ============================================================
# STEP 12: PLAY EXTRACTED VOCAL
# ============================================================

print("\nPlaying extracted vocal...")

sd.play(
    vocal,
    target_fs
)

sd.wait()

print("Vocal playback finished.")


# ============================================================
# STEP 13: KEEP FIGURES OPEN
# ============================================================

plt.show()