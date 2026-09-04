import numpy as np
import soundfile as sf


def load_audio(file_path):
    audio, fs = sf.read(file_path)

    return audio, fs


def validate_audio(audio, fs):
    if audio is None or len(audio) == 0:
        raise ValueError("Audio file contains no samples.")

    if fs <= 0:
        raise ValueError("Invalid sampling rate.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    print("Audio validation successful.")
    print("Channels:", 1 if audio.ndim == 1 else audio.shape[1])
    print("Duration:", len(audio) / fs, "seconds")

    return True