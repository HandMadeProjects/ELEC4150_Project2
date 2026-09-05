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


def skip_leading_silence(audio, fs, threshold=0.01, pre_roll_s=0.05):
    """
    Drop leading digital silence so demos start on actual audio.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono (N,) or stereo (N, 2) samples.
    fs : int
        Sampling rate in Hz.
    threshold : float
        Amplitude treated as silence.
    pre_roll_s : float
        Seconds kept before the first non-silent sample.

    Returns
    -------
    numpy.ndarray
        Audio starting at the first non-silent region.
    """
    if audio.ndim == 2:
        envelope = np.max(np.abs(audio), axis=1)
    else:
        envelope = np.abs(audio)

    above = np.flatnonzero(envelope > threshold)
    if above.size == 0:
        return audio

    start = max(0, int(above[0] - pre_roll_s * fs))
    return audio[start:]


def save_audio(file_path, audio, fs):
    """
    Save audio signal to a standard WAV audio file.

    Parameters
    ----------
    file_path : str
        Target audio file path (e.g. 'outputs/part-1/extracted_vocal.wav').
    audio : numpy.ndarray
        Mono or stereo audio samples.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    str
        Path to saved audio file.
    """
    import os
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Normalize if values exceed [-1.0, 1.0] to prevent hard digital clipping
    data = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak

    sf.write(file_path, data, fs)
    return file_path