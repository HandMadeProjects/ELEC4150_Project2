import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def calculate_spectrogram(audio, fs, **kwargs):
    window = kwargs.get("window", kwargs.get("window_type", "hann"))
    nperseg = kwargs.get("nperseg", kwargs.get("n_fft", 2048))
    hop_length = kwargs.get("hop_length", None)
    if hop_length is not None:
        noverlap = nperseg - hop_length
    else:
        noverlap = kwargs.get("noverlap", 1536)

    frequencies, times, Zxx = stft(
        audio,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )

    magnitude_db = 20 * np.log10(
        np.abs(Zxx) + 1e-10
    )

    return frequencies, times, magnitude_db