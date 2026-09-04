import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def calculate_spectrogram(audio, fs):

    frequencies, times, Zxx = stft(
        audio,
        fs=fs,
        window="hann",
        nperseg=2048,
        noverlap=1536
    )

    magnitude_db = 20 * np.log10(
        np.abs(Zxx) + 1e-10
    )

    return frequencies, times, magnitude_db