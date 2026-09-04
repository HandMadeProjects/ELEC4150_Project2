import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(audio, fs, title="Audio Waveform"):
    if audio.ndim == 2:
        signal = np.mean(audio, axis=1)
    else:
        signal = audio

    time = np.arange(len(signal)) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    