
import numpy as np


def add_awgn(signal, snr_db, seed=None):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal to achieve a target SNR.

    The rubric specifies adding AWGN at SNR = 10 dB.

    Parameters
    ----------
    signal : numpy.ndarray
        Clean input signal (1D array, real or complex).
    snr_db : float
        Desired Signal-to-Noise Ratio in decibels (dB).
        For project rubric: snr_db = 10.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_signal : numpy.ndarray
        Signal with AWGN added.
    noise : numpy.ndarray
        The generated noise array.
    actual_snr_db : float
        Measured SNR of the generated noisy signal in dB.
    """
    if seed is not None:
        np.random.seed(seed)

    sig = np.asarray(signal, dtype=np.float64)
    sig_power = np.mean(sig ** 2)

    if sig_power <= 0:
        raise ValueError("Signal power is zero; cannot calculate noise level.")

    # Calculate required noise power: P_noise = P_sig / (10^(SNR/10))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = np.sqrt(noise_power)

    # Generate white Gaussian noise
    noise = np.random.normal(0.0, noise_std, size=sig.shape)
    noisy_signal = sig + noise

    actual_snr = calculate_snr(sig, noisy_signal)

    return noisy_signal, noise, actual_snr


def calculate_snr(clean_signal, test_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in dB between a clean reference
    and a test (noisy or processed) signal.

    Formula:
        SNR_dB = 10 * log10( P_signal / P_noise )
        where P_noise = mean( (clean - test)^2 )

    Parameters
    ----------
    clean_signal : numpy.ndarray
        Reference clean signal.
    test_signal : numpy.ndarray
        Signal to evaluate against the clean reference.

    Returns
    -------
    snr_db : float
        SNR in dB.
    """
    n = min(len(clean_signal), len(test_signal))
    c = np.asarray(clean_signal[:n], dtype=np.float64)
    t = np.asarray(test_signal[:n], dtype=np.float64)

    sig_power = np.mean(c ** 2)
    noise_diff = c - t
    noise_power = np.mean(noise_diff ** 2)

    if noise_power <= 1e-20:
        return 100.0  # Essentially infinite SNR
    if sig_power <= 1e-20:
        return -100.0

    return float(10.0 * np.log10(sig_power / noise_power))
