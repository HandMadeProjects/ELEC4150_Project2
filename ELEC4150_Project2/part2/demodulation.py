
import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate


def demodulate_ssb(modulated, fs, fc=250_000, mode="usb", bw=4000):
    """
    Demodulate an SSB-AM signal back to baseband.

    Method — Product Detector
    -------------------------
    1. Multiply modulated signal by a local carrier replica:
       v(t) = s(t) × cos(2πfc·t)
    2. Low-pass filter with cut-off = message bandwidth (bw Hz)
       using zero-phase Butterworth filtering (sosfiltfilt) to preserve waveform phase.
    3. Normalise and return recovered audio.

    Parameters
    ----------
    modulated : numpy.ndarray
        Received (possibly noisy) SSB-modulated signal.
    fs : int
        Sampling rate in Hz.
    fc : int
        Carrier frequency in Hz (must match modulator).
    mode : str
        'usb' or 'lsb' — must match modulator.
    bw : float
        Low-pass cut-off frequency in Hz (= message bandwidth).

    Returns
    -------
    recovered : numpy.ndarray
        Demodulated baseband signal (float64, normalised to [-1, 1]).
    """

    N = len(modulated)
    t = np.arange(N) / fs

    # ---------------------------------------------------------
    # Product detector: multiply by local carrier
    # ---------------------------------------------------------

    carrier = np.cos(2 * np.pi * fc * t)
    v = modulated * carrier

    # ---------------------------------------------------------
    # Low-pass filter to remove 2fc components
    # Cut-off = message bandwidth
    # ---------------------------------------------------------

    nyq    = fs / 2.0
    cutoff = min(bw, nyq * 0.95)   # safety margin
    sos    = butter(6, cutoff / nyq, btype="low", output="sos")
    lpf_out = sosfiltfilt(sos, v)

    # ---------------------------------------------------------
    # Normalise
    # ---------------------------------------------------------

    peak = np.max(np.abs(lpf_out))
    if peak > 0:
        recovered = lpf_out / peak
    else:
        recovered = lpf_out

    return recovered


def evaluate_demodulation(original, recovered, max_lag_samples=2000):
    """
    Compute SNR between original message and recovered signal (dB),
    compensating for any filter or transmission group delay.

    Parameters
    ----------
    original : numpy.ndarray
        Original message signal before modulation.
    recovered : numpy.ndarray
        Demodulated signal (may differ in length or phase).
    max_lag_samples : int
        Maximum lag to search for alignment.

    Returns
    -------
    snr_db : float
        Signal-to-noise ratio in dB. Higher = better.
    """
    N   = min(len(original), len(recovered))
    sig = original[:N].astype(np.float64)
    rec = recovered[:N].astype(np.float64)

    # Scale recovered to match original power
    sig_rms = np.sqrt(np.mean(sig ** 2)) + 1e-10
    rec_rms = np.sqrt(np.mean(rec ** 2)) + 1e-10
    rec_scaled = rec * (sig_rms / rec_rms)

    # Align signals via cross-correlation over a limited window
    eval_len = min(N, 20000)
    corr = correlate(sig[:eval_len], rec_scaled[:eval_len], mode="full")
    lags = np.arange(-eval_len + 1, eval_len)
    valid_mask = np.abs(lags) <= max_lag_samples
    best_lag = lags[valid_mask][np.argmax(corr[valid_mask])]

    if best_lag > 0:
        sig_aligned = sig[best_lag:N]
        rec_aligned = rec_scaled[: N - best_lag]
    elif best_lag < 0:
        sig_aligned = sig[: N + best_lag]
        rec_aligned = rec_scaled[-best_lag:N]
    else:
        sig_aligned = sig
        rec_aligned = rec_scaled

    err_power = np.mean((sig_aligned - rec_aligned) ** 2)
    sig_power = np.mean(sig_aligned ** 2)

    if err_power < 1e-20:
        return 100.0

    return float(10.0 * np.log10(sig_power / err_power))

