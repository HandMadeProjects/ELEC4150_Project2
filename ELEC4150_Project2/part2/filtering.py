
import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_anti_aliasing_lpf(signal, fs, cutoff=4000, order=8):
    """
    Low-pass the baseband message to the Part 2 bandwidth limit (4 kHz)
    before RF upsampling / SSB modulation.
    """
    sig = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if cutoff >= nyq * 0.95:
        return sig.copy()

    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, sig)


def apply_rf_bandpass(noisy_rf, fs, fc=250_000, bw=4000, mode="usb", order=3):
    """
    RF Front-End Bandpass Filter (Countermeasure Stage 1).

    Rejects out-of-band AWGN noise before the demodulator.
    For USB SSB-AM, the message power resides strictly between
    fc and fc + bw (250 kHz to 254 kHz).
    
    Any wideband AWGN outside this 4 kHz passband is discarded,
    drastically reducing the noise power entering the demodulation stage.
    Uses zero-phase filtering (sosfiltfilt) to preserve carrier and sideband phase.

    Parameters
    ----------
    noisy_rf : numpy.ndarray
        Noisy RF signal at sampling rate fs.
    fs : int
        Sampling frequency in Hz (e.g. 600 kHz).
    fc : int
        Carrier frequency in Hz (250 kHz).
    bw : int
        Message bandwidth in Hz (4 kHz).
    mode : str
        'usb' or 'lsb'.
    order : int
        Butterworth filter order.

    Returns
    -------
    filtered_rf : numpy.ndarray
        Bandpass-filtered RF signal.
    """
    nyq = fs / 2.0
    guard = 800  # small transition guard band in Hz
    if mode.lower() == "usb":
        f_low = max(fc - guard, 100)
        f_high = min(fc + bw + guard, nyq * 0.99)
    else:
        f_low = max(fc - bw - guard, 100)
        f_high = min(fc + guard, nyq * 0.99)

    low_norm = f_low / nyq
    high_norm = f_high / nyq

    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    filtered_rf = sosfiltfilt(sos, noisy_rf)
    return filtered_rf


def apply_wiener_denoise(audio, fs, bw_signal=4000, n_fft=1024, hop_length=256):
    """
    Baseband Spectral Denoising / Wiener Filter (Countermeasure Stage 2).

    Applies frequency-domain Wiener filtering / spectral subtraction to reduce
    in-band AWGN residuals from recovered baseband audio.
    
    Since the message vocal is band-limited to bw_signal (4 kHz), the spectral
    region above 5 kHz contains purely AWGN channel noise. The noise power in
    this out-of-band region accurately determines the flat white-noise floor,
    enabling robust Wiener filter estimation without needing a priori silent frames.

    Parameters
    ----------
    audio : numpy.ndarray
        Recovered noisy baseband audio signal.
    fs : int
        Audio sampling rate in Hz (e.g. 60 000).
    bw_signal : int
        Maximum frequency of the vocal signal (default 4000 Hz).
    n_fft : int
        STFT analysis window length.
    hop_length : int
        STFT hop length.

    Returns
    -------
    denoised_audio : numpy.ndarray
        Denoised baseband audio.
    """
    from scipy.signal import stft, istft

    N = len(audio)
    if N < n_fft:
        return audio.copy()

    # Compute STFT
    noverlap = n_fft - hop_length
    freqs, times, Z = stft(audio, fs=fs, window="hann", nperseg=n_fft, noverlap=noverlap)

    P = np.abs(Z) ** 2

    # Per-bin noise: quiet-frame statistics (speech has pauses).
    # Also use out-of-band bins as a white-noise floor when they exist.
    noise_psd = np.percentile(P, 20, axis=1, keepdims=True)
    f_noise_min = min(bw_signal + 1500, fs / 2 * 0.5)
    f_noise_max = min(fs / 2 * 0.9, fs / 2 - 100)
    noise_mask = (freqs >= f_noise_min) & (freqs <= f_noise_max)
    if np.any(noise_mask):
        oob_floor = np.mean(P[noise_mask, :])
        noise_psd = np.maximum(noise_psd, oob_floor)

    # Spectral subtraction / Wiener gain with mild oversubtraction
    gain = np.maximum(1.0 - 1.5 * noise_psd / (P + 1e-12), 0.08)

    # Apply gain to STFT coefficients
    Z_clean = Z * gain

    # Inverse STFT to time domain
    _, denoised = istft(Z_clean, fs=fs, window="hann", nperseg=n_fft, noverlap=noverlap)

    # Match length
    if len(denoised) >= N:
        denoised = denoised[:N]
    else:
        denoised = np.pad(denoised, (0, N - len(denoised)))

    return denoised



def apply_countermeasure_pipeline(noisy_rf, clean_rf, fs_rf, fc=250_000, bw=4000, mode="usb"):
    """
    Full dual-stage countermeasure pipeline:
    1. RF Bandpass filter (pre-demodulation)
    2. Measures SNR improvement at RF

    Parameters
    ----------
    noisy_rf : numpy.ndarray
        RF signal corrupted with AWGN.
    clean_rf : numpy.ndarray
        Clean uncorrupted RF signal.
    fs_rf : int
        RF sampling rate in Hz.

    Returns
    -------
    cleaned_rf : numpy.ndarray
        RF signal after RF bandpass countermeasure.
    snr_before : float
        SNR before countermeasure (dB).
    snr_after : float
        SNR after countermeasure (dB).
    improvement_db : float
        SNR improvement (dB).
    """
    from part2.noise import calculate_snr

    snr_before = calculate_snr(clean_rf, noisy_rf)
    cleaned_rf = apply_rf_bandpass(noisy_rf, fs_rf, fc=fc, bw=bw, mode=mode)
    snr_after = calculate_snr(clean_rf, cleaned_rf)
    improvement_db = snr_after - snr_before

    return cleaned_rf, snr_before, snr_after, improvement_db
