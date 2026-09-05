import numpy as np


def fractional_delay(signal, delay_seconds, fs):
    """
    Apply an exact sub-sample fractional time delay using the Fourier Shift Theorem.

    Method
    ------
    By the Fourier Shift Theorem, a continuous time delay tau corresponds to
    a linear phase shift in the frequency domain:
        x(t - tau) <---> X(f) * exp(-j * 2 * pi * f * tau)

    Using the discrete real FFT (rfft) and inverse FFT (irfft), this implements
    ideal bandlimited sinc interpolation, providing continuous sub-sample
    delay resolution with zero dispersion distortion and unity passband gain.

    Parameters
    ----------
    signal : numpy.ndarray
        1D real input audio signal.
    delay_seconds : float
        Time delay in seconds (can be positive or fractional samples).
    fs : int or float
        Sampling frequency in Hz.

    Returns
    -------
    delayed_signal : numpy.ndarray
        Time-delayed signal of identical length.
    """
    sig = np.asarray(signal, dtype=np.float64)
    N = len(sig)

    # Compute one-sided real FFT
    fft_vals = np.fft.rfft(sig, n=N)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Linear phase shift factor: exp(-j * 2 * pi * f * delay)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_seconds)

    # Apply phase shift and invert back to time domain
    delayed_fft = fft_vals * phase_shift
    delayed = np.fft.irfft(delayed_fft, n=N)

    return delayed


def calculate_propagation_delays(distances, c=343.0, relative_to_min=True):
    """
    Calculate propagation delays from source to microphones given distances.

    Parameters
    ----------
    distances : numpy.ndarray
        1D array of distances for each microphone in meters.
    c : float
        Speed of sound propagation in m/s (default 343.0 m/s for dry air at 20°C).
    relative_to_min : bool
        If True, subtract the minimum delay so the first arriving mic has delay = 0.

    Returns
    -------
    delays_seconds : numpy.ndarray
        1D array of delays in seconds.
    """
    dists = np.asarray(distances, dtype=np.float64)
    delays = dists / c
    if relative_to_min:
        delays = delays - np.min(delays)
    return delays


def apply_propagation_delays(signals, distances, fs, c=343.0, relative=True):
    """
    Apply propagation delays to multi-channel microphone signals.

    Parameters
    ----------
    signals : numpy.ndarray
        Either a 1D signal of length N (replicated for M mics)
        or a 2D array of shape (M, N) (e.g. after distance attenuation).
    distances : numpy.ndarray
        1D array of length M containing distances to each microphone in meters.
    fs : int or float
        Sampling rate in Hz.
    c : float
        Speed of sound in m/s (343.0 m/s).
    relative : bool
        If True, use delays relative to the nearest microphone.

    Returns
    -------
    delayed_signals : numpy.ndarray
        2D array of shape (M, N) containing delayed microphone signals.
    delays : numpy.ndarray
        1D array of length M containing delays applied in seconds.
    """
    dists = np.asarray(distances, dtype=np.float64)
    num_mics = len(dists)

    delays = calculate_propagation_delays(dists, c=c, relative_to_min=relative)

    sig_arr = np.asarray(signals, dtype=np.float64)
    if sig_arr.ndim == 1:
        # Replicate 1D signal across all mics
        M_signals = np.tile(sig_arr, (num_mics, 1))
    else:
        M_signals = sig_arr

    N = M_signals.shape[1]
    delayed_signals = np.zeros((num_mics, N), dtype=np.float64)

    for m in range(num_mics):
        delayed_signals[m] = fractional_delay(M_signals[m], delays[m], fs)

    return delayed_signals, delays
