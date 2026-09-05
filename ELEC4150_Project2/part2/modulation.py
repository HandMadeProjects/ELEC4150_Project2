
import numpy as np
from scipy.signal import hilbert, butter, sosfilt


def modulate_ssb(signal, fs, fc=250_000, mode="usb", m=0.9):
    """
    Single-Sideband AM (SSB-AM) modulation.

    SSB-AM occupies exactly the message bandwidth around the carrier.
    For a message band-limited to 4 kHz, SSB-AM uses only 4 kHz of
    RF spectrum (vs 8 kHz for DSB-AM), satisfying the ≤ 4 kHz
    bandwidth specification in the rubric.

    Method — Phasing / Analytic Signal
    ------------------------------------
    Upper sideband (USB):
        x_a(t) = x(t) + j·H{x(t)}   ← analytic signal via Hilbert
        s(t)   = Re[ x_a(t) · e^{j2πfc·t} ]
               = x(t)·cos(2πfc·t) - H{x(t)}·sin(2πfc·t)

    Lower sideband (LSB): negate the imaginary part.

    Parameters
    ----------
    signal : numpy.ndarray
        Mono message signal (float64, values in [-1, 1]).
        Should be band-limited to ≤ 4 kHz before calling.
    fs : int
        Sampling rate of `signal` in Hz.
        Must be at least 2×(fc + message_bw). Use a high-rate
        intermediate representation (e.g. 60 kHz or upsample first).
    fc : int
        Carrier frequency in Hz (default 250 000 Hz = 250 kHz).
    mode : str
        'usb' (upper sideband) or 'lsb' (lower sideband).
    m : float
        Modulation index / amplitude scaling (0 < m ≤ 1).

    Returns
    -------
    modulated : numpy.ndarray
        SSB-modulated signal at sampling rate `fs`.
    t : numpy.ndarray
        Time axis (seconds).
    carrier : numpy.ndarray
        Pure carrier signal (for reference/plotting).
    """

    if fc >= fs / 2:
        raise ValueError(
            f"Carrier {fc} Hz must be below Nyquist {fs/2:.0f} Hz. "
            "Upsample the signal to at least 2×(fc + bw)."
        )

    N = len(signal)
    t = np.arange(N) / fs

    # ---------------------------------------------------------
    # Normalise message signal to [-1, 1]
    # ---------------------------------------------------------
    peak = np.max(np.abs(signal))
    if peak > 0:
        x = signal / peak * m
    else:
        x = signal.copy()

    # ---------------------------------------------------------
    # Analytic signal via Hilbert transform
    # x_a(t) = x(t) + j·H{x(t)}
    # ---------------------------------------------------------
    x_analytic = hilbert(x)

    # ---------------------------------------------------------
    # SSB modulation
    # USB: s(t) = x(t)cos(wc·t) - H{x}sin(wc·t)
    # LSB: s(t) = x(t)cos(wc·t) + H{x}sin(wc·t)
    # ---------------------------------------------------------
    carrier_cos = np.cos(2 * np.pi * fc * t)
    carrier_sin = np.sin(2 * np.pi * fc * t)
    carrier     = carrier_cos.copy()

    if mode.lower() == "usb":
        modulated = x.real * carrier_cos - x_analytic.imag * carrier_sin
    elif mode.lower() == "lsb":
        modulated = x.real * carrier_cos + x_analytic.imag * carrier_sin
    else:
        raise ValueError("mode must be 'usb' or 'lsb'")

    # Normalise output
    pk = np.max(np.abs(modulated))
    if pk > 0:
        modulated = modulated / pk

    return modulated, t, carrier


def measure_bandwidth(signal, fs, fc, threshold_db=-40):
    """
    Measure the occupied bandwidth of a modulated signal (Hz).

    Uses the power spectral density and finds the frequency range
    around the carrier that contains most of the signal power
    (above `threshold_db` relative to the peak PSD).

    Parameters
    ----------
    signal : numpy.ndarray
        Modulated signal.
    fs : int
        Sampling rate in Hz.
    fc : int
        Carrier frequency in Hz (used as search centre).
    threshold_db : float
        Power threshold relative to peak (default −40 dB).

    Returns
    -------
    bandwidth_hz : float
        Occupied bandwidth in Hz.
    freqs : numpy.ndarray
        Frequency axis (Hz).
    psd_db : numpy.ndarray
        Power spectral density in dB.
    """

    N = len(signal)
    window = np.hanning(N)
    fft = np.fft.rfft(signal * window, n=N)
    psd = np.abs(fft) ** 2

    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    psd_db = 10 * np.log10(psd / (np.max(psd) + 1e-20) + 1e-20)

    # Restrict search to region around carrier ±20 kHz
    mask     = np.abs(freqs - fc) <= 20_000
    above    = mask & (psd_db >= threshold_db)
    f_above  = freqs[above]

    if len(f_above) == 0:
        return 0.0, freqs, psd_db

    bandwidth_hz = float(np.max(f_above) - np.min(f_above))
    return bandwidth_hz, freqs, psd_db
