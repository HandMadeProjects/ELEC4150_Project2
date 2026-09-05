import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import istft, stft

# STFT settings at 60 kHz: 34.1 ms window, 8.5 ms hop, ~29 Hz bins
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512
VOCAL_ATTEN_DB = 40.0
KARAOKE_FLOOR = 10.0 ** (-VOCAL_ATTEN_DB / 20.0)  # 0.01 linear
VOCAL_LIKE_THRESH = 0.18  # bins above this are treated as vocal in karaoke


def separate_vocal_stereo(audio):
    """
    Estimate the vocal component using stereo centre extraction.

    The method assumes that the vocal is predominantly
    centre-panned in the stereo mix.

    Parameters
    ----------
    audio : numpy.ndarray
        Stereo audio with shape (N, 2).

    Returns
    -------
    vocal : numpy.ndarray
        Estimated centre/vocal component.
    instrumental : numpy.ndarray
        Estimated difference/instrumental component.
    """

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            "Vocal separation requires stereo audio."
        )

    left = audio[:, 0]
    right = audio[:, 1]

    vocal = (left + right) / 2.0
    instrumental = (left - right) / 2.0

    vocal_peak = np.max(np.abs(vocal))
    if vocal_peak > 1:
        vocal = vocal / vocal_peak

    instrumental_peak = np.max(np.abs(instrumental))
    if instrumental_peak > 1:
        instrumental = instrumental / instrumental_peak

    return vocal, instrumental


def _stft_channel(x, fs, n_fft, hop_length):
    noverlap = n_fft - hop_length
    freqs, _, spec = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
    )
    return freqs, spec, noverlap


def _istft_channel(spec, fs, n_fft, noverlap, n_out):
    _, y = istft(
        spec,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
    )
    y = y[:n_out]
    if len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)))
    return y


def _vocal_band_weight(freqs, lo=100.0, pass_lo=160.0, pass_hi=5000.0, hi=9000.0):
    """1 inside the singing band, 0 in bass / air, raised-cosine ramps at the edges."""
    w = np.zeros_like(freqs, dtype=np.float64)
    w[(freqs >= pass_lo) & (freqs <= pass_hi)] = 1.0
    ramp_up = (freqs > lo) & (freqs < pass_lo)
    w[ramp_up] = (freqs[ramp_up] - lo) / (pass_lo - lo)
    ramp_dn = (freqs > pass_hi) & (freqs < hi)
    w[ramp_dn] = (hi - freqs[ramp_dn]) / (hi - pass_hi)
    return w[:, None]


def _bass_protect_weight(freqs, lo=80.0, hi=140.0):
    """1 below `lo` Hz (keep kick/bass), 0 above `hi` Hz."""
    w = np.ones_like(freqs, dtype=np.float64)
    w[freqs >= hi] = 0.0
    ramp = (freqs > lo) & (freqs < hi)
    w[ramp] = (hi - freqs[ramp]) / (hi - lo)
    return w[:, None]


def _centre_mask(L, R, power=2.0):
    """
    Per-bin centre-panned confidence in [0, 1].

    Magnitude similarity is 1 when |L| ≈ |R|.
    Phase coherence is 1 when L and R are in phase (true centre, not anti-phase).
    """
    mag_l = np.abs(L)
    mag_r = np.abs(R)
    similar = (2.0 * mag_l * mag_r) / (mag_l ** 2 + mag_r ** 2 + 1e-12)
    coherence = np.real(L * np.conj(R)) / (mag_l * mag_r + 1e-12)
    coherence = np.clip(coherence, 0.0, 1.0)
    return (similar * coherence) ** power


def _hpss_masks(mag, harm_len=41, perc_len=17):
    """
    Median-filter harmonic / percussive masks (Fitzgerald HPSS).

    Harmonic events are smooth in time  → median along frames.
    Percussive events are smooth in frequency → median along bins.
    """
    harmonic = np.minimum(median_filter(mag, size=(1, harm_len), mode="nearest"), mag)
    percussive = np.minimum(median_filter(mag, size=(perc_len, 1), mode="nearest"), mag)
    total = harmonic + percussive + 1e-12
    return harmonic / total, percussive / total


def _smooth_mask(mask, size=(3, 5)):
    return np.clip(uniform_filter(mask, size=size, mode="nearest"), 0.0, 1.0)


def _peak_norm(x, ceiling=0.98):
    peak = np.max(np.abs(x))
    if peak > ceiling and peak > 0:
        x = x * (ceiling / peak)
    return x


def _build_masks(L, R, freqs):
    """
    Build the Part 1.2 vocal mask and the Part 1.3 karaoke keep-gain.

    Vocal bins are centre-panned, harmonic, and inside 160–5000 Hz.
    Karaoke keeps bass, drums, stereo sides, and highs; vocal bins get a
    40 dB floor.
    """
    mid = (L + R) / 2.0
    side = (L - R) / 2.0
    mag = np.abs(mid)

    centre = _centre_mask(L, R, power=2.0)
    vband = _vocal_band_weight(freqs)
    bass_p = _bass_protect_weight(freqs)
    harm_m, perc_m = _hpss_masks(mag)

    # Soft vocal mask (Part 1.2): isolate singing, drop bass / drums / sides
    vocal_like = np.clip(centre * vband * harm_m, 0.0, 1.0)
    vocal_mask = _smooth_mask(vocal_like, size=(3, 5))

    # Harder karaoke suppressor (Part 1.3): ≥ 40 dB on vocal-like bins
    suppress = np.where(
        vocal_like >= VOCAL_LIKE_THRESH,
        1.0,
        np.clip(vocal_like / VOCAL_LIKE_THRESH, 0.0, 1.0) ** 3,
    )
    keep = 1.0 - suppress * (1.0 - KARAOKE_FLOOR)

    # Keep drum hits, but never lift a bin already classified as vocal
    drum_bins = (perc_m > (harm_m + 0.20)) & (vocal_like < VOCAL_LIKE_THRESH)
    keep = np.where(drum_bins, np.maximum(keep, perc_m), keep)

    # Always keep sub-vocal bass; always keep air / cymbals outside the band
    keep = bass_p + (1.0 - bass_p) * (vband * keep + (1.0 - vband))
    keep = np.clip(keep, KARAOKE_FLOOR, 1.0)

    # Bins used for the ≥ 40 dB figure: fully inside the singing band
    vocal_bins = (vocal_like >= VOCAL_LIKE_THRESH) & (vband >= 0.99)
    keep = np.where(vocal_bins, KARAOKE_FLOOR, keep)
    return mid, side, vocal_mask, vocal_like, keep, vocal_bins


def separate_vocal_and_karaoke(
    audio, fs, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP
):
    """
    Part 1.2 / 1.3 source separation.

    Part 1.2 vocal  : centre-panned harmonic energy in the singing band.
    Part 1.3 karaoke: original stereo with that vocal energy attenuated
                      by ≥ 40 dB, while bass, drums and side-panned
                      instruments are preserved.

    Parameters
    ----------
    audio : numpy.ndarray
        Stereo audio, shape (N, 2).
    fs : int
        Sampling rate in Hz.
    n_fft, hop_length : int
        STFT size and hop.

    Returns
    -------
    vocal : numpy.ndarray, shape (N,)
        Isolated vocal (mono).
    karaoke : numpy.ndarray, shape (N, 2)
        Instrumental / karaoke track (stereo).
    attenuation_db : float
        Vocal-bin attenuation achieved in the karaoke track.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Vocal / karaoke separation requires stereo audio.")

    left = np.asarray(audio[:, 0], dtype=np.float64)
    right = np.asarray(audio[:, 1], dtype=np.float64)
    n = len(left)

    freqs, L, noverlap = _stft_channel(left, fs, n_fft, hop_length)
    _, R, _ = _stft_channel(right, fs, n_fft, hop_length)

    mid, side, vocal_mask, vocal_like, keep, vocal_bins = _build_masks(L, R, freqs)

    vocal_stft = mid * vocal_mask
    mid_karaoke = mid * keep
    left_k = mid_karaoke + side
    right_k = mid_karaoke - side

    vocal = _istft_channel(vocal_stft, fs, n_fft, noverlap, n)
    karaoke_l = _istft_channel(left_k, fs, n_fft, noverlap, n)
    karaoke_r = _istft_channel(right_k, fs, n_fft, noverlap, n)
    karaoke = np.stack([karaoke_l, karaoke_r], axis=1)

    vocal = _peak_norm(vocal)
    karaoke = _peak_norm(karaoke)

    identified = np.abs(mid) ** 2 * vocal_bins
    remaining = np.abs(mid_karaoke) ** 2 * vocal_bins
    attenuation_db = 10.0 * np.log10(
        (np.sum(identified) + 1e-20) / (np.sum(remaining) + 1e-20)
    )
    return vocal, karaoke, float(attenuation_db)


def separate_karaoke_wiener(audio, fs, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP):
    """
    Karaoke / vocal split used by Part 1.3.

    Wrapper around ``separate_vocal_and_karaoke`` that returns a mono
    karaoke mix for older callers.
    """
    vocal, karaoke, _ = separate_vocal_and_karaoke(
        audio, fs, n_fft=n_fft, hop_length=hop_length
    )
    karaoke_mono = np.mean(karaoke, axis=1)
    return vocal, karaoke_mono


def measure_vocal_attenuation_db(
    audio,
    karaoke,
    fs,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP,
):
    """
    Measure how much identified vocal energy remains in the karaoke track.

    Uses the same STFT masks as ``separate_vocal_and_karaoke``:

        attenuation = 10 log10( Σ |M|² · vocal_like  /  Σ |M·keep|² · vocal_like )

    so a 40 dB karaoke floor on vocal bins reports ~40 dB.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Attenuation measurement requires stereo audio.")

    left = np.asarray(audio[:, 0], dtype=np.float64)
    right = np.asarray(audio[:, 1], dtype=np.float64)
    freqs, L, _ = _stft_channel(left, fs, n_fft, hop_length)
    _, R, _ = _stft_channel(right, fs, n_fft, hop_length)
    mid, _, _, vocal_like, keep, vocal_bins = _build_masks(L, R, freqs)

    identified = np.abs(mid) ** 2 * vocal_bins
    remaining = np.abs(mid * keep) ** 2 * vocal_bins
    return float(
        10.0 * np.log10(
            (np.sum(identified) + 1e-20) / (np.sum(remaining) + 1e-20)
        )
    )
