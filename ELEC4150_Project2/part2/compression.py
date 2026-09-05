
import numpy as np


def compress_vocal(vocal, fs_in, target_bitrate=64000):
    """
    Compress the vocal signal to the target bit-rate using
    μ-law PCM quantisation at 8 kHz.

    Pipeline
    --------
    1. Downsample vocal from fs_in → 8000 Hz
       (Nyquist covers vocal band 0–4 kHz; 8 kHz is the telephony
       standard and satisfies 64 kbps with 8-bit μ-law encoding.)
    2. Apply μ-law companding (ITU-T G.711) and quantise to 8 bits.
    3. Pack quantised samples into a byte string → compressed stream.
    4. Decode: unpack → μ-law expand → upsample back to fs_in.

    Bit-rate achieved
    -----------------
    8 bits/sample × 8000 samples/s = **64 000 bits/s = 64 kbps**

    Parameters
    ----------
    vocal : numpy.ndarray
        Mono vocal signal (float64, values in [-1, 1]).
    fs_in : int
        Input sampling rate in Hz (e.g. 60 000).
    target_bitrate : int
        Target bit-rate in bits/s (default 64 000 = 64 kbps).

    Returns
    -------
    compressed_bytes : bytes
        Raw byte stream of μ-law encoded 8-bit samples.
    decoded : numpy.ndarray
        Decoded vocal at the original sampling rate fs_in (float64).
    fs_out : int
        Sampling rate of the decoded signal (= fs_in).
    actual_bitrate : float
        Actual bit-rate achieved in bits/s.
    """

    # ---------------------------------------------------------
    # 1. Determine telephony sample-rate from target bit-rate
    #    bit_rate = bits_per_sample × sample_rate
    #    → sample_rate = bit_rate / 8 (for 8-bit quantisation)
    # ---------------------------------------------------------

    BITS_PER_SAMPLE = 8
    fs_codec = target_bitrate // BITS_PER_SAMPLE   # 8000 Hz for 64 kbps

    # ---------------------------------------------------------
    # 2. Downsample  fs_in → fs_codec
    # ---------------------------------------------------------

    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(int(fs_codec), int(fs_in))
    up   = fs_codec // g
    down = fs_in    // g

    vocal_ds = resample_poly(vocal.astype(np.float64), up, down)

    # ---------------------------------------------------------
    # 3. μ-law companding (ITU-T G.711)
    #
    #    Normalise → μ-law compress → quantise to 8 bits → pack
    # ---------------------------------------------------------

    MU = 255   # standard μ-law parameter

    # Normalise to [-1, 1]
    peak = np.max(np.abs(vocal_ds))
    if peak > 0:
        vocal_norm = vocal_ds / peak
    else:
        vocal_norm = vocal_ds.copy()

    # μ-law compression: y = sign(x) × ln(1 + μ|x|) / ln(1 + μ)
    vocal_ulaw = (
        np.sign(vocal_norm)
        * np.log1p(MU * np.abs(vocal_norm))
        / np.log1p(MU)
    )

    # Quantise to [0, 255]  (unsigned 8-bit)
    quantised = np.round(
        (vocal_ulaw + 1.0) / 2.0 * (2 ** BITS_PER_SAMPLE - 1)
    ).clip(0, 255).astype(np.uint8)

    # Pack into bytes
    compressed_bytes = quantised.tobytes()

    # ---------------------------------------------------------
    # 4. Decode: unpack → μ-law expand → upsample → fs_in
    # ---------------------------------------------------------

    decoded_uint8 = np.frombuffer(compressed_bytes, dtype=np.uint8)

    # Reverse quantisation
    decoded_ulaw  = decoded_uint8.astype(np.float64) / (
        2 ** BITS_PER_SAMPLE - 1
    ) * 2.0 - 1.0

    # μ-law expansion: x = sign(y) × [(1+μ)^|y| - 1] / μ
    decoded_norm = (
        np.sign(decoded_ulaw)
        * ((1 + MU) ** np.abs(decoded_ulaw) - 1)
        / MU
    )

    # Restore original amplitude
    decoded_norm = decoded_norm * peak

    # Upsample back to fs_in
    g2   = gcd(int(fs_in), int(fs_codec))
    up2  = fs_in    // g2
    dn2  = fs_codec // g2

    decoded = resample_poly(decoded_norm, up2, dn2)

    # Trim / pad to original length
    N = len(vocal)
    if len(decoded) >= N:
        decoded = decoded[:N]
    else:
        decoded = np.pad(decoded, (0, N - len(decoded)))

    # ---------------------------------------------------------
    # 5. Compute actual bit-rate
    # ---------------------------------------------------------

    duration_s    = len(quantised) / fs_codec
    actual_bitrate = (len(compressed_bytes) * 8) / duration_s

    return compressed_bytes, decoded, fs_in, actual_bitrate


def measure_bitrate(compressed_bytes, duration_s):
    """
    Compute the actual bit-rate of a compressed byte stream.

    Parameters
    ----------
    compressed_bytes : bytes
        Compressed audio byte stream.
    duration_s : float
        Duration of the original audio in seconds.

    Returns
    -------
    bitrate : float
        Bit-rate in bits/s.
    """
    return (len(compressed_bytes) * 8) / duration_s


def compression_snr_db(original, decoded):
    """
    Signal-to-noise ratio introduced by compression (dB).

    Parameters
    ----------
    original : numpy.ndarray
        Original signal before compression.
    decoded : numpy.ndarray
        Decoded signal after compression + decompression.

    Returns
    -------
    snr_db : float
        SNR in dB. Higher = better quality.
    """
    N   = min(len(original), len(decoded))
    sig = original[:N]
    err = sig - decoded[:N]

    sig_power = np.mean(sig ** 2)
    err_power = np.mean(err ** 2)

    if err_power < 1e-20:
        return float("inf")

    return 10.0 * np.log10(sig_power / err_power)
