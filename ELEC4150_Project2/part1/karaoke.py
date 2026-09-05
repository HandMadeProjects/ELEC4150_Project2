"""Part 1.3 — karaoke / instrumental extraction."""

from part1.vocal_separation import separate_vocal_and_karaoke


def extract_karaoke(audio, fs):
    """
    Delete centre-panned vocals and keep the backing tune.

    Parameters
    ----------
    audio : numpy.ndarray
        Stereo mix, shape (N, 2).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    karaoke : numpy.ndarray, shape (N, 2)
        Stereo instrumental track.
    attenuation_db : float
        Vocal-bin attenuation in dB (rubric target ≥ 40 dB).
    """
    _, karaoke, attenuation_db = separate_vocal_and_karaoke(audio, fs)
    return karaoke, attenuation_db
