
import numpy as np


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

    # ---------------------------------------------------------
    # Check that the input is stereo
    # ---------------------------------------------------------

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            "Vocal separation requires stereo audio."
        )

    # ---------------------------------------------------------
    # Separate left and right channels
    # ---------------------------------------------------------

    left = audio[:, 0]
    right = audio[:, 1]

    # ---------------------------------------------------------
    # Centre component
    #
    # Signals appearing similarly in both channels
    # are reinforced.
    # ---------------------------------------------------------

    vocal = (left + right) / 2.0

    # ---------------------------------------------------------
    # Difference component
    #
    # Signals common to both channels are suppressed.
    # ---------------------------------------------------------

    instrumental = (left - right) / 2.0

    # ---------------------------------------------------------
    # Normalise to prevent clipping
    # ---------------------------------------------------------

    vocal_peak = np.max(np.abs(vocal))

    if vocal_peak > 1:
        vocal = vocal / vocal_peak

    instrumental_peak = np.max(
        np.abs(instrumental)
    )

    if instrumental_peak > 1:
        instrumental = instrumental / instrumental_peak

    return vocal, instrumental

