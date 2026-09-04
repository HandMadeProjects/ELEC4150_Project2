from scipy.signal import resample_poly


def resample_audio(audio, original_fs, target_fs):
    """
    Resample an audio signal from the original
    sampling rate to the target sampling rate.
    """

    audio_resampled = resample_poly(
        audio,
        target_fs,
        original_fs
    )

    return audio_resampled