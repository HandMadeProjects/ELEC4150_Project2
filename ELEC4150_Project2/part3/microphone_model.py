import numpy as np


def create_circular_array(num_mics=8, radius=1.0):
    """
    Create coordinates for a uniform circular microphone array in the 2D plane.

    Parameters
    ----------
    num_mics : int
        Number of microphones (default 8, matching spec figure).
    radius : float
        Radius of the circular array in meters (default 1.0 m from config).

    Returns
    -------
    mic_coords : numpy.ndarray
        Array of shape (num_mics, 2) where each row is (x_m, y_m) in meters.
    """
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))


def generate_random_source(r_min=2.0, r_max=5.0, seed=None):
    """
    Generate a random sound source position in polar/Cartesian coordinates.

    Parameters
    ----------
    r_min : float
        Minimum distance from origin in meters.
    r_max : float
        Maximum distance from origin in meters.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    source_pos : numpy.ndarray
        Array of shape (2,) representing (x_s, y_s) in meters.
    """
    if seed is not None:
        np.random.seed(seed)

    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(0, 2 * np.pi)
    x_s = r * np.cos(theta)
    y_s = r * np.sin(theta)
    return np.array([x_s, y_s], dtype=np.float64)


def calculate_distances(source_pos, mic_coords):
    """
    Calculate Euclidean distances from the sound source to each microphone.

    Parameters
    ----------
    source_pos : array_like of shape (2,)
        Source coordinates (x_s, y_s).
    mic_coords : array_like of shape (M, 2)
        Microphone coordinates where each row is (x_m, y_m).

    Returns
    -------
    distances : numpy.ndarray
        1D array of length M containing distances in meters.
    """
    src = np.asarray(source_pos, dtype=np.float64)
    mics = np.asarray(mic_coords, dtype=np.float64)
    diff = mics - src
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances


def apply_inverse_square_attenuation(signal, distances, d0=1.0):
    """
    Apply inverse-square power law attenuation to the source audio signal.

    Specification:
        "Assuming the power of the audio signal reduces with the square
        of the distance from the source (i.e. following an inverse-square law)..."

    Physics:
        Power: P(d) = P(d0) * (d0 / d)^2
        Amplitude: A(d) = A(d0) * (d0 / d)

    Parameters
    ----------
    signal : numpy.ndarray
        1D clean audio signal (float64).
    distances : numpy.ndarray
        1D array of distances for each microphone in meters.
    d0 : float
        Reference distance in meters at which signal amplitude is unattenuated (1.0 m).

    Returns
    -------
    attenuated_signals : numpy.ndarray
        2D array of shape (num_mics, len(signal)) with distance-attenuated signals.
    """
    sig = np.asarray(signal, dtype=np.float64)
    dists = np.asarray(distances, dtype=np.float64)

    # Amplitude scaling factor: alpha_m = d0 / d_m
    scale_factors = d0 / np.maximum(dists, 1e-6)

    # Shape: (M, N)
    attenuated = scale_factors[:, np.newaxis] * sig[np.newaxis, :]
    return attenuated


def simulate_microphone_array(
    audio, fs, mic_coords, source_pos, c=343.0, d0=1.0, relative_delays=True
):
    """
    Simulate the complete acoustic reception of a sound source at all microphones,
    combining inverse-square distance attenuation and sub-sample propagation delays.

    Parameters
    ----------
    audio : numpy.ndarray
        Clean mono audio signal from Part 1.
    fs : int or float
        Audio sampling rate in Hz.
    mic_coords : numpy.ndarray
        Array of shape (M, 2) containing microphone (x, y) coordinates in meters.
    source_pos : numpy.ndarray or sequence
        Sound source (x_s, y_s) coordinates in meters.
    c : float
        Speed of sound propagation in m/s (default 343.0 m/s).
    d0 : float
        Reference distance in meters (default 1.0 m).
    relative_delays : bool
        If True, delays are measured relative to the nearest microphone (t0 = 0).

    Returns
    -------
    mic_signals : numpy.ndarray
        Array of shape (M, N) containing received audio signals at each microphone.
    distances : numpy.ndarray
        Array of length M with distance from source to each microphone.
    delays : numpy.ndarray
        Array of length M with propagation time delays in seconds.
    """
    from part3.delay import apply_propagation_delays

    # Step 1: Calculate distances
    distances = calculate_distances(source_pos, mic_coords)

    # Step 2: Apply inverse-square law attenuation
    attenuated = apply_inverse_square_attenuation(audio, distances, d0=d0)

    # Step 3: Apply sub-sample propagation delays
    mic_signals, delays = apply_propagation_delays(
        attenuated, distances, fs, c=c, relative=relative_delays
    )

    return mic_signals, distances, delays

