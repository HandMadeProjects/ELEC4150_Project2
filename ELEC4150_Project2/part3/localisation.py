import numpy as np
from scipy.optimize import least_squares


def estimate_tdoa_gcc_phat(sig1, sig2, fs, max_tau=None):
    """
    Estimate Time Difference of Arrival (TDOA) between two microphone signals
    using the Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    and sub-sample parabolic interpolation.

    Formulation
    -----------
    GCC-PHAT cross-spectrum:
        R_12(f) = (X_1(f) * X_2*(f)) / (|X_1(f) * X_2*(f)| + eps)

    Cross-correlation:
        r_12(tau) = IFFT{ R_12(f) }

    Sub-sample peak refinement:
        3-point parabolic interpolation around the maximum peak achieves
        continuous sub-sample timing accuracy down to fractions of a sample.

    Parameters
    ----------
    sig1 : numpy.ndarray
        Signal from microphone 1.
    sig2 : numpy.ndarray
        Signal from microphone 2.
    fs : int or float
        Sampling frequency in Hz.
    max_tau : float, optional
        Maximum plausible time delay in seconds (e.g. mic_distance / c).

    Returns
    -------
    tdoa_seconds : float
        Estimated delay tau = t1 - t2 in seconds.
    cc : numpy.ndarray
        Cross-correlation array over evaluated lag range.
    lags_seconds : numpy.ndarray
        Corresponding time lags in seconds.
    """
    n = len(sig1) + len(sig2)
    # Zero-pad to next power of 2 for fast FFT
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # Real FFTs
    X1 = np.fft.rfft(sig1, n=n_fft)
    X2 = np.fft.rfft(sig2, n=n_fft)

    # Cross-spectral density
    cross_spec = X1 * np.conj(X2)
    # PHAT-β (β = 0.5): milder than full PHAT so harmonic vocals
    # keep a sharp true-delay peak instead of collapsing to lag 0.
    mag = np.abs(cross_spec) + 1e-12
    norm_spec = cross_spec / np.power(mag, 0.5)

    # Inverse FFT to time domain
    cc = np.fft.irfft(norm_spec, n=n_fft)
    # Shift zero-lag to center
    cc = np.fft.fftshift(cc)
    lags = np.arange(-n_fft // 2, n_fft // 2)
    lags_sec = lags / float(fs)

    # Restrict to search window if max_tau is provided
    if max_tau is not None:
        mask = np.abs(lags_sec) <= (max_tau * 1.05 + 2.0 / fs)
        cc_search = np.where(mask, cc, -np.inf)
    else:
        cc_search = cc

    peak_idx = int(np.argmax(cc_search))
    lag_int = int(lags[peak_idx])

    # Integer alignment, then weighted phase-slope for the fractional sample.
    # Parabola-on-GCC underestimates delays < 1 sample on voiced/harmonic audio.
    sig1_a = np.roll(np.asarray(sig1, dtype=np.float64), -lag_int)
    sig2_a = np.asarray(sig2, dtype=np.float64)
    frac_sec = _fractional_delay_from_phase(sig1_a, sig2_a, fs)
    tdoa_seconds = float(lag_int) / float(fs) + frac_sec

    if max_tau is not None:
        tdoa_seconds = float(np.clip(tdoa_seconds, -max_tau, max_tau))

    return tdoa_seconds, cc, lags_sec


def _fractional_delay_from_phase(sig1, sig2, fs, fmin=200.0, fmax=8000.0):
    """Weighted linear fit of unwrapped cross-spectrum phase → delay (seconds)."""
    n_fft = 2 ** int(np.ceil(np.log2(len(sig1) + len(sig2))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    x1 = np.fft.rfft(sig1, n=n_fft)
    x2 = np.fft.rfft(sig2, n=n_fft)
    cross = x1 * np.conj(x2)

    f_hi = min(fmax, 0.4 * fs)
    band = (freqs >= fmin) & (freqs <= f_hi) & (np.abs(cross) > 1e-12)
    if np.count_nonzero(band) < 16:
        return 0.0

    mag = np.abs(cross[band])
    phase = np.unwrap(np.angle(cross[band]))
    f = freqs[band]
    # phase ≈ -2 π f τ   (weighted least squares)
    w = mag
    a = (-2.0 * np.pi * f) * w
    b = phase * w
    denom = float(np.dot(a, a))
    if denom < 1e-18:
        return 0.0
    tau = float(np.dot(a, b) / denom)
    # Residual after integer alignment must be well under ±1 sample
    return float(np.clip(tau, -1.5 / fs, 1.5 / fs))


def compute_pairwise_tdoas(mic_signals, fs, ref_mic=0, c=343.0, mic_coords=None):
    """
    Compute TDOAs for all microphones relative to a reference microphone.

    Parameters
    ----------
    mic_signals : numpy.ndarray
        Array of shape (M, N) with signals for each of M microphones.
    fs : int or float
        Sampling frequency in Hz.
    ref_mic : int
        Index of the reference microphone (default 0).
    c : float
        Speed of sound in m/s.
    mic_coords : numpy.ndarray, optional
        Microphone coordinates (M, 2) to determine maximum plausible delay.

    Returns
    -------
    tdoas : numpy.ndarray
        1D array of length M containing TDOA (tau_m - tau_ref) in seconds.
    """
    num_mics = len(mic_signals)
    tdoas = np.zeros(num_mics, dtype=np.float64)

    ref_sig = mic_signals[ref_mic]

    for m in range(num_mics):
        if m == ref_mic:
            tdoas[m] = 0.0
            continue

        max_tau = None
        if mic_coords is not None:
            max_dist = np.linalg.norm(mic_coords[m] - mic_coords[ref_mic])
            max_tau = max_dist / c + 1e-4

        tdoas[m], _, _ = estimate_tdoa_gcc_phat(
            mic_signals[m], ref_sig, fs, max_tau=max_tau
        )

    return tdoas


def localise_source_nls(mic_coords, tdoas, c=343.0, ref_mic=0, initial_guess=None):
    """
    Localise the 2D position of the sound source using Non-linear Least Squares (NLS)
    multilateration based on measured TDOAs.

    Physics & Geometry
    ------------------
    For microphone m and reference microphone ref:
        Delta d_m = ||p_s - p_m|| - ||p_s - p_ref||
        Measured Delta d_m = c * tdoa_m

    Residual vector:
        r_m(x, y) = (||(x, y) - p_m|| - ||(x, y) - p_ref||) - c * tdoa_m

    Solves: min_{(x,y)} sum_m r_m(x, y)^2 using robust Levenberg-Marquardt algorithm.

    Parameters
    ----------
    mic_coords : numpy.ndarray
        Array of shape (M, 2) containing microphone coordinates.
    tdoas : numpy.ndarray
        Array of length M containing relative TDOAs in seconds.
    c : float
        Speed of sound in m/s (default 343.0 m/s).
    ref_mic : int
        Index of reference microphone.
    initial_guess : tuple of float, optional
        Starting coordinate for optimization (default (0, 0)).

    Returns
    -------
    estimated_pos : numpy.ndarray
        Estimated (x_s, y_s) coordinates in meters.
    res : OptimizeResult
        Scipy optimization result details.
    """
    mics = np.asarray(mic_coords, dtype=np.float64)
    num_mics = len(mics)
    p_ref = mics[ref_mic]

    # Target distance differences
    delta_d_measured = c * np.asarray(tdoas, dtype=np.float64)

    def residuals(p):
        d_ref = np.sqrt((p[0] - p_ref[0]) ** 2 + (p[1] - p_ref[1]) ** 2)
        d_m = np.sqrt((p[0] - mics[:, 0]) ** 2 + (p[1] - mics[:, 1]) ** 2)
        delta_d_pred = d_m - d_ref
        return delta_d_pred - delta_d_measured

    if initial_guess is None:
        # Coarse grid search for best starting point
        grid_vals = np.linspace(-6, 6, 25)
        best_err = np.inf
        best_p0 = np.array([1.0, 1.0])
        for gx in grid_vals:
            for gy in grid_vals:
                gp = np.array([gx, gy])
                err = np.sum(residuals(gp) ** 2)
                if err < best_err:
                    best_err = err
                    best_p0 = gp
        p0 = best_p0
    else:
        p0 = np.asarray(initial_guess, dtype=np.float64)

    res = least_squares(residuals, p0, method="lm")
    estimated_pos = res.x

    return estimated_pos, res


def steered_response_power_map(
    mic_signals, fs, mic_coords, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0), grid_res=0.1, c=343.0
):
    """
    Compute 2D Steered Response Power (SRP-PHAT) spatial acoustic energy heatmap.

    Parameters
    ----------
    mic_signals : numpy.ndarray
        Multi-channel signals (M, N).
    fs : int or float
        Sampling frequency in Hz.
    mic_coords : numpy.ndarray
        Array of shape (M, 2).
    x_range : tuple of float
        (min_x, max_x) in meters.
    y_range : tuple of float
        (min_y, max_y) in meters.
    grid_res : float
        Spatial resolution in meters.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    X_grid : numpy.ndarray
        2D meshgrid X coordinates.
    Y_grid : numpy.ndarray
        2D meshgrid Y coordinates.
    srp_map : numpy.ndarray
        2D array of acoustic power at each grid point.
    best_pos : numpy.ndarray
        (x, y) coordinate with highest acoustic energy.
    """
    num_mics = len(mic_signals)
    x_vals = np.arange(x_range[0], x_range[1] + grid_res / 2, grid_res)
    y_vals = np.arange(y_range[0], y_range[1] + grid_res / 2, grid_res)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Precompute pairwise GCC-PHAT cross-correlations
    pairs = []
    gcc_data = []
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            pairs.append((i, j))
            _, cc, lags_sec = estimate_tdoa_gcc_phat(mic_signals[i], mic_signals[j], fs)
            gcc_data.append((cc, lags_sec[0], lags_sec[1] - lags_sec[0]))

    srp_map = np.zeros(X_grid.shape, dtype=np.float64)
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

    # Compute distances to all mics for all grid points: shape (num_points, M)
    mics = np.asarray(mic_coords, dtype=np.float64)
    diff = grid_points[:, np.newaxis, :] - mics[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (num_points, M)

    accum_power = np.zeros(len(grid_points), dtype=np.float64)

    for idx, (i, j) in enumerate(pairs):
        cc, lag_min, lag_step = gcc_data[idx]
        tau_ij = (dists[:, i] - dists[:, j]) / c
        # Map delay to CC index
        cc_indices = np.clip(
            np.round((tau_ij - lag_min) / lag_step).astype(int), 0, len(cc) - 1
        )
        accum_power += cc[cc_indices]

    srp_map = accum_power.reshape(X_grid.shape)

    # Find peak
    max_idx = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    best_pos = np.array([X_grid[max_idx], Y_grid[max_idx]], dtype=np.float64)

    return X_grid, Y_grid, srp_map, best_pos


def evaluate_localisation_accuracy(true_pos, estimated_pos):
    """
    Evaluate localisation accuracy by calculating Euclidean error distance.

    Parameters
    ----------
    true_pos : array_like of shape (2,)
        True sound source coordinates (x, y) in meters.
    estimated_pos : array_like of shape (2,)
        Estimated sound source coordinates (x, y) in meters.

    Returns
    -------
    error_meters : float
        Euclidean error distance in meters.
    error_cm : float
        Euclidean error distance in centimeters.
    error_percent : float
        Error relative to true distance from origin (%).
    """
    t_pos = np.asarray(true_pos, dtype=np.float64)
    e_pos = np.asarray(estimated_pos, dtype=np.float64)

    error_meters = float(np.linalg.norm(t_pos - e_pos))
    error_cm = float(error_meters * 100.0)

    true_dist = np.linalg.norm(t_pos)
    error_percent = float((error_meters / max(true_dist, 1e-6)) * 100.0)

    return error_meters, error_cm, error_percent
