import pytest
import numpy as np

from part3.microphone_model import (
    create_circular_array,
    generate_random_source,
    calculate_distances,
    apply_inverse_square_attenuation,
    simulate_microphone_array,
)
from part3.delay import (
    fractional_delay,
    calculate_propagation_delays,
    apply_propagation_delays,
)
from part3.localisation import (
    estimate_tdoa_gcc_phat,
    compute_pairwise_tdoas,
    localise_source_nls,
    evaluate_localisation_accuracy,
)


def test_circular_array_geometry():
    """Verify circular array has correct number of mics, radius, and symmetry."""
    num_mics = 8
    radius = 1.0
    mics = create_circular_array(num_mics=num_mics, radius=radius)

    assert mics.shape == (num_mics, 2)
    # Check all mics are on circle of radius 1.0
    radii = np.sqrt(np.sum(mics ** 2, axis=1))
    np.testing.assert_allclose(radii, radius, atol=1e-10)
    # Center of mass should be at origin
    center = np.mean(mics, axis=0)
    np.testing.assert_allclose(center, [0.0, 0.0], atol=1e-10)


def test_inverse_square_power_attenuation():
    """Verify signal power decreases with square of distance (amplitude with 1/d)."""
    fs = 60000
    t = np.arange(1000) / fs
    sig = np.cos(2 * np.pi * 500 * t)

    dists = np.array([1.0, 2.0, 4.0])
    attenuated = apply_inverse_square_attenuation(sig, dists, d0=1.0)

    p0 = np.mean(attenuated[0] ** 2)
    p1 = np.mean(attenuated[1] ** 2)
    p2 = np.mean(attenuated[2] ** 2)

    # Power at 2m should be 1/4 of power at 1m
    assert abs(p1 / p0 - 0.25) < 1e-4
    # Power at 4m should be 1/16 of power at 1m
    assert abs(p2 / p0 - 0.0625) < 1e-4


def test_subsample_fractional_delay():
    """Verify Fourier phase shift implements exact sub-sample fractional delay."""
    fs = 60000
    np.random.seed(123)
    # Realistic broadband audio signal
    sig = np.random.randn(8000)

    target_delay_samples = 2.45
    delay_sec = target_delay_samples / fs
    delayed = fractional_delay(sig, delay_sec, fs)

    # Energy preservation check (Parseval's theorem for unitary phase shift)
    energy_orig = np.sum(sig ** 2)
    energy_delayed = np.sum(delayed ** 2)
    np.testing.assert_allclose(energy_orig, energy_delayed, rtol=1e-3)

    # Cross-correlation delay estimation (sub-sample resolution < 0.1 samples)
    tdoa, _, _ = estimate_tdoa_gcc_phat(delayed, sig, fs)
    estimated_samples = tdoa * fs
    assert abs(estimated_samples - target_delay_samples) < 0.1



def test_gcc_phat_tdoa_estimation():
    """Verify GCC-PHAT accurately resolves time differences."""
    fs = 60000
    np.random.seed(42)
    noise_sig = np.random.randn(8000)

    delay_sec = 0.00123  # ~73.8 samples
    delayed = fractional_delay(noise_sig, delay_sec, fs)

    measured_tdoa, _, _ = estimate_tdoa_gcc_phat(delayed, noise_sig, fs)
    assert abs(measured_tdoa - delay_sec) < 1e-5


def test_source_localisation_accuracy():
    """Verify full NLS multilateration accurately recovers source coordinates."""
    fs = 60000
    t = np.arange(fs * 1.0) / fs
    # Rich harmonic signal
    audio = (
        np.sin(2 * np.pi * 440 * t)
        + 0.5 * np.sin(2 * np.pi * 880 * t)
        + 0.3 * np.sin(2 * np.pi * 1760 * t)
        + 0.1 * np.random.randn(len(t))
    )

    mics = create_circular_array(num_mics=8, radius=1.0)
    true_source = np.array([2.8, -1.5])

    mic_signals, dists, delays = simulate_microphone_array(
        audio, fs, mics, true_source, c=343.0
    )
    tdoas = compute_pairwise_tdoas(mic_signals, fs, ref_mic=0, mic_coords=mics)
    est_source, _ = localise_source_nls(mics, tdoas, c=343.0, ref_mic=0)

    err_m, err_cm, err_pct = evaluate_localisation_accuracy(true_source, est_source)
    assert err_cm < 5.0, f"Localization error too high: {err_cm:.2f} cm (expected < 5 cm)"


def test_random_source_generation():
    """Verify random source generator produces positions within bounds."""
    for s in [10, 42, 99]:
        pos = generate_random_source(r_min=2.0, r_max=4.0, seed=s)
        dist = np.linalg.norm(pos)
        assert 2.0 <= dist <= 4.0
