import os
import sys
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    AUDIO_FILE,
    TARGET_FS,
    SPEED_OF_SOUND,
    MIC_RADIUS,
    RANDOM_SEED,
)
from part1.audio_io import load_audio, skip_leading_silence
from part1.resampling import resample_audio
from part1.vocal_separation import separate_vocal_and_karaoke
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
    steered_response_power_map,
    evaluate_localisation_accuracy,
)


def run_part3_tests():
    print("=" * 60)
    print("ELEC4150/8150 — PART 3 HEADLESS VERIFICATION TEST")
    print("=" * 60)

    output_dir = os.path.join(PROJECT_ROOT, "outputs", "part-3")
    os.makedirs(output_dir, exist_ok=True)

    passed_checks = 0
    total_checks = 0

    # -------------------------------------------------------------
    # Step 0: Load vocal signal from Part 1
    # -------------------------------------------------------------
    print("\n[Step 0] Loading audio and extracting vocal...")
    audio_path = os.path.join(PROJECT_ROOT, AUDIO_FILE)
    audio, orig_fs = load_audio(audio_path)
    audio = skip_leading_silence(audio, orig_fs)

    # Use 5 seconds of vocal for array simulation
    duration_s = 5.0
    audio_trimmed = audio[: int(orig_fs * duration_s)]
    left_60k = resample_audio(audio_trimmed[:, 0], orig_fs, TARGET_FS)
    right_60k = resample_audio(audio_trimmed[:, 1], orig_fs, TARGET_FS)
    audio_60k = np.stack([left_60k, right_60k], axis=1)
    
    vocal_60k, _, _ = separate_vocal_and_karaoke(audio_60k, TARGET_FS)
    print(f"Input Vocal: {len(vocal_60k)/TARGET_FS:.2f} s at {TARGET_FS} Hz")

    # -------------------------------------------------------------
    # Rubric Item 1: Model System (2 marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 1: Model System & Inverse-Square Attenuation")
    print("-" * 50)
    total_checks += 2

    # Circular array with M=8 microphones, radius = 1.0 m
    num_mics = 8
    mic_coords = create_circular_array(num_mics=num_mics, radius=MIC_RADIUS)
    print(f"Microphone Array: {num_mics} mics uniformly on circle (R = {MIC_RADIUS} m)")

    # Sound source: random or fixed test position
    source_pos = generate_random_source(r_min=2.5, r_max=3.5, seed=RANDOM_SEED)
    src_dist = np.linalg.norm(source_pos)
    print(f"Sound Source Position : ({source_pos[0]:.2f}, {source_pos[1]:.2f}) m (distance: {src_dist:.2f} m)")

    # Distances
    distances = calculate_distances(source_pos, mic_coords)
    min_dist_idx = np.argmin(distances)
    max_dist_idx = np.argmax(distances)
    print(f"Nearest Mic #{min_dist_idx}: distance = {distances[min_dist_idx]:.2f} m")
    print(f"Furthest Mic #{max_dist_idx}: distance = {distances[max_dist_idx]:.2f} m")

    # Inverse-square attenuation check
    attenuated = apply_inverse_square_attenuation(vocal_60k, distances, d0=1.0)
    p_near = np.mean(attenuated[min_dist_idx] ** 2)
    p_far = np.mean(attenuated[max_dist_idx] ** 2)
    expected_ratio = (distances[max_dist_idx] / distances[min_dist_idx]) ** 2
    actual_ratio = p_near / p_far

    print(f"Power Ratio (Nearest/Furthest) Expected: {expected_ratio:.2f}")
    print(f"Power Ratio (Nearest/Furthest) Measured: {actual_ratio:.2f}")

    if abs(actual_ratio - expected_ratio) < 0.05:
        print(">> Check 1a: Inverse-Square Power Law Attenuation: PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 1a: Inverse-Square Attenuation: FAIL ✗")

    if attenuated.shape == (num_mics, len(vocal_60k)):
        print(">> Check 1b: Array output shape (M=8, N): PASS ✓")
        passed_checks += 1
    else:
        print(">> Check 1b: Array output shape: FAIL ✗")

    # -------------------------------------------------------------
    # Rubric Item 2: Delay Simulation & Sub-Sample Resolution (2 marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 2: Propagation Delays & Sub-Sample Resolution")
    print("-" * 50)
    total_checks += 2

    # Speed of sound propagation delays
    delays_theo = calculate_propagation_delays(distances, c=SPEED_OF_SOUND, relative_to_min=True)
    print(f"Speed of Sound: {SPEED_OF_SOUND} m/s")
    for m in range(num_mics):
        print(f"  Mic {m}: d = {distances[m]:.3f} m, delay = {delays_theo[m]*1000:.3f} ms ({delays_theo[m]*TARGET_FS:.2f} samples)")

    # Simulate complete array reception with sub-sample delays
    mic_signals, _, delays_applied = simulate_microphone_array(
        vocal_60k, TARGET_FS, mic_coords, source_pos, c=SPEED_OF_SOUND, relative_delays=True
    )

    # Check propagation delay correctness
    max_delay_ms = np.max(delays_applied) * 1000.0
    expected_max_delay_ms = (np.max(distances) - np.min(distances)) / SPEED_OF_SOUND * 1000.0
    if abs(max_delay_ms - expected_max_delay_ms) < 1e-4:
        print(f">> Check 2a: Propagation Delay matches c = {SPEED_OF_SOUND} m/s: PASS ✓")
        passed_checks += 1
    else:
        print(">> Check 2a: Propagation Delay check: FAIL ✗")

    # Sub-sample delay precision verification
    frac_sample_test = 0.37
    dt_sec = frac_sample_test / TARGET_FS
    delayed_frac = fractional_delay(vocal_60k, dt_sec, TARGET_FS)
    measured_frac_tdoa, _, _ = estimate_tdoa_gcc_phat(delayed_frac, vocal_60k, TARGET_FS)
    measured_frac_samples = measured_frac_tdoa * TARGET_FS
    sample_error = abs(measured_frac_samples - frac_sample_test)
    print(f"Sub-sample Test Delay: {frac_sample_test} samples ({dt_sec*1e6:.2f} µs)")
    print(f"Measured Sub-sample  : {measured_frac_samples:.4f} samples")
    print(f"Sub-sample Error     : {sample_error:.4f} samples (< 0.15 samples / 0.8 mm)")

    if sample_error < 0.15:
        print(">> Check 2b: Continuous Sub-Sample Delay Resolution: PASS ✓")
        passed_checks += 1
    else:
        print(">> Check 2b: Continuous Sub-Sample Delay Resolution: FAIL ✗")

    # Plot Microphone signals waveform
    fig_wf, ax_wf = plt.subplots(figsize=(11, 5))
    t_plot_ms = np.arange(600) / TARGET_FS * 1000.0
    for m in range(num_mics):
        ax_wf.plot(
            t_plot_ms,
            mic_signals[m, :600],
            label=f"Mic {m} (d={distances[m]:.2f}m, τ={delays_applied[m]*1000:.1f}ms)",
            alpha=0.75,
            linewidth=1.0,
        )
    ax_wf.set_title(f"Part 3: Received Signals at 8-Mic Array (Showing Distance Attenuation & Delays)")
    ax_wf.set_xlabel("Time (ms)")
    ax_wf.set_ylabel("Amplitude")
    ax_wf.legend(loc="upper right", ncol=2, fontsize=8)
    ax_wf.grid(True, alpha=0.4)
    fig_wf.tight_layout()
    fig_wf_path = os.path.join(output_dir, "microphone_signals_waveform.png")
    fig_wf.savefig(fig_wf_path, dpi=150)
    plt.close(fig_wf)
    print(f"Saved: {fig_wf_path}")

    # -------------------------------------------------------------
    # Rubric Item 3: Sound Source Localisation & Accuracy (3 marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUBRIC ITEM 3: Sound Source Localisation & Accuracy Evaluation")
    print("-" * 50)
    total_checks += 2

    # Measure TDOAs from simulated multi-channel vocal audio using GCC-PHAT
    measured_tdoas = compute_pairwise_tdoas(
        mic_signals, TARGET_FS, ref_mic=0, c=SPEED_OF_SOUND, mic_coords=mic_coords
    )

    # Localise source coordinates using Non-linear Least Squares (NLS) multilateration
    estimated_pos, opt_res = localise_source_nls(
        mic_coords, measured_tdoas, c=SPEED_OF_SOUND, ref_mic=0
    )

    # Evaluate accuracy
    err_m, err_cm, err_pct = evaluate_localisation_accuracy(source_pos, estimated_pos)

    print(f"True Source Position      : ({source_pos[0]:.4f}, {source_pos[1]:.4f}) m")
    print(f"Estimated Source Position : ({estimated_pos[0]:.4f}, {estimated_pos[1]:.4f}) m")
    print(f"Localisation Error        : {err_cm:.3f} cm ({err_m*1000:.1f} mm)")
    print(f"Relative Position Error   : {err_pct:.3f} %")

    if err_cm < 5.0:
        print(f">> Check 3a: Source Localisation Accuracy ({err_cm:.2f} cm < 5.0 cm): PASS ✓")
        passed_checks += 1
    else:
        print(f">> Check 3a: Source Localisation Accuracy: FAIL ✗ ({err_cm:.2f} cm)")

    # Plot Array Geometry & Localisation Result
    fig_geo, ax_geo = plt.subplots(figsize=(8, 8))
    # Draw array perimeter circle
    circ = plt.Circle((0, 0), MIC_RADIUS, color="gray", fill=False, linestyle="--", label=f"Mic Array Ring (R={MIC_RADIUS}m)")
    ax_geo.add_patch(circ)
    # Plot Mics
    ax_geo.scatter(mic_coords[:, 0], mic_coords[:, 1], color="red", s=80, zorder=5, label="Microphones (M=8)")
    for m in range(num_mics):
        ax_geo.annotate(f"M{m}", (mic_coords[m, 0] * 1.15, mic_coords[m, 1] * 1.15), fontsize=9, fontweight="bold", ha="center")

    # Plot True Source & Estimated Source
    ax_geo.scatter(source_pos[0], source_pos[1], color="blue", marker="*", s=200, zorder=6, label=f"True Source ({source_pos[0]:.2f}, {source_pos[1]:.2f})")
    ax_geo.scatter(estimated_pos[0], estimated_pos[1], color="lime", marker="x", s=150, linewidth=2.5, zorder=7, label=f"Estimated Source ({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f})\nError: {err_cm:.2f} cm")

    # Ray lines from mics to source
    for m in range(num_mics):
        ax_geo.plot([mic_coords[m, 0], source_pos[0]], [mic_coords[m, 1], source_pos[1]], color="gray", alpha=0.2, linestyle=":")

    ax_geo.set_title(f"Part 3: 2D Microphone Array Geometry & Source Localisation\nError = {err_cm:.2f} cm ({err_m*1000:.1f} mm)")
    ax_geo.set_xlabel("X Position (meters)")
    ax_geo.set_ylabel("Y Position (meters)")
    ax_geo.set_xlim(-4.5, 4.5)
    ax_geo.set_ylim(-4.5, 4.5)
    ax_geo.set_aspect("equal")
    ax_geo.grid(True, alpha=0.4)
    ax_geo.legend(loc="upper left")
    fig_geo.tight_layout()
    fig_geo_path = os.path.join(output_dir, "array_geometry_and_source.png")
    fig_geo.savefig(fig_geo_path, dpi=150)
    plt.close(fig_geo)
    print(f"Saved: {fig_geo_path}")

    # Compute and Plot 2D SRP-PHAT Acoustic Power Heatmap
    print("\nGenerating 2D SRP-PHAT acoustic energy heatmap...")
    X_grid, Y_grid, srp_map, best_srp_pos = steered_response_power_map(
        mic_signals[:, : int(TARGET_FS * 1.0)], TARGET_FS, mic_coords,
        x_range=(-4.0, 4.0), y_range=(-4.0, 4.0), grid_res=0.15, c=SPEED_OF_SOUND
    )
    srp_err_m, srp_err_cm, _ = evaluate_localisation_accuracy(source_pos, best_srp_pos)
    print(f"SRP-PHAT Peak: ({best_srp_pos[0]:.2f}, {best_srp_pos[1]:.2f}) m (Error: {srp_err_cm:.1f} cm)")

    fig_srp, ax_srp = plt.subplots(figsize=(8, 7))
    hm = ax_srp.pcolormesh(X_grid, Y_grid, srp_map, shading="auto", cmap="viridis")
    fig_srp.colorbar(hm, ax=ax_srp, label="Steered Acoustic Response Power")
    ax_srp.scatter(mic_coords[:, 0], mic_coords[:, 1], color="red", s=50, label="Microphones", zorder=5)
    ax_srp.scatter(source_pos[0], source_pos[1], color="cyan", marker="*", s=160, label="True Source", zorder=6)
    ax_srp.scatter(best_srp_pos[0], best_srp_pos[1], color="magenta", marker="x", s=120, label=f"SRP-PHAT Peak ({best_srp_pos[0]:.1f}, {best_srp_pos[1]:.1f})", zorder=7)
    ax_srp.set_title(f"Part 3: Steered Response Power (SRP-PHAT) 2D Spatial Heatmap")
    ax_srp.set_xlabel("X (meters)")
    ax_srp.set_ylabel("Y (meters)")
    ax_srp.set_aspect("equal")
    ax_srp.legend(loc="upper left")
    fig_srp.tight_layout()
    fig_srp_path = os.path.join(output_dir, "srp_phat_spatial_heatmap.png")
    fig_srp.savefig(fig_srp_path, dpi=150)
    plt.close(fig_srp)
    print(f"Saved: {fig_srp_path}")

    # Plot GCC-PHAT Cross-correlation curves
    fig_gcc, axes_gcc = plt.subplots(4, 2, figsize=(12, 8))
    axes_gcc = axes_gcc.ravel()
    for m in range(num_mics):
        _, cc, lags_sec = estimate_tdoa_gcc_phat(mic_signals[m], mic_signals[0], TARGET_FS)
        mask_lags = np.abs(lags_sec) <= 0.010
        axes_gcc[m].plot(lags_sec[mask_lags] * 1000, cc[mask_lags], color="steelblue", linewidth=1.1)
        axes_gcc[m].axvline(delays_theo[m] * 1000, color="red", linestyle="--", label=f"True TDOA: {delays_theo[m]*1000:.2f}ms")
        axes_gcc[m].axvline(measured_tdoas[m] * 1000, color="green", linestyle=":", label=f"GCC Peak: {measured_tdoas[m]*1000:.2f}ms")
        axes_gcc[m].set_title(f"Mic {m} vs Mic 0 (TDOA = {measured_tdoas[m]*1000:.2f} ms)", fontsize=9)
        axes_gcc[m].set_xlabel("Lag (ms)", fontsize=8)
        axes_gcc[m].legend(fontsize=7)
        axes_gcc[m].grid(True, alpha=0.3)
    fig_gcc.suptitle("Part 3: GCC-PHAT Cross-Correlation Functions (Showing Sharp TDOA Peaks)", fontsize=11)
    fig_gcc.tight_layout()
    fig_gcc_path = os.path.join(output_dir, "tdoa_gcc_phat_peaks.png")
    fig_gcc.savefig(fig_gcc_path, dpi=150)
    plt.close(fig_gcc)
    print(f"Saved: {fig_gcc_path}")

    if os.path.exists(fig_geo_path) and os.path.exists(fig_srp_path):
        print(">> Check 3b: Visualisation figures generated: PASS ✓")
        passed_checks += 1
    else:
        print(">> Check 3b: Visualisation figures: FAIL ✗")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PART 3 VERIFICATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    print("=" * 60)
    return passed_checks == total_checks


if __name__ == "__main__":
    success = run_part3_tests()
    sys.exit(0 if success else 1)
