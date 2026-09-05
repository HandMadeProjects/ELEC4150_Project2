"""
ELEC4150/8150 Project 2 — Part 3 Live Runner
============================================
Runs Part 3:
  - Circular Microphone Array Simulation (8 mics, R = 1.0 m)
  - Inverse-Square Power Attenuation (P ~ 1/d^2)
  - Speed of Sound Sub-Sample Propagation Delays (c = 343 m/s)
  - GCC-PHAT TDOA Estimation
  - Non-linear Least Squares (NLS) Source Localisation
  - Interactive 2D Spatial Geometry & Heatmap Visualisation
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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
    simulate_microphone_array,
)
from part3.delay import calculate_propagation_delays
from part3.localisation import (
    estimate_tdoa_gcc_phat,
    compute_pairwise_tdoas,
    localise_source_nls,
    steered_response_power_map,
    evaluate_localisation_accuracy,
)

OUTPUTS_DIR = os.path.join("outputs", "part-3")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("=" * 60)
print("RUNNING PART 3: MICROPHONE ARRAY & SOURCE LOCALISATION")
print("=" * 60)

# Step 1: Load Vocal from Audio 2
print("\n[1/5] Loading vocal audio input...")
audio, orig_fs = load_audio(AUDIO_FILE)
audio = skip_leading_silence(audio, orig_fs)
duration_s = 5.0
audio_trimmed = audio[: int(orig_fs * duration_s)]
left_60k = resample_audio(audio_trimmed[:, 0], orig_fs, TARGET_FS)
right_60k = resample_audio(audio_trimmed[:, 1], orig_fs, TARGET_FS)
vocal, _, _ = separate_vocal_and_karaoke(np.stack([left_60k, right_60k], axis=1), TARGET_FS)

# Step 2: Configure Array & Source
print("\n[2/5] Creating 8-microphone circular array & placing source...")
num_mics = 8
mic_coords = create_circular_array(num_mics=num_mics, radius=MIC_RADIUS)
source_pos = generate_random_source(r_min=2.5, r_max=3.5, seed=RANDOM_SEED)
distances = calculate_distances(source_pos, mic_coords)
delays_theo = calculate_propagation_delays(distances, c=SPEED_OF_SOUND, relative_to_min=True)

print(f"Array Radius: {MIC_RADIUS} m ({num_mics} microphones)")
print(f"True Source Position: ({source_pos[0]:.3f}, {source_pos[1]:.3f}) m")
print(f"Speed of sound: {SPEED_OF_SOUND} m/s")
for m in range(num_mics):
    print(
        f"  Mic {m}: ({mic_coords[m, 0]:.2f}, {mic_coords[m, 1]:.2f}) m  "
        f"d={distances[m]:.3f} m  τ={delays_theo[m]*1000:.3f} ms "
        f"({delays_theo[m]*TARGET_FS:.2f} samples)"
    )

# Step 3: Simulate Propagation & Reception
print("\n[3/5] Simulating propagation delays (c = 343 m/s) and inverse-square attenuation...")
mic_signals, _, delays_applied = simulate_microphone_array(
    vocal, TARGET_FS, mic_coords, source_pos, c=SPEED_OF_SOUND, relative_delays=True
)

# Step 4: Estimate TDOAs & Localise Source
print("\n[4/5] Estimating TDOAs via GCC-PHAT and solving position...")
measured_tdoas = compute_pairwise_tdoas(
    mic_signals, TARGET_FS, ref_mic=0, c=SPEED_OF_SOUND, mic_coords=mic_coords
)
estimated_pos, _ = localise_source_nls(
    mic_coords, measured_tdoas, c=SPEED_OF_SOUND, ref_mic=0
)
err_m, err_cm, err_pct = evaluate_localisation_accuracy(source_pos, estimated_pos)

print(f"True Source Coordinates      : ({source_pos[0]:.4f}, {source_pos[1]:.4f}) m")
print(f"Estimated Source Coordinates : ({estimated_pos[0]:.4f}, {estimated_pos[1]:.4f}) m")
print(f"Localisation Error Distance  : {err_cm:.3f} cm ({err_m*1000:.1f} mm)")

# Step 5: Interactive Visualisations
print("\n[5/5] Generating live interactive spatial figures...")

# Figure 1: 2D Geometry and Localisation Map
fig_geo, ax_geo = plt.subplots(figsize=(8, 8))
circ = plt.Circle((0, 0), MIC_RADIUS, color="gray", fill=False, linestyle="--", label=f"Mic Array (R={MIC_RADIUS}m)")
ax_geo.add_patch(circ)
ax_geo.scatter(mic_coords[:, 0], mic_coords[:, 1], color="red", s=90, zorder=5, label="Microphones (M=8)")
for m in range(num_mics):
    ax_geo.annotate(f"M{m}", (mic_coords[m, 0] * 1.15, mic_coords[m, 1] * 1.15), fontsize=9, fontweight="bold", ha="center")

ax_geo.scatter(source_pos[0], source_pos[1], color="blue", marker="*", s=220, zorder=6, label=f"True Source ({source_pos[0]:.2f}, {source_pos[1]:.2f})")
ax_geo.scatter(estimated_pos[0], estimated_pos[1], color="lime", marker="x", s=180, linewidth=2.5, zorder=7, label=f"Estimated ({estimated_pos[0]:.2f}, {estimated_pos[1]:.2f})\nError = {err_cm:.2f} cm")
for m in range(num_mics):
    ax_geo.plot([mic_coords[m, 0], source_pos[0]], [mic_coords[m, 1], source_pos[1]], color="gray", alpha=0.2, linestyle=":")

ax_geo.set_title(f"Part 3: 2D Microphone Array & Localisation Result\nError = {err_cm:.2f} cm ({err_m*1000:.1f} mm)")
ax_geo.set_xlabel("X (meters)")
ax_geo.set_ylabel("Y (meters)")
ax_geo.set_xlim(-4.5, 4.5)
ax_geo.set_ylim(-4.5, 4.5)
ax_geo.set_aspect("equal")
ax_geo.grid(True, alpha=0.4)
ax_geo.legend(loc="upper left")
fig_geo.tight_layout()
fig_geo.savefig(os.path.join(OUTPUTS_DIR, "array_geometry_and_source.png"), dpi=150)

# Figure 2: Received Waveforms at all 8 microphones
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
ax_wf.set_title("Part 3: Received Signals at 8 Mics (Showing Inverse-Square Attenuation & Propagation Delays)")
ax_wf.set_xlabel("Time (ms)")
ax_wf.set_ylabel("Amplitude")
ax_wf.legend(loc="upper right", ncol=2, fontsize=8)
ax_wf.grid(True, alpha=0.4)
fig_wf.tight_layout()
fig_wf.savefig(os.path.join(OUTPUTS_DIR, "microphone_signals_waveform.png"), dpi=150)

# Figure 3: SRP-PHAT Spatial Heatmap
print("Computing 2D SRP-PHAT acoustic energy heatmap...")
X_grid, Y_grid, srp_map, best_srp = steered_response_power_map(
    mic_signals[:, : int(TARGET_FS * 1.0)], TARGET_FS, mic_coords,
    x_range=(-4.0, 4.0), y_range=(-4.0, 4.0), grid_res=0.15, c=SPEED_OF_SOUND
)
fig_srp, ax_srp = plt.subplots(figsize=(8, 7))
hm = ax_srp.pcolormesh(X_grid, Y_grid, srp_map, shading="auto", cmap="viridis")
fig_srp.colorbar(hm, ax=ax_srp, label="Steered Acoustic Response Power")
ax_srp.scatter(mic_coords[:, 0], mic_coords[:, 1], color="red", s=60, label="Microphones", zorder=5)
ax_srp.scatter(source_pos[0], source_pos[1], color="cyan", marker="*", s=180, label="True Source", zorder=6)
ax_srp.scatter(best_srp[0], best_srp[1], color="magenta", marker="x", s=140, label=f"SRP Peak ({best_srp[0]:.1f}, {best_srp[1]:.1f})", zorder=7)
ax_srp.set_title("Part 3: Steered Response Power (SRP-PHAT) 2D Spatial Heatmap")
ax_srp.set_xlabel("X (meters)")
ax_srp.set_ylabel("Y (meters)")
ax_srp.set_aspect("equal")
ax_srp.legend(loc="upper left")
fig_srp.tight_layout()
fig_srp.savefig(os.path.join(OUTPUTS_DIR, "srp_phat_spatial_heatmap.png"), dpi=150)

# Figure 4: GCC-PHAT TDOA peaks (sub-sample delay demonstration)
fig_gcc, axes_gcc = plt.subplots(4, 2, figsize=(12, 8))
axes_gcc = axes_gcc.ravel()
true_tdoa_vs_ref = (distances - distances[0]) / SPEED_OF_SOUND
for m in range(num_mics):
    _, cc, lags_sec = estimate_tdoa_gcc_phat(mic_signals[m], mic_signals[0], TARGET_FS)
    mask_lags = np.abs(lags_sec) <= 0.010
    axes_gcc[m].plot(lags_sec[mask_lags] * 1000, cc[mask_lags], color="steelblue", linewidth=1.1)
    axes_gcc[m].axvline(true_tdoa_vs_ref[m] * 1000, color="red", linestyle="--", label=f"True {true_tdoa_vs_ref[m]*1000:.2f} ms")
    axes_gcc[m].axvline(measured_tdoas[m] * 1000, color="green", linestyle=":", label=f"GCC {measured_tdoas[m]*1000:.2f} ms")
    axes_gcc[m].set_title(f"Mic {m} vs Mic 0", fontsize=9)
    axes_gcc[m].set_xlabel("Lag (ms)", fontsize=8)
    axes_gcc[m].legend(fontsize=7)
    axes_gcc[m].grid(True, alpha=0.3)
fig_gcc.suptitle("Part 3: GCC-PHAT TDOA peaks (sub-sample delay resolution)", fontsize=11)
fig_gcc.tight_layout()
fig_gcc.savefig(os.path.join(OUTPUTS_DIR, "tdoa_gcc_phat_peaks.png"), dpi=150)

plt.show()
print("\nPart 3 complete.")
