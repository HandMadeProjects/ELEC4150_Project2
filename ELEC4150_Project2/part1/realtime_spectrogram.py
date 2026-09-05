"""
ELEC4150/8150 Project 2 — Real-Time DJ-Style Moving Spectrogram & Spectrum Analyzer
=====================================================================================
Plays extracted vocal audio cleanly through speakers while driving a dynamic,
moving real-time visualizer featuring:
  1. DJ Spectrum Analyzer: instantaneous frequency response curve + peak-hold decay
  2. Rolling Waterfall Spectrogram: continuously scrolling time-frequency visualizer
     where incoming audio enters live on the right and rolls into history on the left.
"""

import time
from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import sounddevice as sd

from part1.spectrogram import calculate_spectrogram


def play_with_realtime_spectrogram(audio, fs, duration=20):
    """
    Play audio while displaying a live moving DJ-style rolling spectrogram
    and real-time spectrum analyzer.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio signal at sampling rate fs (e.g. 60 000 Hz).
    fs : int or float
        Sampling rate in Hz (e.g. 60 000 Hz).
    duration : float
        Maximum playback demonstration duration in seconds (default 20 s).
    """

    # Limit to demonstration duration
    max_samples = int(duration * fs)
    audio_segment = np.asarray(audio[:max_samples], dtype=np.float32)
    actual_duration = len(audio_segment) / float(fs)

    print("\n========================================")
    print("LIVE DJ REAL-TIME MOVING SPECTROGRAM")
    print("========================================")
    print(f"Sampling Rate     : {fs} Hz")
    print(f"Playback Duration : {actual_duration:.2f} seconds")

    # --------------------------------------------------------
    # 1. Compute Full STFT at 60 kHz
    # --------------------------------------------------------
    print("Computing STFT matrix for live rolling visualizer...")
    freqs, times, mag_db = calculate_spectrogram(audio_segment, fs)

    # Restrict display to musical/vocal frequencies (0–8 kHz)
    max_freq = 8000.0
    f_mask = freqs <= max_freq
    disp_freqs = freqs[f_mask]
    disp_mag = mag_db[f_mask, :]

    # Decimate frequency bins slightly for ultra-responsive 30+ FPS rendering
    f_sub = 2 if len(disp_freqs) > 100 else 1
    disp_freqs = disp_freqs[::f_sub]
    disp_mag = disp_mag[::f_sub, :]

    num_bins, total_frames = disp_mag.shape
    dt = times[1] - times[0] if len(times) > 1 else 512.0 / fs

    # --------------------------------------------------------
    # 2. Setup Audio Playback at Native Hardware Rate
    # --------------------------------------------------------
    try:
        dev_info = sd.query_devices(kind="output")
        native_fs = int(dev_info["default_samplerate"])
    except Exception:
        native_fs = 44100

    print(f"Output Sound Card : {dev_info.get('name', 'Default')}")
    print(f"Playback Rate     : {native_fs} Hz (hardware native)")

    # Resample to native hardware rate to avoid PortAudio WASAPI underflow
    if native_fs != fs:
        g = gcd(native_fs, int(fs))
        audio_play = resample_poly(
            audio_segment.astype(np.float64),
            native_fs // g,
            int(fs) // g
        ).astype(np.float32)
    else:
        audio_play = audio_segment

    # Normalise audio playback volume and apply gentle 50 ms edge fades
    peak = np.max(np.abs(audio_play))
    if peak > 0:
        audio_play = (audio_play / peak) * 0.88

    fade_len = min(len(audio_play) // 10, int(native_fs * 0.05))
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        audio_play[:fade_len] *= fade_in
        audio_play[-fade_len:] *= fade_out

    # Duplicate mono to stereo for dual-speaker output
    if audio_play.ndim == 1:
        audio_play = np.column_stack([audio_play, audio_play])

    # --------------------------------------------------------
    # 3. Setup DJ-Style Rolling GUI (Dark Theme)
    # --------------------------------------------------------
    plt.ion()
    fig, (ax_spec, ax_wf) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1.6]}
    )
    fig.patch.set_facecolor("#0a0b0e")
    fig.canvas.manager.set_window_title("ELEC4150 — Live DJ Real-Time Spectrogram")

    # --- Panel 1: Top DJ Spectrum Analyzer ---
    ax_spec.set_facecolor("#111318")
    ax_spec.set_title(
        "DJ SPECTRUM ANALYZER — REAL-TIME FREQUENCY BOUNCE & PEAK HOLD",
        color="#00f5d4",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_spec.set_xlabel("Frequency (Hz)", color="#a0aec0", fontsize=9)
    ax_spec.set_ylabel("Magnitude (dB)", color="#a0aec0", fontsize=9)
    ax_spec.set_xlim(0, max_freq)
    ax_spec.set_ylim(-70, 5)
    ax_spec.tick_params(colors="#a0aec0", labelsize=8)
    ax_spec.grid(True, color="#2d3748", linestyle="--", linewidth=0.5, alpha=0.7)

    # Visual guide lines for DJ frequency bands
    ax_spec.axvline(250, color="#f56565", linestyle=":", alpha=0.5)
    ax_spec.axvline(2000, color="#ecc94b", linestyle=":", alpha=0.5)
    ax_spec.axvline(6000, color="#48bb78", linestyle=":", alpha=0.5)
    ax_spec.text(125, -67, "BASS", color="#f56565", fontsize=8, ha="center")
    ax_spec.text(1125, -67, "MIDS / VOCALS", color="#ecc94b", fontsize=8, ha="center")
    ax_spec.text(4000, -67, "PRESENCE", color="#48bb78", fontsize=8, ha="center")
    ax_spec.text(7000, -67, "AIR", color="#63b3ed", fontsize=8, ha="center")

    # Animated curves: live spectrum + peak hold line
    init_spectrum = np.full(num_bins, -70.0)
    spec_line, = ax_spec.plot(disp_freqs, init_spectrum, color="#00f5d4", linewidth=1.5, label="Live")
    peak_line, = ax_spec.plot(disp_freqs, init_spectrum, color="#ff007f", linewidth=1.0, linestyle="--", label="Peak Hold")
    vu_text = ax_spec.text(
        0.98, 0.90, "RMS: -inf dB",
        transform=ax_spec.transAxes,
        color="#00f5d4",
        fontsize=10,
        fontweight="bold",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a202c", edgecolor="#00f5d4", alpha=0.8)
    )

    # --- Panel 2: Bottom Rolling Waterfall Spectrogram ---
    ax_wf.set_facecolor("#111318")
    ax_wf.set_title(
        "LIVE ROLLING WATERFALL SPECTROGRAM — CONTINUOUS TIME SCROLL (DJ CONSOLE)",
        color="#ff007f",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_wf.set_xlabel("Rolling Time Window (seconds into the past  <---  NOW)", color="#a0aec0", fontsize=9)
    ax_wf.set_ylabel("Frequency (Hz)", color="#a0aec0", fontsize=9)
    ax_wf.tick_params(colors="#a0aec0", labelsize=8)

    # Rolling window size: W frames (~1.5 seconds rolling view)
    W = 160
    window_duration = W * dt
    waterfall_buffer = np.full((num_bins, W), -80.0, dtype=np.float32)

    img = ax_wf.imshow(
        waterfall_buffer,
        aspect="auto",
        origin="lower",
        extent=[-window_duration, 0.0, disp_freqs[0], disp_freqs[-1]],
        vmin=-65,
        vmax=-5,
        cmap="magma",
    )

    # Fixed neon "NOW / PLAYHEAD" line at the right edge
    ax_wf.axvline(0, color="#00f5d4", linewidth=2.0, linestyle="-", zorder=10)
    ax_wf.text(
        0.99, 0.93, "PLAYHEAD (NOW)",
        transform=ax_wf.transAxes,
        color="#00f5d4",
        fontsize=9,
        fontweight="bold",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a202c", alpha=0.7)
    )

    time_badge = ax_wf.text(
        0.02, 0.93, "Position: 0.00 s / {:.2f} s".format(actual_duration),
        transform=ax_wf.transAxes,
        color="#ffffff",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a202c", edgecolor="#ff007f", alpha=0.8)
    )

    cbar = fig.colorbar(img, ax=ax_wf, orientation="horizontal", pad=0.18, aspect=40)
    cbar.set_label("Magnitude (dB)", color="#a0aec0", fontsize=8)
    cbar.ax.tick_params(colors="#a0aec0", labelsize=7)

    fig.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)

    # --------------------------------------------------------
    # 4. Asynchronous Live Audio Playback & Real-Time Loop
    # --------------------------------------------------------
    print("\n>> Audio playback started. Enjoy the live DJ rolling visualizer!")
    print(">> Close the visualizer window or press Ctrl+C to stop.")

    peak_hold = np.full(num_bins, -70.0)
    decay_factor = 0.94  # Smooth peak decay

    try:
        sd.play(audio_play, native_fs)
        start_time = time.time()

        while True:
            # Detect if user closed window
            if not plt.fignum_exists(fig.number):
                sd.stop()
                print("\nVisualizer closed by user.")
                break

            elapsed = time.time() - start_time
            if elapsed >= actual_duration:
                break

            # Current playback frame index in STFT
            curr_frame = int(elapsed / dt)
            curr_frame = min(max(0, curr_frame), total_frames - 1)

            # 1. Update rolling waterfall matrix (rolling right to left)
            start_f = max(0, curr_frame - W)
            chunk = disp_mag[:, start_f:curr_frame]
            k = chunk.shape[1]
            if k > 0:
                waterfall_buffer[:, :-k] = waterfall_buffer[:, k:]
                waterfall_buffer[:, -k:] = chunk
                img.set_data(waterfall_buffer)

            # 2. Update DJ Spectrum Analyzer (live bounce + peak hold)
            curr_slice = disp_mag[:, curr_frame]
            peak_hold = np.maximum(peak_hold * decay_factor, curr_slice)

            spec_line.set_ydata(curr_slice)
            peak_line.set_ydata(peak_hold)

            # 3. Update RMS readout and time badge
            rms_sample_idx = int(elapsed * fs)
            rms_chunk = audio_segment[max(0, rms_sample_idx - 1024):rms_sample_idx + 1024]
            if len(rms_chunk) > 0:
                rms_val = np.sqrt(np.mean(rms_chunk ** 2) + 1e-12)
                rms_db = 20 * np.log10(rms_val)
            else:
                rms_db = -60.0

            vu_text.set_text(f"RMS: {rms_db:5.1f} dB")
            time_badge.set_text(f"Position: {elapsed:5.2f} s / {actual_duration:.2f} s")

            # 4. Redraw & yield GUI events smoothly (~30 FPS)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.02)

        sd.wait()
        print("\nPlayback finished.")

    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped by user.")
    except Exception as e:
        sd.stop()
        print(f"\nPlayback error: {e}")
    finally:
        sd.stop()
        plt.ioff()
        plt.close(fig)
