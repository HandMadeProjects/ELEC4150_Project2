# Project Execution Guide — ELEC4150/8150 Project 2

This guide provides simple, to-the-point instructions on how to run each part of the project **live** with interactive graphics and audio playback.

---

## 1. Environment Setup

The project dependencies (`matplotlib`, `scipy`, `sounddevice`, etc.) are installed in the **`baikarms`** conda environment.

In your PowerShell terminal, you can either **activate the environment**:
```powershell
conda activate baikarms
```
...and then run commands normally with `python ...`.

**OR** run them directly with `conda run -n baikarms`:
```powershell
conda run -n baikarms python run_part1.py
```

> **Audio Track**: Configured to use `audio_3-GANGNAM_STYLE.wav` (PSY — Gangnam Style) across all parts.
> You can also specify any audio file using `--audio <path_to_wav>`.

---

## 2. How to Run Each Part Live

### **Part 1: Modular Audio Processing & Live DJ Spectrogram**

Part 1 fulfills the three core aims defined in `cp.txt`:
- **Part 1.1**: Convert the audio to 60 kHz and play full audio + live moving DJ spectrogram.
- **Part 1.2**: Extract vocals only from song file, saving `extracted_vocal.wav`.
- **Part 1.3**: Karaoke track: delete vocals and keep only the tune without vocals, saving `karaoke_instrumental.wav`.

#### **How to Run:**
```powershell
conda activate baikarms

# Part 1.1: Plays 60 kHz full song + live DJ moving spectrogram (Default)
python run_part1.py

# Part 1.2: Plays extracted vocals only + live DJ moving spectrogram
python run_part1.py --track vocal

# Part 1.3: Plays karaoke tune (vocals deleted) + live DJ moving spectrogram
python run_part1.py --track karaoke
```

#### **Generated Files in `outputs/part-1/`:**
- `full_song_60k.wav` — Audio resampled to 60,000 Hz (**Part 1.1**)
- `extracted_vocal.wav` — Clean center-channel vocal extraction (**Part 1.2**)
- `karaoke_instrumental.wav` — Instrumental tune with vocals deleted (**Part 1.3**, ≥ 40 dB vocal attenuation)
- `part1_summary_plots.png` — Multi-panel waveforms and 0–10 kHz frequency spectrum comparison

#### **Live DJ Visualizer Features:**
- **Top Deck**: Real-time DJ Spectrum Analyzer with live frequency bouncing (cyan curve) and peak-hold decay (pink dashed line) across Bass, Mid/Vocals, and Treble bands, plus live RMS decibel readout.
- **Bottom Deck**: Continuous rolling waterfall spectrogram where audio continuously enters at the live `NOW` playhead on the right and scrolls smoothly into the past on the left in real time!

---

### **Part 2: RF Transmission Pipeline**
```powershell
conda activate baikarms
python run_part2.py
```
*(Or in one line: `conda run -n baikarms python run_part2.py`)*

**What you will see and hear (spec order):**
1. Compress the Part 1 vocal to **64 kbps** (ITU-T G.711 μ-law: 8 bit × 8 kHz).
2. SSB-AM USB onto a **250 kHz** carrier, occupied bandwidth **≤ 4 kHz**. The plot shows the vocal spectrum moved from 0–4 kHz up to 250–254 kHz.
3. Demodulate the **clean** RF and **play it** to validate the link (no noise yet).
4. Add **AWGN at SNR = 10 dB**. The RF spectrum and the noisy demodulated waveform show the damage.
5. Dual-stage countermeasure: RF bandpass around the 4 kHz USB, then baseband Wiener. SNR before/after is printed and plotted.
6. Play the **noisy demod**, then the **recovered vocal**, so you can hear the countermeasure.

---

### **Part 3: Microphone Array & Source Localisation**
```powershell
conda activate baikarms
python run_part3.py
```
*(Or in one line: `conda run -n baikarms python run_part3.py`)*

**What you will see:**
1. Simulates an **8-microphone circular array** (radius $R = 1.0$ m).
2. Simulates acoustic sound propagation from a sound source at $c = 343$ m/s with:
   - **Inverse-square power law attenuation** ($P \propto 1/d^2$).
   - **Sub-sample fractional propagation delays** via Fourier phase shift ($e^{-j2\pi f \tau}$).
3. Estimates pairwise TDOAs using **GCC-PHAT** with sub-sample parabolic peak interpolation.
4. Solves 2D source coordinates using **Non-linear Least Squares (NLS) multilateration** (achieving **$< 1$ mm accuracy**).
5. Opens interactive figures:
   - **2D Geometry Map**: Circular array layout, true source position, and estimated source position.
   - **Multi-channel Waveforms**: Shows amplitude decay and time-of-flight delays across all 8 mics.
   - **2D SRP-PHAT Heatmap**: Acoustic energy map across the room space pointing to the source.

---

## 3. Run Entire Project End-to-End

To execute all three parts sequentially in one run:

```powershell
conda activate baikarms
python main.py
```
*(Or in one line: `conda run -n baikarms python main.py`)*

---

## 4. Run Automated Tests

To run the complete test suite (49 automated tests across all parts):

```bash
pytest
```

All tests will execute and report pass/fail status in seconds.

---

## 5. Output Artifacts Directory

All generated high-resolution plots and audio `.wav` files are automatically saved in the `outputs/` folder:

- **Part 1**: `outputs/part-1/`
  - `vocal_waveform.png`
  - `karaoke_waveform.png`
  - `spectrum_comparison.png`
  - `vocal_spectrogram.png`
- **Part 2**: `outputs/part-2/`
  - `modulated_spectrum.png`
  - `noise_countermeasure_comparison.png`
  - `recovered_vocal_waveform.png`
  - `recovered_vocal.wav` (playable audio file)
  - `noisy_recovered_vocal.wav` (baseline without countermeasure)
- **Part 3**: `outputs/part-3/`
  - `array_geometry_and_source.png`
  - `microphone_signals_waveform.png`
  - `tdoa_gcc_phat_peaks.png`
  - `srp_phat_spatial_heatmap.png`
