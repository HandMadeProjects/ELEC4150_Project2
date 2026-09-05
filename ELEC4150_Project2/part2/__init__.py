"""
ELEC4150/8150 Project 2 — Part 2: RF Transmission Pipeline
==========================================================
Modules:
  - compression: μ-law PCM encoding/decoding achieving exactly 64 kbps.
  - modulation: Single-Sideband (SSB-AM) modulation at fc = 250 kHz with BW <= 4 kHz.
  - demodulation: Coherent product detector and recovery of baseband vocal.
  - noise: AWGN generation at target SNR (10 dB) and SNR calculation.
  - filtering: RF front-end bandpass filtering and baseband Wiener denoising countermeasures.
"""

from part2.compression import compress_vocal, measure_bitrate, compression_snr_db
from part2.modulation import modulate_ssb, measure_bandwidth
from part2.demodulation import demodulate_ssb, evaluate_demodulation
from part2.noise import add_awgn, calculate_snr
from part2.filtering import (
    apply_anti_aliasing_lpf,
    apply_rf_bandpass,
    apply_wiener_denoise,
    apply_countermeasure_pipeline,
)

__all__ = [
    "compress_vocal",
    "measure_bitrate",
    "compression_snr_db",
    "modulate_ssb",
    "measure_bandwidth",
    "demodulate_ssb",
    "evaluate_demodulation",
    "add_awgn",
    "calculate_snr",
    "apply_anti_aliasing_lpf",
    "apply_rf_bandpass",
    "apply_wiener_denoise",
    "apply_countermeasure_pipeline",
]
