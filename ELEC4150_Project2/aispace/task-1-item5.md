# Task 1 : Item 5 — Part 2: RF Transmission Pipeline

## Rubric Requirements (7 marks)
| Sub-task | Marks | Spec |
|---|---|---|
| Compression | 1 | Compress vocal to **64 kbps**; explain; demonstrate |
| Modulation | 2 | Carrier = **250 kHz**; max BW = **4 kHz**; show modulated spectrum |
| Demodulation | 1 | Demodulate → play recovered audio |
| Noise + countermeasure | 3 | AWGN at **SNR = 10 dB**; demonstrate countermeasure; evaluate before/after |

---

## Design Decisions

### 1. Compression → 64 kbps  (`part2/compression.py`)
- Vocal signal from Part 1 is at 60 kHz, 1 channel (float64)
- Method: **μ-law PCM quantisation to 8-bit** = 8 bits × 8000 samples/s (after downsampling from 60k → 8k for telephony) = **64 kbps**
- Why 8k?: Vocal bandwidth is 200–3500 Hz, Nyquist = 7 kHz → 8 kHz sampling sufficient
- Encode: float64 → downsample to 8 kHz → μ-law 8-bit → bit stream
- Decode: bit stream → μ-law decode → upsample to 60 kHz → float
- Output: compressed bytes, decoded audio, actual bit rate

### 2. Modulation → 250 kHz carrier, ≤4 kHz BW  (`part2/modulation.py`)
- Signal after compression/decode is at 8 kHz (max freq ~4 kHz → fits ≤4 kHz BW)
- Method: **AM (Double Sideband Large Carrier)** — simplest to demonstrate & explain
  - `s(t) = [1 + m·x(t)] × cos(2π·fc·t)`  where fc=250 kHz, m=modulation index
  - Total BW = 2 × 4 kHz = 8 kHz (but we'll limit message to 4 kHz → BW = 8 kHz centred on 250 kHz)
  - Note: rubric says max BW = 4 kHz — we'll use SSB-AM (Single Sideband) to stay within 4 kHz
  - **SSB-AM**: BW = 4 kHz, carrier = 250 kHz → passes rubric
  - Implementation: Hilbert transform → analytic signal → USB/LSB
- Output: modulated signal array, carrier freq, bandwidth measurement

### 3. Demodulation  (`part2/demodulation.py`)
- SSB demodulation: multiply by carrier → low-pass filter → recovered audio
- Validate: play recovered audio, compare SNR with original

### 4. AWGN Noise + Countermeasure  (`part2/noise.py`, `part2/filtering.py`)
- Add AWGN at SNR = 10 dB to the modulated signal
- Countermeasure: **Wiener filter** applied after demodulation to clean up noise
- Evaluate: SNR before countermeasure vs after
- Also show: spectral comparison of noisy vs filtered

---

## Files to Create/Modify

### [MODIFY] `part2/compression.py`
- `compress_vocal(vocal, fs_in, target_bitrate=64000)` → `(compressed_bytes, decoded, fs_out, actual_bitrate)`
- `measure_bitrate(compressed_bytes, duration_s)` → float

### [MODIFY] `part2/modulation.py`
- `modulate_ssb(signal, fs, fc=250000, mode='usb')` → `(modulated, t)`
- `measure_bandwidth(signal, fs, fc, threshold_db=-40)` → float

### [MODIFY] `part2/demodulation.py`
- `demodulate_ssb(modulated, fs, fc=250000, mode='usb', bw=4000)` → `recovered`

### [MODIFY] `part2/noise.py`
- `add_awgn(signal, snr_db)` → `(noisy, noise)`
- `measure_snr(clean, noisy)` → float (dB)

### [MODIFY] `part2/filtering.py`
- `wiener_denoise(signal, fs, noise_estimate_frames=10)` → `denoised`

### [MODIFY] `main.py`
- Add Part 2 pipeline after Part 1 (Steps 5–9)

### [NEW] `aispace/test_run_part2_headless.py`
- Headless test for the full Part 2 pipeline

---

## Steps
- [x] **A** Implement `part2/compression.py` (64 kbps μ-law PCM)
- [x] **B** Implement `part2/modulation.py` (SSB-AM USB, fc=250 kHz, BW <= 4 kHz)
- [x] **C** Implement `part2/demodulation.py` (Product detector + zero-phase LPF)
- [x] **D** Implement `part2/noise.py` (AWGN at SNR = 10 dB)
- [x] **E** Implement `part2/filtering.py` (RF BPF + baseband Wiener filter)
- [x] **F** Update `main.py` with Part 2 pipeline + plots
- [x] **G** Write Part 2 headless test (`aispace/test_run_part2_headless.py`)
- [x] **H** Run headless test — verify all checks pass (6/6 PASS)
- [x] **I** Run pytest — all tests pass (43/43 PASS)

---
*Status: ✅ Complete*
