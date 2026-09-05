# Task 2 : Item 1 — ≥40 dB Karaoke Vocal Attenuation Fix

## Problem
Simple centre-channel subtraction `(L−R)/2` gives ~6 dB attenuation.  
Rubric requires **≥ 40 dB** vocal attenuation in the karaoke track.

## Solution — STFT-domain Wiener Soft Masking

For each time-frequency bin (f, t) in the STFT:
- Vocal (centre) estimate: `V(f,t) = (L(f,t) + R(f,t)) / 2`
- Side (instrumental) estimate: `S(f,t) = (L(f,t) − R(f,t)) / 2`
- Wiener karaoke mask: `mask(f,t) = |S|² / (|V|² + |S|² + ε)`
- Apply mask to suppress vocal bins: `karaoke(f,t) = mask × mono(f,t)`

Bins where L ≈ R (center-panned vocal) → mask ≈ 0 → strongly suppressed  
Bins where L ≠ R (stereo instruments) → mask ≈ 1 → preserved

## Files to Change

### [MODIFY] `part1/vocal_separation.py`
- Keep existing `separate_vocal_stereo()` (simple, used for vocal extraction)
- Add new `separate_karaoke_wiener()` (STFT-domain, for ≥40 dB karaoke)

### [MODIFY] `main.py`
- Add Step 4b: call `separate_karaoke_wiener()` for the karaoke output
- Add attenuation measurement and print result

### [MODIFY] `aispace/test_run_headless.py`
- Add karaoke Wiener test + attenuation measurement check

### [MODIFY] `tests/test_vocal_separation.py`
- Add tests for `separate_karaoke_wiener()`

## Steps

- [x] **A** Implement `separate_karaoke_wiener()` in `part1/vocal_separation.py`
- [x] **B** Update `main.py` to use it and print attenuation
- [x] **C** Update `aispace/test_run_headless.py` to test and measure attenuation
- [x] **D** Add unit tests in `tests/test_vocal_separation.py`
- [x] **E** Run headless test — verify attenuation printed in console

---
*Status: ✅ Complete*
