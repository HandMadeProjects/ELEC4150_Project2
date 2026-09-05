# Task 2 : Item 2 — Part 1 Modular Aims Update (1.1, 1.2, 1.3)

## Aims & Objectives Defined in cp.txt
- **Part 1.1**: Convert the audio to 60k Hz, save 60k full song, then play audio + live moving DJ spectrogram.
- **Part 1.2**: Only extract vocals from song file, save `extracted_vocal.wav`, plot time & frequency domain comparisons.
- **Part 1.3**: Karaoke track: delete vocals and keep only tune without vocals, save `karaoke_instrumental.wav`, verify vocal attenuation (≥ 20 dB measured).

## Output Files in `outputs/part-1/`
1. `full_song_60k.wav` — Audio resampled to 60,000 Hz (Part 1.1)
2. `extracted_vocal.wav` — Extracted center-panned vocal audio (Part 1.2)
3. `karaoke_instrumental.wav` — Instrumental tune with vocals removed (Part 1.3)
4. `part1_summary_plots.png` — Multi-panel waveforms & spectrum comparison (Original, Vocal, Karaoke)
5. `part1_1_full_song_spectrum.png` — Spectrum of 60 kHz resampled song
6. `part1_2_vocal_analysis.png` — Waveform and spectrum of extracted vocal
7. `part1_3_karaoke_analysis.png` — Waveform and spectrum of karaoke tune

## Proposed Changes
1. **`part1/audio_io.py`**:
   - Add `save_audio(file_path, audio, fs)` to save clean PCM WAV files using `soundfile`.
2. **`run_part1.py`**:
   - Explicitly structure and log according to the three distinct sub-parts:
     - `[Part 1.1] Resample to 60 kHz & Full Song Live DJ Spectrogram`
     - `[Part 1.2] Extract Vocals Only & Save Waveform/Spectrum`
     - `[Part 1.3] Karaoke: Delete Vocals & Keep Tune Only`
   - Save all three `.wav` outputs to `outputs/part-1/`.
   - Support track selection via `--track [full|vocal|karaoke]` (defaulting to `full` song at 60 kHz as specified in 1.1, while supporting vocal or karaoke as well).
3. **`main.py`**:
   - Ensure `main.py` reflects the same three sub-parts and exports all three `.wav` files.
4. **`guide.md`**:
   - Update `guide.md` with the exact commands to run each subpart (1.1, 1.2, 1.3) live.

## Steps
- [x] **A** Add `save_audio` in `part1/audio_io.py` and unit tests in `tests/test_audio_io.py`
- [x] **B** Update `run_part1.py` with structured 1.1, 1.2, 1.3 pipeline, WAV saving, and track selection
- [x] **C** Update `main.py` to match the 1.1, 1.2, 1.3 structure and WAV exports
- [x] **D** Update `guide.md` with instructions for running Part 1.1, 1.2, and 1.3
- [x] **E** Run tests & verify in `baikarms`

---
*Status: ✅ Complete*
