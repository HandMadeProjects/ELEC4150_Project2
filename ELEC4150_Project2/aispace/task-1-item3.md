# Task 1 : Item 3 — Organise Project Files & Update CLAUDE.md

## Current State Assessment

### Problems Found

| Issue | Detail |
|-------|--------|
| ❌ Misplaced file | `part1/part1/realtime_spectrogram.py` is nested in a duplicate `part1/` subfolder |
| ❌ `realtime_spectrogram.py` not integrated | It exists but is never imported or called from `main.py` |
| ❌ `main.py` is a flat script | All logic is inlined — it does not use `part1/` modules at all |
| ❌ Empty stubs | `part2/` and `part3/` files are all empty (placeholders only) |
| ❌ `part1/karaoke.py` is empty | Karaoke/instrumental extraction module not implemented |
| ❌ No `outputs/` directory | No dedicated folder for generated plots/audio |
| ❌ CLAUDE.md lacks project-info reference | `project-info/` directory (PDFs) not mentioned |

### What IS in good shape

- `part1/audio_io.py`, `resampling.py`, `vocal_separation.py`, `spectrogram.py`, `plotting.py` — all implemented  
- `tests/` — 37 passing tests  
- `config.py` — clean global constants  
- `requirements.txt` + `requirements-lock.txt` — complete  
- `aispace/` — task tracking in place  

---

## Proposed Changes

### Step 3.1 — Fix misplaced `realtime_spectrogram.py`
Move `part1/part1/realtime_spectrogram.py` → `part1/realtime_spectrogram.py`  
Then delete the now-empty `part1/part1/` directory.

### Step 3.2 — Create `outputs/` directory
Add `outputs/` at project root with a `.gitkeep` and a README.  
This is where generated plots (PNG) and processed audio will be saved.

### Step 3.3 — Refactor `main.py` to use module imports
Replace the inlined logic in `main.py` with calls to the `part1/` modules:
- `audio_io.load_audio()` / `validate_audio()`
- `resampling.resample_audio()`
- `vocal_separation.separate_vocal_stereo()`
- `spectrogram.calculate_spectrogram()`
- `realtime_spectrogram.play_with_realtime_spectrogram()` ← fixes the audio+spectrogram lag issue
- `plotting.plot_waveform()`
- Save figures to `outputs/`

> ⚠️ **This also fixes the reported lag issue**: `realtime_spectrogram.py` already uses a
> proper callback-based architecture (`sounddevice.OutputStream` + queue) that runs audio
> and spectrogram simultaneously without blocking. The current `main.py` uses `sd.play()` +
> `sd.wait()` which is sequential (audio blocks the display).

### Step 3.4 — Update CLAUDE.md
- Add `project-info/` section (rubric + spec PDFs)
- Add `outputs/` to the directory tree
- Update the structure diagram to reflect the fixed layout
- Add "How to run" section with conda env instructions

### Step 3.5 — Update `aispace/task-index.md`
Add Item 3 row.

---

## ⚠️ Clarification Needed

> **Q: For Step 3.3 (main.py refactor)** — The real-time spectrogram function runs audio AND spectrogram simultaneously (this is the fix for the lag issue you described). However, `main.py` currently also has:
> - A static spectrogram plot (plt.specgram)
> - A spectrum comparison plot (before vs after vocal extraction)
> - Simple `sd.play()` + `sd.wait()` playback
>
> Should I:
> - **(A — Recommended)** Replace the sequential audio+spectrogram section with `play_with_realtime_spectrogram()`, keep the static plots (spectrum comparison, etc.)
> - **(B)** Keep `main.py` exactly as-is, just fix imports/structure only

**Please confirm before I execute Step 3.3.**

---

## Steps Checklist

- [x] **3.1** Moved `part1/part1/realtime_spectrogram.py` → `part1/realtime_spectrogram.py`, removed duplicate dir
- [x] **3.2** Created `outputs/` directory with `.gitkeep`
- [x] **3.3** Refactored `main.py` — uses all `part1/` modules + real-time spectrogram (lag fixed)
- [x] **3.4** Updated `CLAUDE.md` — full directory tree, project-info/, outputs/, how-to-run section
- [x] **3.5** Updated `aispace/task-index.md`

**Verification:** 37/37 tests still passing after restructure ✅

---
*Status: ✅ Complete*
