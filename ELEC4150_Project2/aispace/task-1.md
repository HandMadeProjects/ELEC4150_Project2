# Task 1 — ELEC4150 Project 2 Setup & Testing

## Overview
Three sub-tasks for setting up and validating the ELEC4150 Project 2 workspace:
- **Item 0**: Create/update `CLAUDE.md` (exclude `cp.txt` from all reviews)
- **Item 1**: Create conda env `baikarms` and install all required libraries
- **Item 2**: Create test cases, run the project and verify results

---

## Item 0 — Create / Update CLAUDE.md

- [x] **0.1** Created `CLAUDE.md` at the project root — `cp.txt` excluded from all AI reviews.

---

## Item 1 — Conda Environment Setup (`baikarms`)

- [x] **1.1** Created conda env `baikarms` (Python 3.11)
- [x] **1.2** Installed all required libraries from `requirements.txt`:
  - numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.1, librosa 0.11.0, sounddevice 0.5.6, soundfile 0.14.0
- [x] **1.3** Verified all imports — all pass ✅
- [x] **1.4** Generated `requirements-lock.txt` with pinned versions

---

## Item 2 — Create Test Cases & Run the Project

### 2a — Create Unit Tests

- [x] **2.1** Created `tests/` directory
- [x] **2.2** Created `tests/test_audio_io.py` — 10 tests
- [x] **2.3** Created `tests/test_resampling.py` — 7 tests
- [x] **2.4** Created `tests/test_vocal_separation.py` — 10 tests
- [x] **2.5** Created `tests/test_spectrogram.py` — 10 tests
- [x] **2.6** Created `tests/conftest.py` with shared fixtures

### 2b — Run Tests

- [x] **2.7** Ran full pytest suite: **37/37 PASSED in 9.89s** ✅
- [x] **2.8** No failures — no fixes needed

### 2c — Run the Main Script

> ⚠️ **Pending user action**: Step 2.9 runs `main.py` in interactive mode (audio plays via speakers, 3 matplotlib figures open). Please run: `conda activate baikarms && python main.py`

- [ ] **2.9** Run `main.py` interactively
- [ ] **2.10** Confirm figures and audio playback

---

## Notes / Decisions Log

| # | Question | Answer |
|---|----------|--------|
| 1 | Run `main.py` interactively or headless? | Interactive (audio + figures) |

---
*Status: 🟢 Tests complete — awaiting main.py run confirmation*
