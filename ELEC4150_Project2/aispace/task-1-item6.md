# Task 1 : Item 6 — Part 3: Microphone Array & Source Localisation

## Rubric Requirements (7 marks)
| Sub-task | Marks | Specification |
|---|---|---|
| **Model system** | 2 | Inverse-square power law ($P \propto 1/d^2$); takes audio, $f_s$, mic coordinates, random source $(x,y)$; outputs mic signals using Part 1 vocal. |
| **Delay simulation** | 2 | Speed of sound $c = 343$ m/s; relative propagation time delays for each mic based on distance; **sub-sample delay resolution** achieved and demonstrated. |
| **Sound source localisation** | 3 | Deduce sound source location $(x,y)$ using relative delays and array geometry; explain method; demonstrate result; evaluate accuracy (error distance). |

---

## Technical Design Decisions

### 1. Microphone Array Geometry (`part3/microphone_model.py`)
- **Array Configuration**: Circular array with $M = 8$ microphones placed uniformly on a circle of radius $R = 1.0$ m centered at $(0, 0)$, matching `config.py` (`MIC_RADIUS = 1.0 m`):
  $$\theta_m = \frac{2\pi m}{M}, \quad x_m = R \cos(\theta_m), \quad y_m = R \sin(\theta_m) \quad (m = 0, \dots, M-1)$$
- **Sound Source**: Randomly generated position $(x_s, y_s)$ at distance $r \in [2.0, 5.0]$ m from array center.
- **Inverse-Square Law Attenuation**:
  - Distance: $d_m = \sqrt{(x_s - x_m)^2 + (y_s - y_m)^2}$
  - Power reduces with square of distance: $P(d_m) \propto 1/d_m^2$
  - Signal amplitude scales as: $\alpha_m = \frac{d_0}{d_m}$, where $d_0 = 1.0$ m is reference distance.

### 2. Propagation Delay & Sub-Sample Resolution (`part3/delay.py`)
- Acoustic propagation delay: $\tau_m = \frac{d_m}{c}$ seconds ($c = 343$ m/s).
- Number of samples delayed: $D_m = \tau_m \cdot f_s = \frac{d_m \cdot f_s}{c}$.
- **Sub-Sample Fractional Delay Method**:
  - Frequency-domain linear phase shift via FFT:
    $$X_{\text{delayed}}(f) = X(f) \cdot e^{-j 2\pi f \tau_m}$$
  - Bandlimited sinc interpolation property of continuous-time inverse Fourier transform guarantees **exact sub-sample delay resolution** without waveform distortion or aliasing.

### 3. TDOA & Source Localisation (`part3/localisation.py`)
- **TDOA Estimation**:
  - Generalized Cross-Correlation with Phase Transform (**GCC-PHAT**) between microphone pairs $(i, j)$:
    $$R_{ij}(f) = \frac{X_i(f) \cdot X_j^*(f)}{|X_i(f) \cdot X_j^*(f)| + \epsilon}$$
    $$\text{CC}_{ij}(\tau) = \text{IFFT}\{ R_{ij}(f) \}$$
  - Parabolic sub-sample interpolation around the cross-correlation peak to retrieve continuous time differences $\hat{\tau}_{ij}$.
- **Position Estimation (Multilateration)**:
  - Theoretical distance difference: $\Delta d_{ij}(\mathbf{p}) = \|\mathbf{p} - \mathbf{p}_i\| - \|\mathbf{p} - \mathbf{p}_j\|$
  - Non-linear Least Squares (NLS) optimization:
    $$\min_{(x_s, y_s)} \sum_{i < j} \left( \Delta d_{ij}(x_s, y_s) - c \cdot \hat{\tau}_{ij} \right)^2$$
  - Also produce **SRP-PHAT (Steered Response Power)** 2D spatial acoustic power heatmap over the room grid.
- **Accuracy Evaluation**:
  - Localization error metric: $E = \sqrt{(x_s - \hat{x}_s)^2 + (y_s - \hat{y}_s)^2}$ (in mm / cm).

---

## Files to Create/Modify

### [MODIFY] `part3/microphone_model.py`
- `create_circular_array(num_mics=8, radius=1.0)` -> array coordinates $(M, 2)$
- `calculate_distances(source_pos, mic_coords)` -> distance vector $(M,)$
- `apply_attenuation(signal, distances, d0=1.0)` -> attenuated signals $(M, N)$
- `simulate_array(audio, fs, mic_coords, source_pos, c=343.0)` -> multi-channel signals $(M, N)$

### [MODIFY] `part3/delay.py`
- `fractional_delay(signal, delay_seconds, fs)` -> delayed signal using FFT phase shift
- `apply_propagation_delays(signal, distances, fs, c=343.0)` -> delayed signals $(M, N)$
- `demonstrate_subsample_resolution(fs=60000)` -> verification of sub-sample shift

### [MODIFY] `part3/localisation.py`
- `estimate_tdoa_gcc_phat(sig1, sig2, fs, max_tau=None)` -> $\tau_{ij}$ with parabolic peak interpolation
- `compute_all_tdoas(mic_signals, fs, ref_mic=0)` -> vector of measured TDOAs
- `localise_source_nls(mic_coords, tdoas, fs, c=343.0, initial_guess=(0,0))` -> $(\hat{x}_s, \hat{y}_s)$
- `steered_response_power_map(mic_signals, fs, mic_coords, grid_x, grid_y, c=343.0)` -> 2D power grid

### [MODIFY] `part3/__init__.py`
- Expose all primary Part 3 functions.

### [NEW] `aispace/test_run_part3_headless.py`
- Headless verification script testing all Part 3 rubric items.

### [NEW] `tests/test_part3.py`
- Pytest suite for geometry, fractional delay, GCC-PHAT, and localization accuracy.

### [MODIFY] `main.py`
- Integrate Part 3 pipeline and save outputs to `outputs/part-3/`.

---

## Steps
- [x] **A** Implement `part3/microphone_model.py` (circular array geometry & inverse-square attenuation)
- [x] **B** Implement `part3/delay.py` (speed of sound propagation & exact FFT sub-sample delay)
- [x] **C** Implement `part3/localisation.py` (GCC-PHAT TDOA, NLS multilateration, SRP-PHAT heatmap)
- [x] **D** Expose modules in `part3/__init__.py`
- [x] **E** Write unit tests in `tests/test_part3.py` (6/6 tests passing)
- [x] **F** Write and run headless verification script `aispace/test_run_part3_headless.py` (6/6 checks passing)
- [x] **G** Integrate Part 3 into `main.py` and generate all figures in `outputs/part-3/`
- [x] **H** Run full pytest suite across Part 1, 2, and 3 (49/49 tests passing)

---
*Status: ✅ Complete*
