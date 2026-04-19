# Chapter 4 — Experimental Setup (6–8 pages)

File: `experiments/experiments.tex` (CREATE this directory + files)

---

## Section 4.1: Synthetic Data Generation (2–3 pages)

### What to write

**SyntheticDataGenerator Class**
- Generates signals of the form y = x_true + f_true + noise
- Controlled generation with known ground truth for all components
- This enables supervised training and precise quantitative evaluation

**Peak Generation**
- Shape: piecewise-linear peaks (triangular) and Gaussian peaks
- Number of peaks: configurable range (typically 1–many per signal)
- Peak width: 1–7 samples for N=1024 (very sharp, characteristic of chromatography)
- Peak height: 13–215× baseline amplitude (wide dynamic range)
- All peaks are positive (physical constraint — detector response is additive)
- Peak overlap: controlled probability of overlapping peaks
- Position: random placement along signal length

**Baseline Generation**
- Method 1: Low-order polynomials (degree 2–4)
  - Models gradual, monotonic drift
- Method 2: Sinusoidal combinations
  - Models periodic temperature fluctuations
- Method 3: Spline interpolation of random control points
  - Models arbitrary smooth drift patterns
- Baselines are always smooth and low-frequency (physical constraint)

**Noise Model**
- Additive Gaussian noise: w ~ N(0, sigma^2)
- Controllable SNR via sigma parameter
- v7 uses sigma = 0.01 (normalized scale)
- Noise is i.i.d. — models electronic detector noise

**Dataset Configuration**
- Signal length: N = 1024
- Total signals: 200 (can be configured)
- Train/test split: 80/20 (160 train, 40 test)
- Normalization: signals normalized to consistent amplitude range

### Key details from code
- `SyntheticDataGenerator` class defined in `train.py` (all versions v2+)
- v2 introduced the generator; v3-v7 refined peak/baseline properties
- Peak width carefully tuned to be chromatographically realistic
- Baseline amplitude calibrated relative to peak heights

### Figures & Tables
- **Fig 4.1**: Example synthetic data samples — 3-4 panels showing:
  - (a) Complete signal y = x + f + w
  - (b) True peaks x_true
  - (c) True baseline f_true
  - (d) Noise w
  Generate from `demo.py`
- **Table 4.1**: Data generation parameter ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| N | 1024 | Signal length |
| num_peaks | 1–varied | Number of peaks |
| peak_width | 1–7 samples | Peak FWHM |
| peak_height | 13–215× | Height relative to baseline |
| baseline_type | poly/sin/spline | Drift model |
| noise_std | 0.01 | Gaussian noise level |

---

## Section 4.2: Training Protocol (2–3 pages)

### What to write

**Architecture Configuration**
| Parameter | Value | Notes |
|-----------|-------|-------|
| K (layers) | 8 | Unrolled BEADS iterations |
| N (signal length) | 1024 | Training signal length |
| fc | 0.006 | Butterworth cutoff (fixed) |
| d | 1 | Filter order |
| shared_params | False | Independent params per layer |
| learn_step | True | Learnable step sizes |
| learn_output_gain | True | Learnable output scaling |

**Optimizer**
- Adam optimizer
- Learning rate: 1e-3
- No LR scheduling between stages (continuous optimization)
- Batch size: 24

**Three-Stage Curriculum (reference Ch. 3.6 for motivation)**

| Stage | Epochs | Active Losses | Purpose |
|-------|--------|---------------|---------|
| A | 5 | MSE only | Establish basic separation |
| B | 15 | MSE + baseline + asymmetric + ortho | Teach correct separation |
| C | 10 | Full 11-term composite | Fine-tune leakage suppression |

**Loss Weight Schedule**

| Loss Term | Stage A | Stage B | Stage C |
|-----------|---------|---------|---------|
| alpha_mse | 1.0 | 1.0 | 1.0 |
| alpha_l1 | 0.0 | 0.0 | 0.01 |
| alpha_tv | 0.0 | 0.0 | 0.01 |
| alpha_smooth | 0.0 | 0.0 | 0.2 |
| alpha_neg | 0.0 | 0.0 | 0.5 |
| alpha_baseline | 0.0 | 0.5 | 0.5 |
| alpha_leakage | 0.0 | 0.0 | 0.3 |
| alpha_ortho | 0.0 | 0.1 | 0.1 |
| alpha_asym_baseline | 0.0 | 1.0 | 1.0 |
| alpha_envelope | 0.0 | 0.0 | 0.5 |
| alpha_freq | 0.0 | 0.0 | 0.05 |

**Intermediate Supervision**
- When enabled (Stage C): compute loss at every unrolled layer output
- Layer weights: linearly increasing 0.1 → 1.0 (more weight on refined estimates)
- Prevents early layers from learning bad representations

**Computational Details**
- Device: CPU (stable), optional CUDA/MPS support
- Typical training time: 30 epochs total (5+15+10)
- Checkpoints saved per epoch

### Key details from code
- `train.py` contains the full training loop with stage management
- `stage_configs` list defines the three stages
- MLflow integration for experiment tracking
- ProcessManager for training orchestration (via Orchestration/ web interface)

### Figures & Tables
- **Fig 4.2**: Training loss curves across 3 stages — show total loss and key components
  - Generate from training logs or MLflow data
- **Table 4.2**: Hyperparameter summary (combined table from above)

---

## Section 4.3: Evaluation Metrics (1–2 pages)

### What to write

**Peak Reconstruction Metrics**

| Metric | Formula | Purpose |
|--------|---------|---------|
| MSE(x, x_hat) | (1/N) sum(x_i - x_hat_i)^2 | Overall peak estimation accuracy |
| MAE(x, x_hat) | (1/N) sum|x_i - x_hat_i| | Robust to outliers |
| Correlation | corr(x, x_hat) | Shape similarity (scale-invariant) |

**Peak Localization Metrics**

| Metric | Formula | Purpose |
|--------|---------|---------|
| Peak position MAE | mean|pos_true - pos_pred| | Location accuracy (in samples) |
| Peak match rate | #matched / #true × 100% | Detection completeness |

**Baseline Metrics**

| Metric | Formula | Purpose |
|--------|---------|---------|
| MSE(f, f_hat) | (1/N) sum(f_i - f_hat_i)^2 | Baseline estimation accuracy |
| Area error | |integral(x) - integral(x_hat)| / integral(x) | Quantification accuracy |

**Quality Metrics (for hybrid inference)**

| Metric | Formula | Purpose |
|--------|---------|---------|
| baseline_hf_ratio | energy(high-freq f_hat) / energy(f_hat) | Baseline smoothness |
| residual_hf_rms | RMS of high-freq residual | Noise in separation |

- Define each precisely with formula
- Explain which metrics matter most for chromatographic applications (area error is critical for quantification)

### Key details from code
- Metrics computed in `demo.py` evaluation section
- Quality metrics defined in `analysis/hybrid_diagnostics.py`
- Correlation with ground truth: 0.73–0.86 observed in v6 diagnostics

---

## Section 4.4: Baseline Comparison Methods (1–2 pages)

### What to write

**Classical BEADS**
- Use manually-tuned parameters as the "oracle" baseline
- Parameters: fc=0.006, r=6.0, d=1, lam0/lam1/lam2 tuned per signal family
- 30 iterations to convergence
- Implementation: your `beads.py` (v0)

**pybaselines Methods**
- arPLS: Asymmetric Reweighted PLS (Baek et al.)
- airPLS: Adaptive Iteratively Reweighted PLS
- SNIP: Statistics-sensitive Non-linear Iterative Peak-clipping
- AsLS: Asymmetric Least Squares
- All from the `pybaselines` Python package
- Parameters: use default or literature-recommended values

**Parameter Selection Strategy**
- For classical methods: grid search over reasonable parameter ranges
- Report both "default" and "best-tuned" results for fair comparison
- LBEADS-NET advantage: no per-signal tuning needed (parameters are learned)

**Real Chromatogram Data**
- From the original BEADS paper dataset
- Used in `demo_chromatogram.py`
- No ground truth available → qualitative comparison + quality metrics

### Key details from code
- v4+ compare against pybaselines implementations
- `demo_chromatogram.py` loads real chromatogram data
- Hybrid diagnostics compare raw LBEADS vs refined vs hybrid outputs

### Tables
- **Table 4.3**: Comparison methods summary — method, parameters, source
