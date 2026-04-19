# PROMPT: Write Chapter 4 — Experimental Setup (6–8 pages)

## ROLE
You are an expert academic writer producing the experimental setup chapter of an NYU Tandon MS thesis. This chapter must be precise enough that someone could replicate the experiments. LaTeX output.

## OUTPUT
Produce the full LaTeX content for `experiments/experiments.tex`. Begin with `\chapter{Experimental Setup}`. Contains Sections 4.1–4.4.

## SOURCE FILES TO READ

**Primary — read these fully:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/train.py` — Read the `SyntheticDataGenerator` class carefully (peak generation, baseline generation, noise model, dataset creation). Read the training loop: optimizer, learning rate, batch size, stage configs, intermediate supervision logic. Read loss weight values in each stage config.
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — Read the `__init__` parameters of `LBEADS_NET`: num_layers, signal_length, fc, d, shared_params, learn_step, learn_output_gain. Read default values.
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo.py` — Read how evaluation is done: which metrics are computed, how predictions are compared to ground truth, how results are visualized.

**For comparison methods:**
- `Implementations/0. BEADS/WithNumpy/beads.py` — Classical BEADS parameters used (fc=0.006, r=6.0, d=1)
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo_chromatogram.py` — How real chromatogram data is loaded and processed

**For leakage metrics:**
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Section 7: evaluation metrics (BLI, peak-region RMSE, stratified evaluation, peak area preservation, peak height preservation, non-negativity fraction, residual autocorrelation)

**For diagnostics data:**
- `Implementations/6. LBEADS_NETv6 [Strong]/analysis/` — Read hybrid_diagnostics.txt if available for actual metric values

**Reference:**
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 4 specification

## SECTION SPECIFICATIONS

### Section 4.1: Synthetic Data Generation (2–3 pages)

**Present the `SyntheticDataGenerator` formally. Extract exact parameters from `train.py`.**

1. **Overview**: Generate signals y = x_true + f_true + w with known ground truth for all three components. This enables supervised training and precise quantitative evaluation.

2. **Peak generation** — extract from `SyntheticDataGenerator`:
   - Shape: describe the peak model used (Gaussian, piecewise-linear, or other — check the actual code)
   - Number of peaks: configurable range
   - Peak width: 1–7 samples for N=1024 (very sharp, characteristic of chromatography)
   - Peak height: 13–215× baseline amplitude (dynamic range)
   - All peaks positive (physical constraint)
   - Overlap: configurable probability of overlapping peaks
   - Position: random placement along signal length

3. **Baseline generation** — three methods:
   - Low-order polynomials (degree 2–4): models gradual monotonic drift
   - Sinusoidal combinations: models periodic temperature fluctuations
   - Spline interpolation of random control points: models arbitrary smooth drift
   - All baselines smooth and low-frequency (physical constraint)

4. **Noise model**:
   - Additive Gaussian: w ~ N(0, σ²)
   - σ = 0.01 (normalized scale) in v7
   - i.i.d. — models electronic detector noise

5. **Dataset configuration**:
   - Signal length: N = 1024
   - Total signals: 200 (check actual code for exact number)
   - Train/test split: 80/20 (160 train, 40 test)
   - Normalization: describe any normalization applied

**Table 4.1**: Data generation parameter ranges (parameter | range | description)

**Fig 4.1**: Example synthetic samples — 4 panels:
(a) Complete signal y
(b) True peaks x_true
(c) True baseline f_true
(d) Noise w
Reference `demo.py` output or `synthetic_data_samples.png` from any version.

### Section 4.2: Training Protocol (2–3 pages)

**Architecture table** — Table 4.2 (extract exact values from code):
| Parameter | Value | Notes |
| K (unrolled layers) | 8 | Upgraded from 5 in v1 |
| N (signal length) | 1024 | Training signal length |
| fc (cutoff frequency) | 0.006 | Fixed hyperparameter |
| d (filter order) | 1 | Butterworth order |
| shared_params | False | Independent params per layer |
| learn_step | True | Learnable step sizes |
| learn_output_gain | True | Learnable output scaling |

**Optimizer**: Adam, lr=1e-3, no learning rate scheduling between stages (continuous optimization), batch_size=24

**Three-stage curriculum** — reference Ch. 3.6 for motivation:

| Stage | Epochs | Active Losses | Purpose |
| A | 5 | MSE only | Basic separation |
| B | 15 | MSE + baseline + asymmetric + ortho | Correct separation |
| C | 10 | Full 11-term composite | Fine-tune |

**Full loss weight schedule** — extract exact values from `stage_configs` in train.py:

Create a large table with all 11 loss weights × 3 stages. Every value must match the actual code.

**Intermediate supervision** (Stage C):
- Compute loss at every unrolled layer output
- Layer weights: linearly increasing from 0.1 to 1.0
- Early layers weighted less (their estimates are rough)
- Late layers weighted more (refined estimates)

**Computational details:**
- Device: CPU (primary, stable), optional CUDA/MPS
- Total epochs: 30 (5+15+10)
- Checkpoints saved per epoch
- MLflow integration for experiment tracking

**Fig 4.2**: Training loss curves across 3 stages — total loss vs epoch with stage boundary markers. Reference training output from v7.

### Section 4.3: Evaluation Metrics (1–2 pages)

**Present ALL metrics with precise mathematical definitions.**

**Peak reconstruction metrics:**
| Metric | Formula | Purpose |
| MSE(x, x̂) | (1/N)Σ(xᵢ - x̂ᵢ)² | Overall estimation accuracy |
| MAE(x, x̂) | (1/N)Σ|xᵢ - x̂ᵢ| | Robust to outliers |
| Pearson correlation | corr(x, x̂) | Shape similarity (scale-invariant) |

**Peak localization metrics:**
| Peak position MAE | mean|pos_true - pos_pred| | Location accuracy (samples) |
| Peak match rate | #matched/#true × 100% | Detection completeness |

**Baseline metrics:**
| MSE(f, f̂) | (1/N)Σ(fᵢ - f̂ᵢ)² | Baseline estimation accuracy |
| Area error | |∫x - ∫x̂| / ∫x | Quantification accuracy |

**Leakage-specific metrics** (from leakage research):
| BLI (Baseline Leakage Index) | ReLU(f̂-f_true) · peak_mask / peak_energy | Over-estimation in peak regions |
| Peak-to-baseline error ratio | RMSE(f̂[peak]) / RMSE(f̂[non-peak]) | >1 indicates leakage |
| Stratified evaluation | All metrics computed separately for sparse vs dense regions | Quantifies degradation by difficulty |

**Hybrid quality metrics:**
| baseline_hf_ratio | energy(high-freq f̂) / energy(f̂) | Baseline smoothness |
| residual_hf_rms | RMS of high-freq residual | Noise in separation |

### Section 4.4: Baseline Comparison Methods (1–2 pages)

**Classical BEADS** (oracle baseline):
- Manually tuned parameters: fc=0.006, r=6.0, d=1
- λ₀, λ₁, λ₂ tuned per signal family
- 30 iterations to convergence
- Implementation: `0. BEADS/WithNumpy/beads.py`

**pybaselines methods** (using pybaselines Python package):
- arPLS (Baek et al., 2015)
- airPLS (Zhang et al., 2010)
- SNIP (Ryan et al., 1988)
- AsLS (Eilers, 2003)
- Each with default parameters + grid-searched best parameters

**Parameter selection**:
- Classical methods: grid search over reasonable parameter ranges
- Report both "default" and "best-tuned" results for fair comparison
- LBEADS-NET advantage: NO per-signal tuning needed

**Real chromatogram data**:
- From original BEADS paper dataset
- Loaded in `demo_chromatogram.py`
- No ground truth → qualitative comparison + quality metrics

**Table 4.3**: Comparison methods (method | parameters | source | notes)

## STYLE CONSTRAINTS
- Reproducibility is the goal: another researcher should be able to replicate your experiments from this chapter
- All numbers must match the actual code — check train.py and lbeads_net.py for exact values
- Use `\begin{table}` for all tables, `\begin{figure}` for figures
- Cross-reference Ch. 3 for method details: "The composite loss function described in Section~\ref{sec:loss}..."
- Do not describe the method here — only the experimental setup
