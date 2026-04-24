# Chapter 4 & 5 — Experiments & Results: Surgical Plan

## Narrative Arc

The experiments tell the story of **how LBEADS-NET was built**, not just what it is. Two phases:

**Phase 1 — Progressive Development (v0→v5):**
Each version adds one key idea. Show metrics improving (or not) at each step.

**Phase 2 — Ablation Study:**
Starting from the full v5/v7 loss configuration, systematically remove terms to show which ones actually matter.

---

## Phase 1: The Development Journey

### Experiment 1.0 — Classical BEADS Baseline (v0)
**What:** Run classical BEADS on the synthetic test set with hand-tuned parameters.
**Source:** `Implementations/0. BEADS/WithNumpy/beads.py`
**Purpose:** Establish the "oracle" upper bound — what's achievable with perfect per-signal tuning.
**Metrics:** Peak MSE, baseline MSE, peak correlation, area error.
**Key result:** Good performance BUT required manual tuning per signal.

### Experiment 1.1 — Basic Unrolling, MSE Only (v1/v2 equivalent)
**What:** Train LBEADS_NET_Fast with ONLY reconstruction loss (alpha_mse=1.0, all others=0).
**Purpose:** Show that naive unrolling learns basic separation but suffers from leakage.
**Expected:** Peaks are roughly recovered, but baseline follows peak shapes.
**Metrics:** Same as above + introduce BLI (Baseline Leakage Index).

### Experiment 1.2 — Add Sparsity Losses (v3 equivalent)
**What:** Add L1 (alpha_l1=0.01) and TV (alpha_tv=0.01) on peaks.
**Purpose:** Show sparsity priors help but WORSEN leakage in dense regions.
**Key insight:** L1 is the CAUSE of leakage, not the cure. This is the central thesis finding.
**Metrics:** Same + stratified sparse/dense region metrics.

### Experiment 1.3 — Add Baseline Supervision (v4 equivalent)
**What:** Add masked baseline reconstruction (alpha_baseline=1.0).
**Purpose:** Show that direct supervision on baseline is the critical breakthrough.
**Expected:** Major improvement — baseline stops following peaks.
**This was the single most impactful change in the entire project.**

### Experiment 1.4 — Add Orthogonality + Non-Negativity (v5 equivalent)
**What:** Add ortho (alpha_ortho=0.5), non-neg (alpha_neg=0.1), smooth (alpha_smooth=0.01), baseline_tv (alpha_baseline_tv=0.1).
**Purpose:** Show incremental improvement from enforcing peak-baseline separation.
**This is the thesis model configuration.**

### Experiment 1.5 — Add Anti-Leakage Battery (v7 equivalent)
**What:** On top of v5, add asymmetric baseline (alpha_asym=1.0), leakage penalty (alpha_leakage=1.0), envelope (alpha_envelope=0.5), freq separation (alpha_freq=0.05).
**Purpose:** Show the full anti-leakage engineering effort.
**Expected:** Further leakage reduction, but diminishing returns.

---

## Phase 2: Ablation Study

Starting from the FULL loss config (all 12 terms), remove groups to show contribution.

### Ablation A — No sparsity priors
Remove: L1, TV (alpha_l1=0, alpha_tv=0)
**Question:** Do sparsity priors help or hurt?

### Ablation B — No baseline supervision
Remove: masked baseline recon, asymmetric baseline (alpha_baseline=0, alpha_asym=0)
**Question:** How critical is direct baseline supervision?

### Ablation C — No leakage-specific penalties
Remove: leakage, ortho, envelope, freq (alpha_leakage=0, alpha_ortho=0, alpha_envelope=0, alpha_freq=0)
**Question:** Do the anti-leakage terms actually help beyond baseline supervision?

### Ablation D — No non-negativity
Remove: alpha_neg=0
**Question:** Does enforcing peak positivity matter?

### Ablation E — Baseline supervision only (minimal config)
Only: alpha_mse=1.0 + alpha_baseline=1.0
**Question:** What's the simplest loss that still works?

### Ablation F — Architecture depth
Fix loss config, vary K: 5, 10, 15, 20, 30 layers
**Question:** How many layers are needed?

---

## Metrics to Report

### Standard metrics (per signal, averaged over test set)
- Peak MSE: mean((x_pred - x_true)^2)
- Peak MAE: mean(|x_pred - x_true|)
- Peak correlation: pearsonr(x_pred, x_true)
- Baseline MSE: mean((f_pred - f_true)^2)
- Area error: |sum(x_pred) - sum(x_true)| / sum(x_true) × 100%

### Leakage-specific metrics
- BLI (Baseline Leakage Index): sum(ReLU(f_pred - f_true) * peak_mask) / sum(x_true * peak_mask)
- PBER (Peak-to-Baseline Error Ratio): RMSE_peak_regions / RMSE_baseline_regions
- Peak area preservation: per-peak area ratio

### Stratified metrics
- Report ALL metrics separately for:
  - Sparse signals (few well-separated peaks)
  - Dense signals (many overlapping peaks)
  - Mixed signals

---

## What Needs to Run

### Scripts to create/modify
1. **eval_progressive.py** — Runs experiments 1.0–1.5
   - Trains v5 model with progressive loss configs
   - Evaluates on shared test set
   - Saves metrics per experiment

2. **eval_ablation.py** — Runs ablations A–F
   - Same architecture, varies loss config
   - Saves metrics per ablation

3. **eval_comparison.py** — Runs classical methods
   - BEADS (tuned), arPLS, airPLS, SNIP, AsLS via pybaselines
   - On same test set

All scripts should save results to a common `results/` directory as JSON/CSV for easy table generation.

---

## Tables for the Thesis

### Table 5.1: Progressive Development Results
| Config | Loss Terms | Peak MSE | Baseline MSE | BLI | Correlation |
|--------|-----------|----------|--------------|-----|-------------|
| 1.0 Classical BEADS (tuned) | — | ... | ... | ... | ... |
| 1.1 MSE only | 1 | ... | ... | ... | ... |
| 1.2 + Sparsity | 3 | ... | ... | ... | ... |
| 1.3 + Baseline supervision | 4 | ... | ... | ... | ... |
| 1.4 + Ortho/non-neg (v5) | 8 | ... | ... | ... | ... |
| 1.5 + Full anti-leakage (v7) | 12 | ... | ... | ... | ... |

### Table 5.2: Ablation Study
| Ablation | Removed Terms | Peak MSE | BLI | Δ from full |
|----------|--------------|----------|-----|-------------|
| Full config | — | ... | ... | baseline |
| A: No sparsity | L1, TV | ... | ... | ... |
| B: No baseline sup. | baseline, asym | ... | ... | ... |
| C: No anti-leakage | leak, ortho, env, freq | ... | ... | ... |
| D: No non-neg | neg | ... | ... | ... |
| E: Minimal | only MSE + baseline | ... | ... | ... |
| F: K=5/10/15/20/30 | (depth varies) | ... | ... | ... |

### Table 5.3: Classical Methods Comparison
| Method | Peak MSE | Baseline MSE | BLI | Tuning Required |
|--------|----------|--------------|-----|-----------------|
| LBEADS-NET (v5) | ... | ... | ... | None (learned) |
| BEADS (tuned) | ... | ... | ... | 6 params/signal |
| arPLS | ... | ... | ... | 1 param |
| airPLS | ... | ... | ... | 1 param |
| SNIP | ... | ... | ... | 1 param |

---

## Figures to Generate

- **Fig 5.1**: Training convergence curves (loss vs epoch, 3 stages marked)
- **Fig 5.2**: Progressive development — bar chart of BLI across experiments 1.0–1.5
- **Fig 5.3**: Ablation — bar chart of key metrics
- **Fig 5.4**: Learned parameter evolution across 20 layers (KEY FIGURE)
- **Fig 5.5**: Qualitative examples — 4 panels showing easy/medium/hard/failure cases
- **Fig 5.6**: Real chromatogram comparison (LBEADS vs BEADS vs arPLS)
- **Fig 5.7**: Leakage visualization — dense-peak signal showing baseline leakage before/after mitigation

---

## Execution Order

1. First: Create the shared synthetic test set (fixed seed, save to disk)
2. Run experiment 1.0 (classical BEADS) — establishes baseline
3. Run experiments 1.1–1.5 (progressive development) — each trains a model
4. Run ablations A–F — each trains a model
5. Run classical comparisons — arPLS, airPLS, SNIP, AsLS
6. Collect all results → generate tables and figures
7. THEN write Chapters 4 and 5