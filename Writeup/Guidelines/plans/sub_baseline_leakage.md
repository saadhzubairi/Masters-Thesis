# Sub-Plan: Baseline Leakage Problem — Deep Dive

**Source material**: `Writeup/Guidelines/baseline-leakage/`
- `peaks-leaking-into-baseline.md` — 567-line engineering briefing (problem analysis, techniques, code sketches, implementation phases)
- `resources.md` — 74 curated references across 8 categories

This was a central research effort in your thesis. The material is rich enough to significantly strengthen **multiple chapters**, not just one section. Below is how to distribute it.

---

## Where This Content Goes in the Thesis

| Thesis Location | What Goes There | From Leakage Doc |
|-----------------|-----------------|------------------|
| **Ch. 2.2.1** (BEADS detail) | Why BEADS leaks — the 3 failure modes | Section 1 (sparsity violation, fc mismatch, uniform regularization) |
| **Ch. 2.2.2** (Other methods) | How arPLS/SNIP handle dense peaks better | Section 4 (comparative ranking) |
| **Ch. 2.3** (Algorithm unrolling) | Gharbi et al. confirming leakage is fundamental to unrolled sparse-recovery | Section 1 (MLSP 2024 reference) |
| **Ch. 3.7** (Composite loss) | All loss term formulations | Section 2 (all 8 loss terms with math) |
| **Ch. 3.3** (ISTA variant) | Learned proximal operators as alternative | Section 3 (LearnedProximal, ISTA-Net+) |
| **Ch. 3.8** (Hybrid pipeline) | Wave-U-Net difference output layer idea | Section 5 (peaks = signal - baseline) |
| **Ch. 5** (Results) | BLI metric, stratified evaluation | Section 7 (evaluation metrics) |
| **Ch. 6.1** (Dense-peak leakage) | THE core discussion section | Sections 1, 2, 4 (the full analysis) |
| **Ch. 7.3** (Future work) | Conformer, Mamba, DEQ, signal-adaptive params | Sections 3, 5 (alternative architectures) |

---

## Detailed Content Plan

### For Chapter 2 — Background: Root Cause Analysis

**Add to Section 2.2.1 (BEADS detail)** — 0.5–1 page

Three mechanistic failure modes of BEADS in dense-peak regions:

1. **Sparsity assumption violation**
   - L1 penalty on x and derivatives assumes peaks are sparse
   - Dense peaks (metabolomics, complex mixtures) → composite signal is NOT sparse
   - L1 over-shrinks peaks → surplus energy attributed to baseline
   - This is inherent to the BEADS formulation, not an implementation bug

2. **Low-pass filter bandwidth mismatch**
   - fc determines what counts as "baseline"
   - Overlapping peak tails → sustained elevation → looks low-frequency to the filter
   - Filter interprets overlapping peak tails as baseline content
   - Too-high fc → direct peak leakage; too-low fc → misses real baseline variation
   - Fundamental: no single fc handles both sparse and dense regions

3. **Uniform regularization**
   - Classical BEADS applies same lambda_0, lambda_1, lambda_2 everywhere
   - Dense-peak regions need DIFFERENT regularization than sparse regions
   - No mechanism to adapt spatially → one-size-fits-all compromise

**Add to Section 2.2.2 (Other methods)** — a paragraph

Key insight: arPLS has NO sparsity assumption. It fits a smooth curve below data with adaptive weighting. In dense regions:
- arPLS failure mode = "elevation" (baseline too high, but smooth)
- BEADS failure mode = "leakage" (baseline follows peak shapes)
- Elevation is less damaging than leakage for quantification
- This insight motivates LBEADS learning to behave like arPLS in dense regions and like BEADS in sparse regions

**Add to Section 2.3 (Algorithm unrolling)** — a paragraph

Gharbi, Chouzenoux, Pesquet & Duval (MLSP 2024, Signal Processing 2024):
- Compared unrolled primal-dual, unrolled ISTA, unrolled Half-Quadratic for 1D chromatographic signal restoration
- Found unrolled HQ **underestimates peak intensities** (measured by TSNR)
- Confirms leakage is **fundamental to the class** of unrolled sparse-recovery networks, not specific to LBEADS-NET
- DIRAS+ (Analytical Chemistry, 2025) explicitly called out baseline leakage as a fundamental limitation of end-to-end deep learning

This is important: it validates that your problem is real and recognized by the community.

---

### For Chapter 3 — Method: What You Built to Fight Leakage

The loss functions in Section 2 of the leakage doc map directly to Ch. 3.7 (already planned). But the leakage doc adds richer detail:

**Enhance Section 3.7.2 (Sparsity Penalties)** — add context

The leakage doc introduces a **gradient-overlap penalty** not in the current plan:
```
L_grad_overlap = ||nabla(x_peak) odot nabla(x_base)||_1
```
- Penalizes regions where BOTH components have significant gradients simultaneously
- Targets exactly the boundary regions where leakage occurs
- lambda_grad = 0.01–0.1
- Note whether you implemented this or decided against it (and why)

**Enhance Section 3.7.4 (Smoothness)** — add locally adaptive variant

The leakage doc proposes locally adaptive smoothness:
```python
weights = 1.0 + 10.0 * |x_peak_estimate|.detach()
L_adaptive_smooth = (weights * D2_baseline^2).mean()
```
- Increase smoothness constraint where peaks are detected
- `.detach()` on peak estimate prevents degenerate solution (zeroing peaks to reduce smoothness cost)
- Note: this is a subtlety worth discussing — gradient-stopping to prevent mode collapse

**Enhance Section 3.7.9 (Envelope Constraint)** — add proxy for real data

When ground truth is unavailable:
```python
residual = baseline_pred - soft_local_min(y, window)
```
using differentiable soft minimum via log-sum-exp:
```python
soft_min = -tau * logsumexp(-y_unfold / tau, dim=1)
```
- Derived from AsLS family (Eilers, 2005) and arPLS (Baek et al., 2015)
- Important for real-data inference where f_true is unknown

**New content for Section 3.3** — mention learned proximal operators

The leakage doc's LearnedProximal (3-layer 1D CNN replacing soft-thresholding):
- Residual structure: output = soft_threshold(x) + CNN(x)
- Follows ISTA-Net+ (Zhang & Ghanem, CVPR 2018)
- Note whether you implemented this or kept it as future work
- If not implemented: explain why (added complexity, convergence concerns, sufficient improvement from loss terms alone)

---

### For Chapter 5 — Results: Leakage-Specific Metrics

**Add to Section 5.1.3 (Quantitative Comparison)** — add stratified evaluation

The leakage doc proposes critical evaluation methodology:

**Baseline Leakage Index (BLI)** — a novel metric you researched:
```python
def baseline_leakage_index(baseline_pred, baseline_true, peak_signal, peak_mask):
    residual = baseline_pred - baseline_true  # positive = over-estimation
    leakage = ReLU(residual) * peak_mask      # only over-estimation in peak regions
    return leakage.sum() / (peak_energy + eps)
```
- BLI > 0 indicates baseline absorbing peak energy
- Directly measures the failure mode

**Peak-to-Baseline Error Ratio**:
```
ratio = RMSE(baseline_pred[peak_mask], baseline_true[peak_mask]) / RMSE(baseline_pred[~peak_mask], baseline_true[~peak_mask])
```
- Ratio > 1 indicates worse performance in peak regions = leakage
- CAE+ paper (Sensors, 2024): airPLS shows 2–3x higher error in peak vs baseline regions

**Stratified Evaluation**:
- Classify regions as "sparse" (<1 peak per window) or "dense" (>3 overlapping peaks per window)
- Report ALL metrics separately for sparse vs dense
- Dense-region penalty = metric_dense / metric_sparse - 1 → quantifies degradation

**Additional metrics from leakage doc**:
| Metric | Formula | Purpose |
|--------|---------|---------|
| Peak-region RMSE | RMSE within ±3sigma of each peak | Directly measures leakage |
| Peak area preservation | \|A_pred - A_true\| / A_true × 100% | Quantitative accuracy |
| Peak height preservation | \|H_pred - H_true\| / H_true × 100% | Detection sensitivity |
| L∞ (max error) | max\|b_pred - b_true\| | Worst-case leakage |
| Non-negativity fraction | % of corrected signal < 0 | Should be ~0 + noise floor |
| Residual autocorrelation | Durbin-Watson in peak-free regions | Should be white noise |

---

### For Chapter 6 — Discussion: The Core Leakage Analysis

**This is where the leakage problem gets its deepest treatment.**

Section 6.1 (already planned) should be enriched with:

**The mitigation journey across versions** — tell the story:

| Version | What was tried | Result | Why it wasn't enough |
|---------|---------------|--------|---------------------|
| v3 | Sparsity losses (L1, TV) | Made peaks sparser but didn't fix leakage | L1 is the CAUSE of leakage, not the cure |
| v4 | Baseline supervision (masked MSE) | Major improvement — baseline stops following peaks | Only works where ground truth available; masking imperfect |
| v6 | Gradient-based orthogonality | Modest improvement | Gradient penalty is indirect — penalizes derivatives not values |
| v7 | Asymmetric baseline loss (9:1) | Reduced over-estimation significantly | Doesn't prevent co-activation, just penalizes it asymmetrically |
| v7 | Element-wise orthogonality | Better mutual exclusivity | Leakage persists in transition zones between peaks and baseline |
| v7 | Envelope constraint | Prevents baseline excursions above local min | Local min can be wrong in dense regions (min is still elevated) |
| v7 | Frequency separation | Enforces spectral roles | Dense peaks have low-freq content too — imperfect separation |
| v7 | Softplus constraint | Eliminates negative-peak artifacts | Doesn't address the root cause |

**The fundamental insight** (the thesis-level takeaway):
- All mitigations are **regularizers on top of a sparsity-biased formulation**
- The core BEADS optimization assumes sparsity; no amount of loss engineering fully overcomes this when the assumption is violated
- To truly fix it: need a non-sparsity-based decomposition or signal-adaptive sparsity
- Gharbi et al. (2024) and DIRAS+ (2025) confirm this is a **community-wide problem**, not unique to LBEADS-NET

**What the leakage doc calls "the key insight for LBEADS-NET"**:
- arPLS has no sparsity assumption → doesn't leak in dense regions
- LBEADS-NET should ideally learn to behave like arPLS in dense regions and like BEADS in sparse regions
- Signal-adaptive regularization (predicting lambda from signal features) is the principled fix
- This connects directly to future work

---

### For Chapter 7 — Future Work: What Could Fix It

**Enrich Section 7.3 with concrete future directions from the leakage doc:**

1. **Learned Proximal Operators** (already in plan, now enriched)
   - LearnedProximal: 3-layer 1D CNN with residual structure
   - Follows ISTA-Net+ (Zhang & Ghanem, CVPR 2018)
   - Hybrid ISTA (Zheng et al., TPAMI 2022): maintains convergence guarantees
   - The key advantage: can learn to NOT shrink in dense regions

2. **Signal-Adaptive Parameter Prediction** (already in plan, now enriched)
   - SignalAdaptiveParams: small encoder (Conv1d → GELU → AdaptiveAvgPool → Linear → Softplus)
   - Predicts per-signal adjustments to lambda_0, lambda_1, lambda_2
   - Follows ISTA-Net++ (You et al., IEEE TIP 2021)
   - DIRAS+ uses CNN+XGBoost for per-spectrum lambda prediction
   - This is the most principled fix: dense regions get lower lambda_0 automatically

3. **Conformer-Based Proximal** (new from leakage doc)
   - Conformer = Conv + Self-Attention in sandwich structure
   - Convolution captures local peak shapes
   - Self-attention captures global baseline trends
   - Linear attention (DF-Conformer) for O(N) complexity
   - Reference: Gulati et al. (Interspeech 2020)

4. **ADMM Reformulation** (already in plan, now enriched)
   - ADMM variable splitting: peaks and baseline as separate auxiliary variables
   - ADMM-Net (Yang et al., NeurIPS 2016): piecewise-linear shrinkage more flexible than soft-thresholding
   - Natural separation between data fidelity and regularization substeps = natural peak/baseline separation

5. **Deep Equilibrium Model (DEQ)** (new from leakage doc)
   - Convert LBEADS-NET to DEQ: iterate until convergence per signal
   - Dense-peak signals that need more iterations get them automatically
   - MsDC-DEQ-Net (Yu & Dansereau, 2024): demonstrated for compressed sensing
   - GUDL (2025): DEQ with GSURE for unsupervised sparse recovery
   - O(1) memory via implicit differentiation
   - Recommendation: implement after fixed-stage improvements validated

6. **WaveNet-Style Dilated Convolutions** (new from leakage doc)
   - Exponentially increasing dilation (1, 2, 4, ..., 512)
   - Large receptive field without downsampling → preserves sample-level resolution
   - Gated activation (tanh × sigmoid) acts as learnable filter
   - Won first place in MIT RF Challenge (ICASSP 2024) with learnable dilation

7. **Mamba / State Space Models** (new from leakage doc)
   - Linear O(N) complexity with data-dependent selection
   - No paper has applied Mamba to chromatography baseline correction — novel direction
   - Selection mechanism could learn to identify peak vs baseline regions

8. **Peak-Aware Masked Autoencoder Pre-training** (new from leakage doc)
   - Mask 90% of peak regions + 30% of baseline regions
   - Train MAE to reconstruct → decoder learns to interpolate baseline under peaks
   - Novel approach, not yet published

---

## References to Add to thesis.bib

The `resources.md` file has 40+ references organized by category. Key additions beyond what's already planned:

**Most critical** (directly validate your problem):
- Gharbi et al. (MLSP 2024) — confirms leakage in unrolled sparse-recovery
- DIRAS+ (Analytical Chemistry, 2025) — calls out leakage as fundamental DL limitation
- CAE+ (Han et al., Sensors, 2024) — quantifies peak-region error ratios

**Algorithm unrolling** (strengthen Ch. 2 & 3):
- ISTA-Net+ (Zhang & Ghanem, CVPR 2018)
- ALISTA (Liu et al., ICLR 2019)
- Hybrid ISTA (Zheng et al., TPAMI 2022)
- ADMM-Net (Yang et al., NeurIPS 2016)
- DeMUN (Entropy, 2025)

**Neural baseline correction** (strengthen Ch. 2):
- Kensert et al. (J. Chromatography A, 2021) — 190K synthetic chromatograms
- Chen et al. (Analyst, 2022) — ResNet+UNet for Raman
- 1dTrans (Zhao et al., Spectrochimica Acta, 2025) — first Transformer for baseline
- RSPSSL (Hu et al., Light: Science & Applications, 2024) — self-supervised

**Architectures** (strengthen Ch. 7 future work):
- Conformer (Gulati et al., Interspeech 2020)
- Wave-U-Net (Stoller et al., ISMIR 2018)
- Mamba (Gu & Dao, 2023)
- DEQ (Bai, Kolter & Koltun, NeurIPS 2019)

**Data generation** (strengthen Ch. 4):
- Grushka (Anal. Chem., 1972) — EMG peak model
- Hacohen & Weinshall (ICML 2019) — curriculum learning

---

## Summary: How This Strengthens the Thesis

The baseline leakage research adds:

1. **Depth to Ch. 2**: Three mechanistic failure modes of BEADS, positioned against the broader literature (Gharbi et al. confirming it's a community problem)

2. **Rigor to Ch. 3**: Richer loss term motivation, gradient-stopping subtleties, consideration of alternatives (learned proximal, gradient-overlap)

3. **Better evaluation in Ch. 5**: BLI metric, stratified evaluation, peak-region RMSE — diagnostic metrics that directly measure the failure mode

4. **Analytical strength in Ch. 6**: The mitigation journey (v3→v7) told as a systematic engineering narrative, with the fundamental insight that sparsity-biased formulations have inherent limits in dense regimes

5. **Concrete future work in Ch. 7**: 8 well-researched directions with citations, code sketches, and feasibility assessment

6. **40+ additional references**: Validates your work against 2024-2025 state of the art
