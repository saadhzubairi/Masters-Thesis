# PROMPT: Write Chapter 5 — Results & Analysis (12–15 pages)

## ROLE
You are an expert academic writer producing the results chapter of an NYU Tandon MS thesis. Present results objectively. Use data from the source files. Include both successes AND honest failures. LaTeX output.

## OUTPUT
Produce the full LaTeX content for `results/results.tex`. Begin with `\chapter{Results and Analysis}`. Contains Sections 5.1–5.2 (and optionally 5.3).

## SOURCE FILES TO READ

**Critical — read for actual data:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/train/` — training logs, loss curves per epoch. Look for any saved metrics files.
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net_baseline_fix_1773526203.pth` — latest checkpoint. Load and extract learned parameter values (log_lam0, log_lam1, log_lam2, log_r per layer) for Section 5.1.2. Use torch.load() to inspect.
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo/` — demo output images and data
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo-chrom/` — chromatogram demo output
- `Implementations/7. LBEADS_NETv7 [Stronger]/mlruns/` — MLflow experiment data (metrics per epoch)
- `Implementations/7. LBEADS_NETv7 [Stronger]/analysis/` — analysis results

**For v6 diagnostics (actual numbers):**
- `Implementations/6. LBEADS_NETv6 [Strong]/analysis/` — hybrid_diagnostics.txt and hybrid_diagnostics.json contain:
  - Learned parameters per layer (lam0: 0.00194–0.00208, lam1: 0.285–0.398, lam2: 0.226–0.315, r: 5.83–6.23)
  - Quality metrics per sample (baseline_hf_ratio: 0.048–0.123, residual_hf_rms: 0.011–0.021)
  - Correlation with ground truth: 0.73–0.86
  - Energy comparisons (x_lbeads, x_post, x_refine, x_hybrid)
  - Selected inference stage ("short_refine" for ALL samples)

**For comparison:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo.py` — evaluation code, how metrics are computed
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo_chromatogram.py` — real data processing

**For context on what to report:**
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Section 7: evaluation metrics to use (BLI, stratified evaluation, peak area preservation)
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 5 specification

## SECTION SPECIFICATIONS

### Section 5.1: Synthetic Data Results (6–8 pages)

#### 5.1.1: Convergence and Training Dynamics (2 pages)

**Extract from training logs / MLflow:**
- Loss curves across stages A→B→C. Mark stage transitions clearly.
- Loss decomposition: show how each loss term contributes over training:
  - Stage A: MSE dominates
  - Stage B transition: brief spike when baseline/asymmetric losses activate, then decrease
  - Stage C: full loss with all terms active
- Convergence: does the model converge within each stage? Any instabilities?

**Fig 5.1**: Training convergence curves (total loss vs epoch, stage boundaries marked)
**Fig 5.2**: Loss decomposition (multi-line or stacked area showing each term)

If exact training log data is not available in the files, describe the expected behavior based on the training configuration and note which output files contain the raw data.

#### 5.1.2: Learned Parameter Evolution (2–3 pages) — THE KEY RESULT

**THIS IS THE MOST IMPORTANT RESULT IN THE THESIS. It demonstrates that algorithm unrolling adds value over fixed-parameter BEADS.**

**Extract actual parameter values from checkpoint.** Load `lbeads_net_baseline_fix_1773526203.pth` and report:
- λ₀ per layer (layers 1–8): from exp(log_lam0) for each BEADSLayer
- λ₁ per layer: from exp(log_lam1)
- λ₂ per layer: from exp(log_lam2)
- r per layer: from exp(log_r)
- step_size per layer
- output_gain

**If v7 checkpoint extraction is not possible, use v6 data** (from diagnostics):
```
lam0: 0.00194–0.00208 (very weak sparsity — allows peaks)
lam1: 0.285–0.398 (moderate smoothness with variation across layers)
lam2: 0.226–0.315 (moderate fidelity with variation)
r: 5.83–6.23 (close to default 6.0, slight layer variation)
step_size: ~0.986 (near 1.0)
output_gain: ~1.18 (slight amplification)
```

**Analysis — write about:**
1. **Do parameters specialize?** YES — show variation across layers. Early layers may use different λ values than late layers. Discuss the pattern: early = coarse separation, late = fine refinement.
2. **Compare to classical BEADS**: classical uses SAME parameters for ALL iterations. LBEADS-NET uses DIFFERENT parameters per layer. This is a richer optimization.
3. **Interpretation**: What does it mean that λ₀ is very small across all layers? (The network has found that strong sparsity penalty is unnecessary when the baseline is separately supervised.) What does r≈6 mean? (The learned asymmetry is close to the default, suggesting this is near-optimal.)
4. **This is emergent**: parameter specialization was NOT designed — it emerged from end-to-end training.

**Fig 5.3**: Learned parameter evolution per layer — bar chart or grouped bar with one group per layer, showing λ₀, λ₁, λ₂, r values. THIS IS THE KEY FIGURE.

#### 5.1.3: Quantitative Comparison (2 pages)

**Table 5.1**: Full metrics comparison — the main results table:

| Method | Peak MSE | Peak MAE | Corr | Baseline MSE | Area Error | BLI |
| LBEADS-NET (raw) | | | | | | |
| LBEADS-NET (hybrid) | | | | | | |
| BEADS (tuned) | | | | | | |
| arPLS | | | | | | |
| airPLS | | | | | | |
| SNIP | | | | | | |
| AsLS | | | | | | |

**Extract numbers from evaluation scripts or generate them.** If exact comparison data is not yet available, describe the table structure and note which scripts (`demo.py`) generate these numbers. Report mean ± std over 40 test samples.

**Additional analysis:**
- Breakdown by signal difficulty (if data available): easy (few peaks, high SNR) vs hard (many peaks, low SNR)
- Statistical significance: note whether differences are meaningful given test set size

**Table 5.2**: Stratified metrics (sparse vs dense regions) — report BLI and peak-region RMSE separately for sparse and dense regions.

#### 5.1.4: Qualitative Examples (2 pages)

**Choose 4 examples strategically:**
1. **Easy success**: well-separated peaks, moderate drift → LBEADS works well
2. **Complex success**: difficult baseline, LBEADS outperforms classical BEADS
3. **Partial success**: dense peaks, some leakage visible but manageable
4. **Failure case**: extreme density or drift, LBEADS struggles → honest assessment

**Fig 5.4**: Multi-panel comparison figure (4 rows). Each row shows:
- Input signal y
- Ground truth x_true and f_true
- LBEADS-NET output (x̂, f̂)
- Classical BEADS output

Reference `demo.py` output images or describe what to generate.

### Section 5.2: Real Chromatogram Results (4–5 pages)

#### 5.2.1: Chromatogram Demo Results (2 pages)
- Apply LBEADS-NET to real chromatogram data (from BEADS paper dataset)
- No ground truth → qualitative assessment + quality metrics
- Visual comparison: LBEADS vs classical BEADS side-by-side
- Discuss: peak shape preservation, baseline smoothness, artifacts
- **Fig 5.5**: Real chromatogram comparison panels
- Reference output from `demo_chromatogram.py` and `demo-chrom/` directory

#### 5.2.2: Hybrid Pipeline Performance (1–2 pages)

**Use actual data from v6 diagnostics:**
- Compare stages: LBEADS alone vs denoised vs refined vs hybrid
- "short_refine" selected for ALL 6 test samples → LBEADS alone doesn't fully converge
- baseline_hf_ratio: 0.048–0.123
- residual_hf_rms: 0.011–0.021
- Correlation: 0.73–0.86
- Energy: x_lbeads << x_true (network undershoots), x_refine corrects

**Key finding**: classical refinement is always selected. LBEADS-NET alone is insufficient BUT provides a good initialization → hybrid converges faster than classical BEADS from scratch.

**Table 5.3**: Hybrid vs standalone — quality metrics for each pipeline stage.
**Fig 5.6**: Same signal processed LBEADS-only vs hybrid.

#### 5.2.3: Where It Works and Where It Doesn't (1 page)
- **Success**: moderate drift, well-separated peaks, moderate noise
- **Partial**: dense-peak regions show some leakage (manageable)
- **Failure**: extreme drift, very dense overlapping peaks, very low SNR
- This transitions to Chapter 6 naturally

## STYLE CONSTRAINTS
- Present results objectively — let the data speak
- Include BOTH successes and failures (the thesis is stronger for honesty)
- All numbers must come from actual experiments or be clearly marked as "to be generated using [script]"
- Figures and tables are essential — this is a results chapter
- Cross-reference Ch. 3 for method and Ch. 4 for setup
- Don't analyze limitations deeply here — save that for Ch. 6
