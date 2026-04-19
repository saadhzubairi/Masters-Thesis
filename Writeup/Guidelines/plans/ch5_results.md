# Chapter 5 — Results & Analysis (12–15 pages)

File: `results/results.tex` (CREATE this directory + files)

---

## Section 5.1: Synthetic Data Results (6–8 pages)

### 5.1.1: Convergence and Training Dynamics (2 pages)

#### What to write
- Loss curves across three training stages (A→B→C)
- Show how total loss evolves and where stage transitions occur
- Loss decomposition: contribution of each term over training
  - MSE dominates in Stage A
  - Baseline and asymmetric losses activate in Stage B — brief spike then decrease
  - Full loss activates in Stage C — initial increase then stabilization
- Stage transition behavior: how the loss landscape changes when new terms are added
- Convergence patterns: does the model converge within each stage?

#### Key details from your work
- Training typically runs 30 epochs (5+15+10)
- MLflow logs per-epoch losses (check mlruns/ directory)
- Training logs available in Implementations/7. LBEADS_NETv7 [Stronger]/train/
- Multiple training runs available from different experiments

#### Figures
- **Fig 5.1**: Training convergence curves — total loss vs epoch, with stage boundaries marked
- **Fig 5.2**: Loss decomposition — stacked area or multi-line plot showing each loss term's contribution

---

### 5.1.2: Learned Parameter Evolution (2–3 pages)

#### THIS IS A KEY RESULT — UNIQUE TO ALGORITHM UNROLLING

#### What to write
- How lambda_0, lambda_1, lambda_2, r values differ across layers 1→K (K=8)
- Do parameters specialize?
  - Early layers (1-3): expect coarser regularization (larger lambdas, more aggressive)
  - Middle layers (4-6): transitional behavior
  - Late layers (7-8): expect finer regularization (smaller lambdas, precise)
- Compare learned parameters to manually-tuned BEADS parameters:
  - BEADS: same parameters for all iterations
  - LBEADS: different parameters per layer → richer optimization
- Discuss parameter interpretation:
  - What does it mean when lambda_0 varies across layers?
  - Does r change? (Should it — asymmetry may matter more in early vs late iterations)

#### Key details from your work (from v6 diagnostics)
```
Learned parameters across 5 layers (v6 data):
  lam0: 0.00194 – 0.00208  (very small — weak sparsity, allows peaks)
  lam1: 0.285 – 0.398      (moderate smoothness variation)
  lam2: 0.226 – 0.315      (moderate fidelity variation)
  r:    5.83 – 6.23         (close to default 6.0, slight variation)
  step: ~0.986              (near 1.0)
  output_gain: ~1.18        (slight amplification)
```

- v7 uses 8 layers instead of 5 — need to extract updated parameter profiles
- Parameter values logged during training and available in checkpoint .pth files

#### Figures
- **Fig 5.3**: Learned parameter evolution across layers — bar chart or line plot for each parameter (lambda_0, lambda_1, lambda_2, r) across layers 1-8. This is a KEY FIGURE for the thesis.

#### Why this matters
- Demonstrates that unrolling adds value over fixed-parameter iteration
- Layer specialization is an EMERGENT property — not designed, but learned
- Validates the algorithm unrolling approach: the network discovers that different iteration stages need different parameters

---

### 5.1.3: Quantitative Comparison (2 pages)

#### What to write
- Full comparison table: LBEADS-NET vs classical BEADS vs pybaselines methods
- Metrics: MSE, MAE, correlation, peak position MAE, area error
- Report: mean +/- std over test set (40 samples)
- Breakdown by difficulty:
  - By SNR: how does performance degrade with more noise?
  - By peak density: single peak vs many overlapping peaks
  - By baseline type: polynomial vs sinusoidal vs spline

#### What you need to generate
- Run evaluation on test set with all comparison methods
- Use demo.py or write evaluation script
- Compare: LBEADS-NET raw, LBEADS-NET hybrid, classical BEADS (tuned), arPLS, airPLS, SNIP, AsLS

#### Tables
- **Table 5.1**: Full metrics comparison — main results table

| Method | Peak MSE | Peak MAE | Corr | Baseline MSE | Area Error |
|--------|----------|----------|------|--------------|------------|
| LBEADS-NET | | | | | |
| LBEADS-NET (hybrid) | | | | | |
| BEADS (tuned) | | | | | |
| arPLS | | | | | |
| airPLS | | | | | |
| SNIP | | | | | |
| AsLS | | | | | |

---

### 5.1.4: Qualitative Examples (2 pages)

#### What to write
- 3–4 carefully selected example signals showing:
  - Input signal y
  - Ground truth peaks x_true and baseline f_true
  - LBEADS-NET output (x_hat, f_hat)
  - Classical BEADS output
  - Best pybaselines output
- Choose examples strategically:
  - Example 1: Easy case — well-separated peaks, moderate drift → show LBEADS works well
  - Example 2: Success case — complex baseline, LBEADS outperforms classical
  - Example 3: Partial success — dense peaks, some leakage but manageable
  - Example 4: Failure case — extreme density or drift, LBEADS struggles → honest assessment

#### Figures
- **Fig 5.4**: Multi-panel comparison figure (4 rows × 3-4 columns)
  - Generate from demo.py or evaluation script

---

## Section 5.2: Real Chromatogram Results (4–5 pages)

### 5.2.1: Chromatogram Demo Results (2 pages)

#### What to write
- Apply LBEADS-NET to real chromatogram data (from BEADS paper dataset)
- Visual comparison: before/after baseline correction
- Side-by-side: LBEADS-NET vs classical BEADS
- Discuss qualitative differences:
  - Does LBEADS preserve peak shapes better?
  - Is the baseline smoother?
  - Are there artifacts?
- NOTE: no ground truth for real data — assessment is qualitative + quality metrics

#### Key details from your work
- `demo_chromatogram.py` processes real chromatogram data
- Real signals are longer (N >> 1024) — may need sliding window or padding
- Train/inference length mismatch is relevant here (discuss in Ch. 6)

#### Figures
- **Fig 5.5**: Real chromatogram comparison — panels showing raw signal, LBEADS output, BEADS output

---

### 5.2.2: Hybrid Pipeline Performance (1–2 pages)

#### What to write
- Compare: LBEADS alone vs LBEADS + denoising vs LBEADS + classical refinement vs full hybrid
- How stage selection via quality scoring works in practice
- Which stage gets selected most often and why

#### Key details from your work (from v6 diagnostics)
```
Hybrid diagnostics (6 test samples):
  Selected stage: "short_refine" for ALL samples
  baseline_hf_ratio: 0.048 – 0.123
  residual_hf_rms: 0.011 – 0.021
  Correlation with ground truth: 0.73 – 0.86

Energy comparisons:
  x_lbeads << x_true (network undershoots)
  x_refine ≈ x_lbeads + correction
  x_hybrid = x_refine (selects refined path)
```

- Key finding: classical refinement is always selected → LBEADS alone doesn't fully converge
- But LBEADS provides a good initialization → refinement converges faster

#### Tables & Figures
- **Table 5.3**: Hybrid vs standalone performance comparison
- **Fig 5.6**: Hybrid pipeline comparison — same signal processed with LBEADS-only vs hybrid

---

### 5.2.3: Where It Works and Where It Doesn't (1 page)

#### What to write
- Success cases: signals with moderate drift, well-separated peaks, moderate noise
- Partial success: dense-peak regions show some leakage but manageable
- Failure cases: extreme drift, very dense overlapping peaks, very low SNR
- This section transitions naturally to Chapter 6 (Discussion)

---

## Section 5.3: Additional Datasets (if applicable, 2–3 pages)

#### What to write — only if you have results
- RRUFF mineral spectra: Raman spectroscopy baseline correction
- MIT-BIH data: ECG baseline wander correction
- Cross-domain generalization: does LBEADS trained on chromatograms transfer?
- If not done yet, mention as future work in Ch. 7

---

## Overall Successes to Highlight
- Learned parameter specialization across layers (the KEY result)
- Competitive with manually-tuned classical BEADS on synthetic data
- Hybrid pipeline provides practical robustness for real data
- Quality scoring enables automatic method selection

## Honest Failures to Acknowledge
- LBEADS alone doesn't fully converge — needs hybrid refinement
- Network undershoots peak amplitudes (x_lbeads << x_true)
- Classical refinement selected for all test samples — LBEADS isn't self-sufficient
- Dense-peak regions still show leakage despite all the mitigation attempts
- Real data performance gap compared to synthetic (distribution shift)
