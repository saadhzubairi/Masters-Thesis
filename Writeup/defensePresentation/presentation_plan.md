# LBEADS-NET Thesis Defense Presentation — Design

## Format
- **Tool**: LaTeX Beamer
- **Duration**: 20–25 minutes (leaves 30–40 min for Q&A)
- **Slides**: 28
- **Structure**: "The Three Acts" — narrative arc
- **Math level**: Key equations only (BEADS cost, ISTA update, soft-thresholding bias)

## Structure

### Act I: "The Problem" (5 min, Slides 1–8)
Establishes chromatography, baseline drift, BEADS and its tuning burden, the DL alternative, the interpretability–generalization gap, and introduces algorithm unrolling.

| Slide | Title | Key Content |
|-------|-------|-------------|
| 1 | Title | LBEADS-NET, name, NYU Tandon, Selesnick, date |
| 2 | What is a Chromatogram? | Annotated signal: y = x + f + w |
| 3 | Why Baseline Correction Matters | Correct vs wrong baseline → concentration error |
| 4 | The Classical Solution: BEADS | BEADS cost function (Eq 2.1), 6 params |
| 5 | The Tuning Problem | Same signal, 3 parameter sets → different results |
| 6 | The Deep Learning Alternative | Kensert, ResUNet, DIRAS+ — generalize but black-box |
| 7 | The Gap | 2×2 matrix: interpretable × generalizable |
| 8 | Our Approach: Algorithm Unrolling | Iteration → layer, parameters → learnable |

### Act II: "The Attempt" (10 min, Slides 9–21)
Architecture (3 min) then the development timeline (7 min).

| Slide | Title | Key Content |
|-------|-------|-------------|
| 9 | From BEADS to BEADSLayer | MM iteration vs ISTA layer side-by-side |
| 10 | The ISTA Layer | Gradient step + asymmetric soft-thresholding equations |
| 11 | Why Not Conjugate Gradient? | Depth 96 vs 20, gradient vanishing visual |
| 12 | Full Architecture | Pipeline diagram, 5 params × 20 layers = 100 |
| 13 | Timeline Overview | Horizontal timeline Exp 1.0→1.5 |
| 14 | Exp 1.0: Classical BEADS | Default params, 99.99% area error |
| 15 | Exp 1.1: Naive Unrolling | MSE only, 2× improvement, BLI=0.593 |
| 16 | Exp 1.2: + Sparsity Priors | ℓ₁+TV+non-neg, shapes improve, BLI unchanged |
| 17 | Exp 1.3: + Baseline Supervision | Best peak correlation (0.793), most impactful |
| 18 | Exp 1.4: + Orthogonality (v5) | Best baseline MSE, thesis model |
| 19 | Exp 1.5: Full v7 | All 12 terms → performance DEGRADES |
| 20 | Over-Regularization Lesson | Complexity vs performance curve |
| 21 | The BLI Plateau | Bar chart: BLI ≈ 0.59 across all configs |

### Act III: "The Discovery" (5 min, Slides 22–28)
Why leakage is fundamental, positive findings, limitations, future work.

| Slide | Title | Key Content |
|-------|-------|-------------|
| 22 | Why Leakage is Fundamental | Soft-thresholding shrinkage bias equation |
| 23 | Three Root Causes | Sparsity violation, fixed f_c, uniform λ |
| 24 | Emergent Parameter Specialization | λ₀ increasing, r non-monotonic, η stable |
| 25 | What Actually Worked | 5 ranked findings |
| 26 | Limitations | Honest: synthetic-only, no baselines, O(N²), 50% area error |
| 27 | Future Directions | Group sparsity, banded ops, real data |
| 28 | Thank You | Summary one-liner, questions |
