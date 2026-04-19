# PROMPT: Write Chapter 2 — Background & Related Work (12–15 pages)

## ROLE
You are an expert academic writer producing a chapter for an NYU Tandon MS thesis. Write in formal academic English. LaTeX output. This chapter must give the reader everything they need to understand Chapter 3 (LBEADS-NET method).

## OUTPUT
Produce the full LaTeX content for `background/background.tex`. Begin with `\chapter{Background and Related Work}`. Contains Sections 2.1–2.4.

## SOURCE FILES TO READ

**Primary — read these fully:**
- `Implementations/0. BEADS/WithNumpy/beads.py` — the classical BEADS algorithm implementation (NumPy). Study the functions: `beads()`, filter construction, penalty functions, iterative update loop. Extract the exact mathematical operations for Section 2.2.1.
- `Literature/Research Papers/BEADS.pdf` — original Ning, Selesnick & Duval (2014) paper. Use for mathematical formulation, variable definitions, and citation.
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Section 1 ("Why BEADS leaks") for the three failure modes to include in 2.2.1. Section 4 for arPLS comparison. Section 7 for neural approaches list.
- `Writeup/Guidelines/baseline-leakage/resources.md` — complete reference list organized by category (foundational, unrolling, DEQ, neural baseline, architectures, NMF, curriculum learning, code)
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 2 specification

**For mathematical background (skim relevant sections):**
- `Literature/Custom Notes/ISTA_ADMM/Document.pdf` — ISTA and ADMM derivations (for Section 2.3)
- `Literature/Custom Notes/TVD/Document.pdf` — Total Variation Denoising (for TV penalty context)
- `Literature/Custom Notes/LASSO_L1_RLS/Document.pdf` — L1 norm and reweighted least squares (for sparsity context)
- `Literature/From DSP-II Lectures/2_IntroductiontoSparsity/sparse_SP_intro.pdf` — sparsity fundamentals
- `Optimization Notes/ADMM.pdf` — ADMM theory

**For unrolling context:**
- `Implementations/1. LBEADS_NETv1/lbeads_net.py` — scan to see how BEADS maps to neural network layers (the BEADSLayer class, LBEADS_NET vs LBEADS_NET_Fast variants). This is background for Section 2.3 (algorithm unrolling).

## SECTION SPECIFICATIONS

### Section 2.1: Chromatogram Signal Processing (3–4 pages)

**Signal model** — Present the core decomposition formally:
```
y = x + f + w
```
where y ∈ ℝᴺ is the observed chromatogram, x ∈ ℝᴺ is the sparse peak component, f ∈ ℝᴺ is the smooth baseline, w ∈ ℝᴺ is additive noise. This decomposition is the foundation for the entire thesis.

**Physical causes of baseline drift** — Explain each briefly:
- Column bleed (stationary phase degradation during temperature programming)
- Detector aging (sensitivity drift over instrument lifetime)
- Temperature gradients (oven ramps in gradient elution)
- Mobile phase composition changes (in gradient HPLC, changing solvent ratios)
- Electronic drift (amplifier drift, ADC offsets)

**Why baseline matters for quantification:**
- Peak area ∝ analyte concentration (the fundamental quantitative relationship)
- Incorrect baseline → incorrect peak areas → incorrect concentrations
- Limit of detection determined by baseline noise level
- Calibration curves systematically biased by baseline error

**Fig 2.1**: Annotated chromatogram diagram showing peaks (with labels), baseline, and noise. Reference output from `Implementations/7. LBEADS_NETv7 [Stronger]/demo.py` as the source for generating this.

### Section 2.2: Classical Baseline Correction Methods (5–7 pages)

#### 2.2.1: BEADS — Baseline Estimation And Denoising using Sparsity (3–5 pages)
**THIS IS THE CORE — full mathematical treatment required. The reader must understand every component because Chapter 3 unrolls this algorithm layer by layer.**

Cover in this order:

1. **The optimization problem** — Write the full BEADS cost function:
   ```
   minimize  ||Dx||₁ + λ₀||x||₁ + (λ₁/2)||Hf||₂² + (λ₂/2)||f||₂²
   subject to  y = x + f
   ```
   Define every symbol. Reference the implementation in `0. BEADS/WithNumpy/beads.py`.

2. **MM / variable splitting** — High-level derivation of the iterative reweighting scheme. The L1 norm is non-smooth; BEADS uses Majorization-Minimization with penalty approximation: w(z) = 1/(|z| + ε). Build the diagonal weight matrices Λ = diag(w(D@x)) and Γ = diag(1/(|x|+ε)).

3. **BAfilt construction** — The band-pass filter via cascaded Butterworth sections:
   - scipy.signal.butter(d, fc) computes filter coefficients
   - Coefficients → banded Toeplitz matrices A and B
   - Role of filter order d (1 or 2) and cutoff frequency fc
   - B is the band-pass filter matrix, A is normalizing

4. **The iterative update scheme:**
   - f-update: (B^TB + λ₁H^TH + λ₂I)f = B^T(y - x) — banded linear system, O(N) solve
   - x-update: M = 2λ₀Γ + D'ΛD; x = A·inv(B^TB + A'MA)·d_vec — shrinkage with asymmetric penalty
   - Θ weighting matrix and how r controls asymmetry (negative peaks penalized more)

5. **Parameter table** — Create Table 2.1:
   | Parameter | Controls | Typical Range |
   | λ₀ | Peak sparsity (L1 weight) | 0.001–1.0 |
   | λ₁ | Baseline smoothness (1st derivative) | 0.01–10 |
   | λ₂ | Baseline fidelity (2nd derivative) | 0.01–10 |
   | r | Asymmetry ratio | 1–10 |
   | fc | Cutoff frequency | 0.001–0.1 |
   | d | Filter order | 1 or 2 |

6. **Three failure modes in dense-peak regions** (from `peaks-leaking-into-baseline.md` Section 1):
   - Sparsity assumption violation: L1 over-shrinks dense peaks → surplus becomes baseline
   - Low-pass filter bandwidth mismatch: overlapping peak tails look low-frequency
   - Uniform regularization: same λ everywhere, no spatial adaptivity

7. **Why parameters are hard to tune** — Different instruments/methods need different values; parameters interact nonlinearly; no principled way to set from signal properties alone. This motivates the entire thesis.

**Fig 2.2**: BEADS iteration block diagram showing the f-update → x-update → reweight cycle.

#### 2.2.2: Other Classical Methods (1–2 pages)
Brief coverage, positioning relative to BEADS:
- **arPLS** (Baek et al., 2015): asymmetric reweighted PLS. No sparsity assumption → fails via "elevation" not "leakage" in dense regions. Better than BEADS in dense-peak regions.
- **airPLS** (Zhang et al., 2010): adaptive iteratively reweighted PLS. Three known failure modes in complex regions.
- **SNIP** (Ryan et al., 1988): statistics-sensitive non-linear iterative peak-clipping. Window-based, single parameter.
- **AsLS** (Eilers, 2003): asymmetric least squares. Foundational method.
- **Rubberband**: connect minima with straight lines — simple but crude.
- **Why BEADS is the best unrolling candidate**: explicit optimization (not heuristic), clear per-iteration structure (f-update → x-update → reweight), interpretable parameters, rich enough to benefit from learning.

### Section 2.3: Algorithm Unrolling (4–5 pages)

**Core concept:**
- Take iterative algorithm running K iterations → "unroll" as K sequential network layers
- Algorithm parameters (step sizes, regularization weights) → learnable via backprop
- Architecture encodes algorithmic structure — NOT a black-box

**LISTA lineage:**
- Gregor & LeCun (2010): Learned ISTA — the founding work
- Each layer = one ISTA step with learnable dictionary and threshold
- Unrolled network converges faster than original ISTA with same K

**General framework:**
```
Iterative:  x^{k+1} = T(x^k; θ)
Unrolled:   Layer k: x^{k+1} = T(x^k; θ_k)  where θ_k LEARNABLE per layer
```

**Why it works** — 4 advantages over black-box DL:
1. Interpretable intermediate representations (layer k output = k-iteration estimate)
2. Parameter efficient (only learn algorithm parameters, not arbitrary weights)
3. Domain knowledge baked in (better inductive bias)
4. Analyzable (can compare learned parameters to theory)

**Examples beyond LISTA:**
- ALISTA (Liu et al., ICLR 2019): analytic weights
- ISTA-Net+ (Zhang & Ghanem, CVPR 2018): learned transforms
- Hybrid ISTA (Zheng et al., TPAMI 2022): convergence guarantees
- ADMM-Net (Yang et al., NeurIPS 2016): unrolled ADMM for MRI
- DeMUN (Entropy, 2025): comprehensive examination of unrolled networks

**Validation from literature:**
- Gharbi et al. (MLSP 2024, Signal Processing 2024): compared unrolled primal-dual, ISTA, and Half-Quadratic for 1D chromatographic restoration. Found unrolled HQ underestimates peak intensities → confirms leakage is fundamental to unrolled sparse-recovery networks. This is the most directly relevant published work.
- DIRAS+ (Analytical Chemistry, 2025): explicitly calls out baseline leakage as a fundamental limitation of end-to-end deep learning.

**How this applies to BEADS:**
- Each BEADS iteration → one BEADSLayer
- Learnable per layer: λ₀, λ₁, λ₂, r (log-parameterized for positivity)
- Fixed: fc (not learnable — outside autograd, discussed in Ch. 3.5)
- K=8 layers total

**Fig 2.3**: Generic algorithm unrolling concept diagram.

### Section 2.4: Neural Approaches to Baseline Correction (1–2 pages)
Brief survey of recent neural methods — position LBEADS-NET relative to them:
- Kensert et al. (J. Chromatography A, 2021): 1D conv autoencoder trained on 190K synthetic chromatograms
- Chen et al. (Analyst, 2022): ResNet+UNet for Raman baseline correction
- CAE+ (Han et al., Sensors, 2024): convolutional autoencoder with comparison function. Peak preservation 0.851–0.96. airPLS shows 2–3× higher error in peak regions.
- 1dTrans (Zhao et al., Spectrochimica Acta, 2025): first Transformer for baseline estimation
- RSPSSL (Hu et al., Light: Science & Applications, 2024): self-supervised Raman processing
- DIRAS+ (Analytical Chemistry, 2025): physics-aware with ML-predicted parameters
- **LBEADS-NET positioning**: combines interpretability of classical (BEADS structure preserved) with adaptivity of neural (parameters learned from data). Unlike pure neural approaches, intermediate representations are interpretable as BEADS iteration outputs.

## STYLE CONSTRAINTS
- This is the longest chapter — be thorough but not repetitive
- All mathematical notation must be consistent with what Chapter 3 uses
- Use `\newcommand` definitions from `definitions.tex`: \vx (peaks), \vf (baseline), \vy (observed), \mB (BAfilt), etc.
- Every claim about a method needs a citation
- The BEADS section (2.2.1) should be detailed enough that a reader could implement BEADS from it
- Use `\cite{}` with keys: `\cite{ning2014beads}`, `\cite{baek2015arpls}`, `\cite{zhang2010airpls}`, `\cite{gregor2010lista}`, `\cite{monga2021unrolling}`, `\cite{gharbi2024unrolled}`, `\cite{zhang2018istanet}`, etc.
