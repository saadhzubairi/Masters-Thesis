# Chapter 2 — Background & Related Work (12–15 pages)

File: `background/background.tex`

---

## Section 2.1: Chromatogram Signal Processing (3–4 pages)

### What to write

**Signal Model (core formulation)**
- y = x + f + w where:
  - y: observed chromatogram (measured detector response)
  - x: sparse peak component (analyte elution responses)
  - f: smooth baseline (low-frequency drift)
  - w: additive noise (detector noise, electronic noise)
- This decomposition is the foundation — everything in the thesis works from this model

**Physical Causes of Baseline Drift**
- Column bleed: stationary phase degradation, especially during temperature programming
- Detector aging: sensitivity changes over instrument lifetime
- Temperature gradients: oven temperature ramps during gradient elution
- Mobile phase composition changes: in gradient HPLC, changing solvent ratios shift baseline
- Electronic drift: amplifier drift, ADC offsets

**Why Baseline Matters for Quantification**
- Peak area integration: area under a peak ∝ analyte concentration (Beer-Lambert analog)
- If baseline is wrong, peak areas are wrong → concentration errors
- Limit of detection: baseline noise level determines smallest detectable peaks
- Calibration curves: systematic baseline error biases all calibrations

### Key details from your work
- Your synthetic generator (`SyntheticDataGenerator` in train.py) creates baselines via:
  - Low-order polynomials (degree 2-4)
  - Sinusoidal combinations (simulating periodic drift)
  - Spline interpolation of random control points
- Peaks generated as piecewise-linear or Gaussian shapes
  - Width: 1–7 samples (for N=1024)
  - Height: 13–215× baseline amplitude
  - 1–many peaks per signal
- Noise: additive Gaussian with controllable SNR (std=0.01 in v7)

### Figure
- **Fig 2.1**: Annotated chromatogram with peaks, baseline, and noise labeled. Generate from `demo.py` with clear annotations.

---

## Section 2.2: Classical Baseline Correction Methods (5–7 pages)

### 2.2.1: BEADS — Baseline Estimation And Denoising using Sparsity (3–5 pages)

**THIS IS THE CORE — cover in full mathematical detail**

### What to write

**The Optimization Problem**
```
minimize  ||Dx||_1 + lambda_0 * ||x||_1 + (lambda_1/2) * ||Hf||_2^2 + (lambda_2/2) * ||f||_2^2
subject to  y = x + f  (+ noise tolerance)
```

**Variable Splitting / MM Approach**
- Majorization-Minimization framework for non-smooth L1 terms
- Iterative reweighting: w(z) = 1/(|z| + eps) approximates L1 penalty
- Weighted diagonal matrices: Lambda = diag(w(D@x)), Gamma = diag(1/(|x|+eps))

**BAfilt Construction**
- Band-pass filter via cascaded Butterworth sections
- Filter coefficients computed via scipy.signal.butter
- Role of filter order d (1 or 2) and cutoff frequency fc
- B matrix: banded Toeplitz structure from filter coefficients
- A matrix: derived from filter, forms basis for banded system

**The Iterative Update Scheme**

f-update: banded linear solve
```
(B^T B + lambda_1 * H^T H + lambda_2 * I) f = B^T(y - x)
```
- Banded structure enables efficient O(N) solve
- H is the difference matrix for smoothness enforcement

x-update: proximal/shrinkage with asymmetric penalty
```
M = 2*lambda_0*Gamma + D'*Lambda*D
x = A @ inv(B^T B + A'*M*A) @ d_vec
```
- Theta weighting: how r controls asymmetry (penalize negative peaks more)

**Role of Each Parameter**
| Parameter | Controls | Typical Range |
|-----------|----------|---------------|
| lambda_0 | Peak sparsity (L1 penalty weight) | 0.001–1.0 |
| lambda_1 | Baseline smoothness (1st derivative) | 0.01–10 |
| lambda_2 | Baseline fidelity (2nd derivative) | 0.01–10 |
| r | Asymmetry ratio (negative vs positive) | 1–10 |
| fc | Cutoff frequency (band-pass filter) | 0.001–0.1 |
| d | Filter order | 1 or 2 |

**Why Parameters Are Hard to Tune**
- Different chromatographic systems have different optimal parameter sets
- fc depends on baseline drift frequency — varies by instrument/method
- r depends on noise characteristics
- lambda values interact nonlinearly
- No principled way to set them from signal properties alone

### Key details from your work
- `Implementations/0. BEADS/WithNumpy/beads.py`: Your NumPy port of the algorithm
- `Legacy Matlab Code/BEADS_toolbox/beads.m`: Reference MATLAB implementation
- `Literature/Research Papers/BEADS.pdf`: Original paper
- Default values used in your code: fc=0.006, r=6.0, d=1, lam0/lam1/lam2 varied
- The classical BEADS typically runs 30 iterations to converge

### Figures & Tables
- **Fig 2.2**: BEADS algorithm block diagram / iteration flow (create from algorithm structure)
- **Table 2.1**: Parameter roles and typical ranges (from table above)

---

### 2.2.2: Other Classical Methods (1–2 pages)

### What to write
Brief coverage of alternatives, positioning relative to BEADS:

- **arPLS** (Baek et al.): Asymmetric Reweighted Penalized Least Squares
  - Iteratively reweights residuals — points above baseline get lower weight
  - Simpler than BEADS but less principled
- **airPLS**: Adaptive Iteratively Reweighted PLS
  - Adaptive version of arPLS with automatic weight adjustment
- **SNIP**: Statistics-sensitive Non-linear Iterative Peak-clipping
  - Window-based approach, clips signal to remove peaks iteratively
- **AsLS**: Asymmetric Least Squares
  - Foundational method, asymmetric penalty on residuals
- **Rubberband method**: Connect minima with straight lines — simple but crude

**Why BEADS is the best unrolling candidate:**
- Explicit optimization formulation (not heuristic)
- Clear per-iteration structure (f-update → x-update → reweight)
- Interpretable parameters (each lambda has a meaning)
- Rich enough to benefit from learning (6 parameters to learn)

### Key details from your work
- v4 and later versions compare against pybaselines implementations (arPLS, airPLS, SNIP, AsLS)
- Classical methods used as baselines in evaluation (Ch. 4/5)

---

## Section 2.3: Algorithm Unrolling (4–5 pages)

### What to write

**Core Idea**
- Take an iterative algorithm that runs K iterations to converge
- "Unroll" it: lay out K iterations as K sequential layers in a neural network
- Algorithm parameters (step sizes, regularization weights) become learnable via backpropagation
- Network architecture encodes algorithmic structure — NOT a black-box

**LISTA Lineage**
- Gregor & LeCun (2010): Learned ISTA (LISTA) — the founding work
- Unrolled K iterations of ISTA for sparse coding
- Each layer: one ISTA step with learnable dictionary and threshold
- Showed: unrolled network converges faster than original ISTA with same K

**General Framework**
```
Iterative Algorithm:          Unrolled Network:
x^{k+1} = T(x^k; theta)  →  Layer k: x^{k+1} = T(x^k; theta_k)
                               where theta_k are LEARNABLE per layer
```
- Computation graph: forward pass = running the algorithm
- Backpropagation: adjusts parameters to minimize task loss
- Each layer can have independent parameters → parameter specialization

**Why It Works**
- Preserves algorithmic structure → interpretable intermediate representations
- Fewer parameters than generic deep networks (structured vs free)
- Domain knowledge baked into architecture → better inductive bias
- In some cases: convergence guarantees carry over

**Key Advantages Over Black-Box Deep Learning**
- Interpretable: layer k output = estimate after k algorithm iterations
- Parameter efficient: only learn algorithm parameters, not arbitrary weights
- Generalizable: algorithmic structure transfers across problem sizes
- Analyzable: can study learned parameters and compare to theory

**Examples Beyond LISTA**
- Unrolled ADMM for various inverse problems
- Learned ISTA variants (LISTA, ALISTA, ISTA-Net)
- Deep unfolding for MRI reconstruction (ADMM-Net, VN-Net)
- Unrolled algorithms for compressed sensing, image denoising, beamforming

**How This Applies to BEADS**
- Each BEADS iteration → one BEADSLayer
- Learnable per layer: lambda_0, lambda_1, lambda_2, r (log-parameterized)
- Fixed: fc (not learnable — autograd boundary, discussed in Ch. 3.5)
- K=8 layers → 8 "iterations" with per-layer parameter specialization

### Key details from your work
- v1 was the proof of concept: BEADSLayer with learnable log_lam0, log_lam1, log_lam2, log_r
- Two variants tried: exact sparse solve (LBEADS_NET) vs gradient descent approx (LBEADS_NET_Fast)
- The Fast/ISTA variant ultimately won — better gradient flow (detailed in Ch. 3.3)
- Your DSP-II lecture notes in `Literature/From DSP-II Lectures/` cover ISTA, ADMM, sparse coding foundations

### Figures
- **Fig 2.3**: Generic algorithm unrolling concept diagram — iterative algorithm → unrolled network with learnable parameters per layer

### References
- Gregor & LeCun (2010) — LISTA
- Monga et al. — algorithm unrolling survey
- Chen et al. — learned ISTA variants
- ADMM-Net papers
- Your Literature/ folder has PDFs for ISTA/ADMM background

---

## Successes to Highlight
- The signal model y=x+f+w is cleanly defined and used consistently across the thesis
- BEADS is well-suited for unrolling — your entire project validates this choice
- Algorithm unrolling gives you the best of both worlds: interpretability + learning

## Challenges to Acknowledge
- BEADS is complex (6 params, banded matrices, iterative reweighting) — unrolling it is nontrivial
- fc non-learnability is a fundamental limitation of the unrolling approach for BEADS specifically
