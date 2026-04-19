# PROMPT: Write Chapter 3 — Proposed Method: LBEADS-NET (15–18 pages)

## ROLE
You are an expert academic writer producing the core contribution chapter of an NYU Tandon MS thesis. This is the most important chapter — it describes the novel method. Write with mathematical precision. LaTeX output.

## OUTPUT
Produce the full LaTeX content for `method/method.tex`. Begin with `\chapter{Proposed Method: LBEADS-NET}`. Contains Sections 3.1–3.8.

## SOURCE FILES TO READ

**Critical — read these completely and carefully:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — THE architecture file (1282 lines). Study:
  - `BEADSLayer` class: learnable parameters (log_lam0, log_lam1, log_lam2, log_r, step_size), forward pass, f-update, x-update
  - `LBEADS_NET` class: how K layers are chained, output_gain, softplus application
  - `_banded_apply()`: efficient banded matrix operations
  - `_setup_filters()`: how A, B matrices are built from Butterworth coefficients
  - Constants: EPS0=1e-6, EPS1=1e-6
  - How intermediate outputs are collected

- `Implementations/7. LBEADS_NETv7 [Stronger]/train.py` — THE training file (2031 lines). Study:
  - `SparsityLoss` class: ALL loss computation methods:
    - `forward()`: how total loss is assembled
    - `_asymmetric_baseline_loss()`: asymmetric penalty implementation
    - `_envelope_loss()`: soft local minimum via log-sum-exp
    - `_freq_separation_loss()`: FFT-based frequency separation
    - Orthogonality computation: element-wise ||x_pred ⊙ f_pred||₁
  - `stage_configs` list: the three training stages with their loss weights
  - Training loop: how stages transition, intermediate supervision logic
  - `SyntheticDataGenerator` class (needed for understanding data but mainly Ch. 4)

**For the CG variant discussion:**
- `Implementations/6. LBEADS_NETv6 [Strong]/lbeads_net.py` — contains `_cg_solve_fixed()` (the CG solver that was abandoned). Read this to understand what was tried and why it failed.

**For the evolution narrative:**
- `Implementations/1. LBEADS_NETv1/lbeads_net.py` — the original BEADSLayer (simpler version, shows starting point)
- `Implementations/5. LBEADS_NETv5 [Adaptive Post Processing]/lbeads_net.py` — scan `HybridConfig`, `hybrid_infer_1d()`, `beads_classic_with_init()` for Section 3.8

**For loss function research:**
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Sections 2-3 for loss term motivation, gradient-overlap penalty, adaptive smoothness, learned proximal operators. Section 7 for BLI metric.

**Reference:**
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 3 specification

## SECTION SPECIFICATIONS

### Section 3.1: Problem Formulation (1–2 pages)
- Restate y = x + f + w
- Goal: learn (y; Θ) → (x̂, f̂)
- Design philosophy: **unroll BEADS, don't replace it**
  - Preserve optimization structure of BEADS
  - Make algorithm parameters learnable
  - Retain interpretability of intermediate outputs
- Refer reader to Ch. 2.2.1 for full BEADS background

### Section 3.2: From BEADS to BEADSLayer (4–5 pages)
**This section must be precise enough that a reader can understand exactly what happens inside each layer. Reference the actual code.**

Write in this order:

1. **Learnable parameters per layer** — Create Table 3.1:
   | Parameter | Stored As | Actual Value | Purpose |
   | λ₀ | log_lam0 (nn.Parameter) | exp(log_lam0) | Peak sparsity |
   | λ₁ | log_lam1 | exp(log_lam1) | Baseline smoothness |
   | λ₂ | log_lam2 | exp(log_lam2) | Baseline fidelity |
   | r | log_r | exp(log_r) | Asymmetry ratio |
   | step_size | step_size | direct | Gradient step (ISTA) |
   | output_gain | output_gain | direct | Output scaling |

   Explain log-space parameterization: λ = exp(log_λ) guarantees positivity without constrained optimization. Each layer has INDEPENDENT parameters.

2. **BAfilt within the layer** — From `_setup_filters()` in lbeads_net.py:
   - Butterworth coefficients via scipy.signal.butter(d, fc)
   - Coefficients → banded Toeplitz matrices A ∈ ℝᴺˣᴺ and B ∈ ℝᴺˣᴺ
   - These are PRECOMPUTED ONCE and stored as buffer tensors (not learnable)
   - fc and d are fixed — explained in Section 3.5

3. **f-update** — The banded linear system:
   ```
   (B^TB + λ₁H^TH + λ₂I) f = B^T(y - x)
   ```
   - Left side has banded structure → O(N) solve
   - H = second-difference operator
   - In ISTA variant: replace exact solve with single gradient step (see 3.3)

4. **x-update** — Proximal/shrinkage:
   - Compute weights: wᵢ = 1/(|xᵢ| + ε)
   - Build: Γ = diag(w), Λ = diag(w(Dx))
   - M = 2λ₀Γ + D'ΛD
   - x = A·inv(B^TB + A'MA)·d_vec
   - Θ weighting via r: negative residuals penalized r× more

5. **Algorithm box** — Algorithm 3.1: BEADSLayer forward pass pseudocode

6. **Fig 3.1**: Single BEADSLayer architecture diagram

### Section 3.3: The ISTA-Style Fast Variant (2–3 pages)
**KEY ANALYTICAL CONTENT — this demonstrates deep understanding of the design space.**

1. **CG-based variant** (what was tried first):
   - Used conjugate gradient to solve the f-update linear system
   - Ran K_inner CG steps inside each BEADSLayer
   - Implementation: `_cg_solve_fixed()` in v6's lbeads_net.py
   - Advantage: more accurate per-layer solution

2. **Why CG failed** — gradient analysis:
   - Backprop depth: K_inner CG × K_outer layers (e.g., 10×8 = 80 sequential ops)
   - Vanishing gradients: gradient norms decay exponentially across layers
   - For decay_rate ≈ 0.95: 0.95^80 ≈ 0.017 → 98% gradient loss
   - Early layer parameters receive negligible gradients → don't update → don't learn
   - CG convergence criterion (||r||₂ < tol) is NOT differentiable → zero-gradient regions
   - Different samples converge in different steps → ragged computation graph

3. **The ISTA fix**:
   - Replace K_inner CG steps with 1 proximal gradient step
   - f^{k+1} = f^k - step_size · ∇_f L + proximal_term
   - Backprop depth: 1 × K_outer = 8 operations (manageable)
   - Learned step_size per layer compensates for reduced per-layer accuracy
   - step_size typically converges to ~0.986 (near 1.0)

4. **Analytical comparison**:
   - CG: more accurate per-layer, but gradients don't flow → network doesn't learn
   - ISTA: less accurate per-layer, but clean gradient flow → network LEARNS to compensate with more layers
   - Trade: per-iteration accuracy for trainability

5. **Fig 3.2**: Gradient magnitude across layers — CG (exponential decay) vs ISTA (stable)

6. **Generalizable lesson**: not all iterative algorithms unroll equally well. The inner solver's gradient properties determine trainability. Simpler per-layer + more layers > complex per-layer + fewer layers.

### Section 3.4: Full K-Layer Pipeline (2–3 pages)
- K=8 BEADSLayers stacked sequentially (upgraded from K=5 in v1)
- Each layer: independent learnable parameters
- Input: y, x⁰=0, f⁰=0
- Each layer refines: x^{k+1}, f^{k+1} = BEADSLayer_k(y, x^k, f^k)
- Output: x̂_K, f̂_K

**Parameter specialization** (KEY observation):
- Early layers: coarse separation (larger effective regularization)
- Later layers: fine refinement (precise peak/baseline boundary)
- This specialization EMERGES from training — not manually designed
- Validates the algorithm unrolling approach: the network discovers that different stages need different parameters
- Detailed parameter values reported in Ch. 5.1.2

**Output processing:**
- output_gain (learnable) scales peak estimate
- softplus(x, β=5) applied for non-negativity (see 3.7.5)
- Intermediate outputs at every layer → used for supervision (3.6) and quality scoring (3.8)

**Fig 3.3**: Full K-layer LBEADS-NET pipeline (data flow through all 8 layers)

### Section 3.5: Non-Learnable fc and the Autograd Boundary (1–2 pages)
**ANALYTICAL SECTION — demonstrates understanding of framework limitations.**

- fc controls Butterworth filter in BAfilt
- BAfilt uses `scipy.signal.butter` → NumPy/SciPy operations OUTSIDE PyTorch autograd
- ∂loss/∂fc = 0 — no gradient flows through scipy
- fc must be fixed as hyperparameter (fc=0.006 in all experiments)
- This means fc determines the baseline/peak frequency boundary but CANNOT adapt per signal

**Why this is fundamental:**
- Butterworth filter design involves transcendental functions (poles/zeros in complex plane)
- Even with PyTorch reimplementation, differentiating through pole computation is nontrivial
- This is a general challenge for any unrolled algorithm using non-differentiable subroutines

**Potential solutions** (connect to Ch. 7 future work):
- Differentiable IIR filter design (parameterize frequency response in PyTorch)
- Replace Butterworth with learnable 1D convolution kernels
- Surrogate gradients (finite-difference approximation)
- Meta-learning (predict fc from signal features)
- Differentiable DSP libraries (torchaudio)

### Section 3.6: Multi-Stage Curriculum Training (2–3 pages)

**Motivation**: 11-term composite loss creates complex loss landscape. Direct optimization from epoch 1 fails — conflicting gradients, training instability. Curriculum: introduce terms gradually.

**Stage A — Warmup (5 epochs):**
- Loss: MSE only (α_mse=1.0, all others=0.0)
- Purpose: establish basic peak/baseline separation without conflicting gradient signals
- The network learns a rough decomposition

**Stage B — Structured Learning (15 epochs):**
- Activate: α_baseline=0.5, α_asym_baseline=1.0, α_ortho=0.1
- Purpose: teach CORRECT separation. Baseline supervision prevents leakage. Asymmetric penalty prevents over-estimation. Orthogonality enforces mutual exclusivity.
- Baseline supervision uses masking: only supervise in non-peak regions

**Stage C — Refinement (10 epochs):**
- Activate: ALL remaining terms (L1, TV, smooth, neg, leakage, envelope, freq)
- Enable intermediate supervision: compute loss at every unrolled layer
- Layer weights: linearly increasing 0.1 → 1.0 (more weight on refined estimates)
- Purpose: fine-tune leakage suppression, enforce all structural constraints

**Table 3.2**: Training stage configuration

**Fig 3.5**: Training stage progression diagram

**Implementation**: `stage_configs` list in train.py. Optimizer: Adam, lr=1e-3. No LR reset between stages. Loss weights change discretely at stage boundaries.

### Section 3.7: Composite Loss Function (3–4 pages)
**Cover ALL 11 terms. For each: mathematical formula, purpose, weight, and which stage activates it.**

Write subsections 3.7.1–3.7.10 covering:

1. **Reconstruction** (α_mse=1.0): ||y - (x̂ + f̂)||₂²
2. **L1 sparsity** (α_l1=0.01): ||x̂||₁
3. **Total variation** (α_tv=0.01): ||∇x̂||₁
4. **Baseline supervision** (α_baseline=0.5): ||f̂[mask] - f_true[mask]||₂² — THE CRITICAL FIX from v4. Explain masking strategy.
5. **Smoothness** (α_smooth=0.2): ||∇²f̂||₂²
6. **Non-negativity** (α_neg=0.5): ||max(0,-x̂)||² + architectural softplus(x,β=5). Explain why softplus over ReLU (smoother gradients, no dead neurons, β=5 ≈ ReLU behavior).
7. **Leakage** (α_leakage=0.3): correlation between peak residual and baseline
8. **Asymmetric baseline** (α_asym=1.0, α=0.9): over-estimation penalized 9× more. From `_asymmetric_baseline_loss()`.
9. **Orthogonality** (α_ortho=0.1): ||x̂ ⊙ f̂||₁ — element-wise. Adapted from orthogonal NMF. Evolution: v6 gradient-based → v7 direct product.
10. **Envelope** (α_envelope=0.5): baseline ≤ soft_local_min(y). From `_envelope_loss()`. Log-sum-exp soft minimum for differentiability.
11. **Frequency separation** (α_freq=0.05): FFT-based. From `_freq_separation_loss()`. High-freq in baseline penalized + low-freq in peaks penalized.

**Table 3.3**: Complete loss summary (term, formula, weight, stage activated)

### Section 3.8: Hybrid Inference Pipeline (1–2 pages)
- Forward pass → initial (x̂, f̂)
- Optional adaptive lowpass denoising on peaks
- Optional classical BEADS refinement: use learned parameters as warm-start initialization, run additional BEADS iterations
- Quality scoring: evaluate each stage output via baseline_hf_ratio (high-freq content in baseline) and residual_hf_rms. Select stage with best score.
- Why hybrid: LBEADS provides good initialization, classical BEADS provides convergence guarantees. Together > either alone.
- Reference `HybridConfig`, `hybrid_infer_1d()` from v5's lbeads_net.py

**Algorithm 3.3**: Hybrid inference pseudocode
**Fig 3.4**: Hybrid inference pipeline diagram

## STYLE CONSTRAINTS
- Mathematical precision: every variable defined, every equation numbered
- Use `\begin{algorithm}` environment for pseudocode (Algorithm 3.1, 3.2, 3.3)
- Use `\begin{table}` for Tables 3.1, 3.2, 3.3
- Cross-reference Ch. 2 for BEADS background: "As described in Section~\ref{sec:beads}..."
- Code references: when describing implementation, cite the file but do NOT include Python code. Translate to mathematical notation.
- The ISTA vs CG analysis (3.3) and fc analysis (3.5) are KEY examiner-grade sections — write them with analytical depth, not just description
