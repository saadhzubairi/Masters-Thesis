# Chapter 3 — Proposed Method: LBEADS-NET (15–18 pages)

File: `method/method.tex` (CREATE this directory + files)

**This is the core contribution chapter — write FIRST**

---

## Section 3.1: Problem Formulation (1–2 pages)

### What to write
- Restate signal decomposition: y = x + f + w
- Goal: learn a mapping (y; Theta) → (x_hat, f_hat) where Theta are trainable parameters
- Design philosophy: **unroll BEADS, don't replace it**
  - Preserve BEADS's optimization structure
  - Make parameters learnable via backpropagation
  - Retain interpretability of intermediate outputs

### Key details from code
- From `lbeads_net.py`: The LBEADS_NET class takes y as input and produces (x_pred, f_pred)
- Forward pass chains K BEADSLayer modules sequentially
- Each layer refines the estimate: x^{k+1}, f^{k+1} = BEADSLayer_k(y, x^k, f^k)

---

## Section 3.2: From BEADS to BEADSLayer (4–5 pages)

### What to write

**Mapping One BEADS Iteration → One Neural Network Layer**

Each BEADSLayer performs exactly one BEADS iteration but with learnable parameters:

**Learnable Parameters per Layer**
| Parameter | Parameterization | Purpose |
|-----------|-----------------|---------|
| log_lam0 | exp(log_lam0) = lambda_0 | Peak sparsity weight |
| log_lam1 | exp(log_lam1) = lambda_1 | Baseline smoothness (1st deriv) |
| log_lam2 | exp(log_lam2) = lambda_2 | Baseline fidelity (2nd deriv) |
| log_r | exp(log_r) = r | Asymmetry ratio |
| step_size | learned | Gradient step size (ISTA variant) |
| output_gain | learned | Output scaling factor |

- **Log-space parameterization**: lambda = exp(log_lambda) guarantees positivity without constrained optimization
- Parameters are independent per layer — enables specialization

**BAfilt Construction Within the Layer**
- Butterworth filter coefficients computed via scipy.signal.butter(d, fc)
- Coefficients → banded Toeplitz matrices A and B
- These matrices are FIXED (precomputed) — fc and d are not learnable
- B is the band-pass filter matrix, A is the normalizing matrix
- Both have banded structure → efficient O(N) operations

**f-update Step: Banded Linear System Solve**
```
(B^T B + lambda_1 * H^T H + lambda_2 * I) f = B^T(y - x)
```
- Left-hand side has banded structure → O(N) solve
- H = difference operator for smoothness
- This is the bottleneck step — two variants attempted (see 3.3)

**x-update Step: Proximal/Shrinkage Operator**
- Compute penalty weights: w_i = 1/(|x_i| + eps)
- Build weighted matrices: Gamma = diag(w), Lambda = diag(w(D@x))
- Shrinkage: M = 2*lambda_0*Gamma + D'*Lambda*D
- Solve: x = A @ inv(B^T B + A'*M*A) @ d_vec
- The Theta weighting matrix controls asymmetry via r:
  - Negative residuals penalized r× more than positive ones
  - Encourages positive peak estimates

### Key details from code
- `lbeads_net.py` lines defining BEADSLayer class
- `_banded_apply()`: efficient banded matrix-vector multiplication
- Filter setup: `_setup_filters()` builds A, B from butterworth coefficients
- EPS0=1e-6, EPS1=1e-6 for numerical stability in weight functions
- Matrices precomputed once and cached as buffers

### Figures & Algorithm Boxes
- **Fig 3.1**: Single BEADSLayer architecture diagram — show input (y, x^k), internal operations (f-update, x-update, reweighting), output (x^{k+1}, f^{k+1})
- **Algorithm 3.1**: BEADSLayer forward pass pseudocode

---

## Section 3.3: The ISTA-Style Fast Variant (2–3 pages)

### THIS IS KEY ANALYTICAL CONTENT — EXAMINER-GRADE

### What to write

**Original Design: CG-Based Variant**
- Conjugate Gradient (CG) used to solve the f-update linear system
- Run K_inner CG steps inside each BEADSLayer
- Advantage: more accurate per-layer solution
- Implementation: `_cg_solve_fixed()` in lbeads_net.py (v6)

**Why CG Failed**
- Gradient must backpropagate through K_inner CG iterations × K_outer unrolled layers
- Deep computation graph → vanishing gradients
- CG convergence criterion (residual norm < tolerance) is not differentiable
- Empirically: gradient magnitudes decay exponentially across layers
- Early layers receive negligible gradients → parameters don't update → no learning

**The Fix: ISTA-Style Splitting**
- Replace CG with a single proximal gradient step for f-update
- Instead of solving the linear system exactly:
  ```
  f^{k+1} = f^k - step_size * gradient + proximal_term
  ```
- Trade per-iteration accuracy for better gradient flow
- Each layer does less work but gradients flow cleanly through all K layers

**Analytical Argument: Gradient Flow Comparison**
- CG variant: gradient passes through K_inner × K_outer = potentially 80+ operations
- ISTA variant: gradient passes through K_outer = 8 operations
- The learned step_size per layer compensates for reduced per-layer accuracy
- ISTA variant: network uses more layers to reach similar accuracy, but LEARNS to do so

### Key details from your work
- v1 had both LBEADS_NET (exact solve) and LBEADS_NET_Fast (gradient descent)
- v6 explicitly implemented CG solver: `_cg_solve_fixed()` with batched CG
- v6 experiments (P1, P2, P3 phases) showed CG variant underperformed
- v7 settled on ISTA-style approach with learned step sizes
- Learned step sizes typically converge to ~0.986 (near 1.0)

### Figures
- **Fig 3.2**: Gradient magnitude comparison across layers — CG variant (exponential decay) vs ISTA variant (stable gradients). Generate from training diagnostics or create schematic illustration.

### What makes this examiner-grade
- Shows you understand WHY things fail, not just what works
- Connects to broader algorithm unrolling literature: inner solver gradient properties matter
- Provides design guidance: simpler per-layer operations + more layers > complex per-layer operations + fewer layers

---

## Section 3.4: Full K-Layer Pipeline (2–3 pages)

### What to write
- K=8 BEADSLayers stacked sequentially (upgraded from 5 in earlier versions)
- Each layer has independent learnable parameters
- Input: y (observed signal), x^0=0, f^0=0 (initial estimates)
- Output: x^K (peak estimate), f^K (baseline estimate)

**Parameter Specialization**
- Early layers: coarse separation (large step sizes, aggressive regularization)
- Later layers: fine refinement (smaller adjustments, precise separation)
- This specialization emerges naturally from training — not designed manually
- Compare learned params to manually-tuned BEADS params (results in Ch. 5)

**Intermediate Outputs**
- Every layer produces (x^k, f^k) — interpretable as k-iteration BEADS estimate
- Used for intermediate supervision (see 3.6)
- Used for quality-scored stage selection in hybrid inference (see 3.8)

### Key details from code
- `LBEADS_NET.forward()` chains layers and optionally returns intermediate outputs
- `output_gain` parameter scales final peak estimate
- Softplus(x, beta=5) applied after output_gain for non-negativity (see 3.7.5)
- v7 increased from 5 → 8 layers for better separation

### Figures
- **Fig 3.3**: Full K-layer LBEADS-NET pipeline showing data flow through all 8 layers

---

## Section 3.5: Non-Learnable fc and the Autograd Boundary (1–2 pages)

### ANOTHER KEY ANALYTICAL SECTION

### What to write
- fc (cutoff frequency) controls the Butterworth filter in BAfilt
- BAfilt uses `scipy.signal.butter` → NumPy/SciPy operations OUTSIDE PyTorch autograd
- Cannot compute d(loss)/d(fc) — no gradient flows through scipy
- Consequence: fc must be fixed as a hyperparameter (fc=0.006 in all experiments)

**Why This Matters**
- fc determines the frequency boundary between "baseline" and "peaks"
- Different signals may have different optimal fc values
- A truly adaptive system would learn fc from data
- This is a fundamental limitation of the current architecture

**Potential Solutions (connect to future work, Ch. 7)**
- Differentiable IIR filter design: parameterize filter coefficients directly in PyTorch
- Surrogate gradients: approximate d(BAfilt)/d(fc)
- Meta-learning: predict fc from signal features
- Replace butterworth with learnable convolution kernels

### Key details from your work
- fc=0.006 used throughout all versions
- This was identified early but never resolved — it's an inherent scipy/PyTorch boundary issue
- The filter matrices A, B are precomputed once and stored as buffers
- Changing fc requires rebuilding all filter matrices

---

## Section 3.6: Multi-Stage Curriculum Training (2–3 pages)

### What to write

**Motivation**
- The full composite loss has 11 terms with different gradients
- Direct optimization from scratch fails — loss landscape is too complex
- Curriculum learning: introduce loss terms gradually

**Stage A: Warmup (5 epochs)**
- Loss: MSE only (alpha_mse = 1.0, all others = 0.0)
- Purpose: establish basic peak/baseline separation
- Prevents early divergence from conflicting loss gradients
- Network learns approximate signal decomposition

**Stage B: Structured Learning (15 epochs)**
- Loss: MSE + baseline supervision + asymmetric loss + orthogonality
- Active weights: alpha_mse=1.0, alpha_baseline=0.5, alpha_asym_baseline=1.0, alpha_ortho=0.1
- Purpose: teach correct peak/baseline separation
- Baseline supervision with masking (only supervise in non-peak regions)
- Asymmetric loss prevents baseline over-estimation

**Stage C: Refinement (10 epochs)**
- Loss: full composite with all 11 terms active
- Adds: envelope constraint, frequency separation, all sparsity terms
- Purpose: fine-tune leakage suppression
- Intermediate supervision enabled: compute loss at every unrolled stage

**Stage Transitions**
- No learning rate reset between stages (continuous optimization)
- Loss weights change discretely at stage boundaries
- Monitor for instability at transitions

### Key details from your work
- v3 had no curriculum — all losses from start, led to instability
- v4 introduced baseline supervision but not staged
- v7 formalized the three-stage approach in `stage_configs` list in train.py
- Intermediate supervision: linearly increasing weights 0.1 → 1.0 across K layers
- Optimizer: Adam with lr=1e-3, batch_size=24

### Tables & Figures
- **Table 3.2**: Training stage configuration (stage, epochs, active loss terms, weights)
- **Fig 3.5**: Training stage progression diagram — show which losses activate when

---

## Section 3.7: Composite Loss Function (3–4 pages)

### What to write — cover ALL 11 terms

#### 3.7.1: Reconstruction Loss
- L_mse = ||y - (x_hat + f_hat)||_2^2
- Anchor term, alpha_mse = 1.0
- Ensures decomposition reconstructs the original signal

#### 3.7.2: Sparsity Penalties
- L1: ||x_hat||_1 (alpha_l1 = 0.01) — promotes sparse peaks
- TV: ||nabla x_hat||_1 (alpha_tv = 0.01) — promotes piecewise-constant peaks
- Together: peaks should be sparse with sharp edges

#### 3.7.3: Baseline Supervision
- L_baseline = ||f_pred[mask] - f_true[mask]||_2^2 where mask = non-peak regions
- alpha_baseline = 0.5
- Masking strategy: threshold x_true to find peak locations, supervise baseline only where peaks are absent
- This was the CRITICAL FIX (v4) — without it, peaks leak into baseline

#### 3.7.4: Smoothness Penalty
- L_smooth = ||nabla^2 f_hat||_2^2 (second derivative of baseline)
- alpha_smooth = 0.2
- Enforces smooth, low-frequency baseline — complementary to frequency separation

#### 3.7.5: Non-Negativity
- Soft penalty: L_neg = ||max(0, -x_hat)||^2 (alpha_neg = 0.5)
- PLUS architectural constraint: softplus(x, beta=5) after output_gain
- Soft penalty provides gradient signal during training
- Softplus provides hard guarantee at inference
- Why softplus over ReLU: smoother gradients, avoids dead neuron problem
- beta=5 gives near-ReLU behavior while maintaining differentiability

#### 3.7.6: Baseline Leakage Penalty
- Measures correlation between peak residual and baseline
- alpha_leakage = 0.3
- Detects when peak energy leaks into baseline estimate
- Complements baseline supervision (which uses ground truth)

#### 3.7.7: Asymmetric Baseline Loss (NEW in v7)
- Over-estimation penalty: alpha * (f_pred - f_true)^2 when f_pred > f_true
- Under-estimation penalty: (1-alpha) * (f_pred - f_true)^2 when f_pred <= f_true
- alpha = 0.9 → penalize over-estimation 9x more
- Rationale: baseline rising above true level is worse than slight under-estimation (causes artificial peak area reduction)

#### 3.7.8: Element-Wise Orthogonality (evolved across versions)
- v6: gradient-based — mean(peak_weights * |diff1(f_pred)|^2)
- v7: direct element-wise — ||x_pred odot f_pred||_1 (alpha_ortho = 0.1)
- Enforces mutual exclusivity: where peaks are active, baseline should be zero and vice versa
- Direct product is more effective than gradient-based penalty

#### 3.7.9: Envelope Constraint (NEW in v7)
- Compute local signal minimum via soft sliding window (window=51, log-sum-exp trick)
- L_envelope = ||ReLU(f_pred - local_min)||_2^2 (alpha_envelope = 0.5)
- Prevents baseline from rising above true baseline in sparse regions
- Soft minimum uses temperature-scaled log-sum-exp for differentiability

#### 3.7.10: Frequency Separation Loss (NEW in v7)
- Compute FFT of x_pred and f_pred
- Split frequency axis into low and high bands
- L_freq = ||FFT(f_pred)[high]||^2 + ||FFT(x_pred)[low]||^2
- alpha_freq = 0.05
- Enforces spectral role: baseline = low-frequency, peaks = broadband
- Most expensive loss term — may skip in final training stage

### Key details from code
- `SparsityLoss` class in train.py implements all loss computations
- Each loss term has its own method: `_asymmetric_baseline_loss()`, `_envelope_loss()`, `_freq_separation_loss()`
- Total loss computed as weighted sum with configurable alphas
- Loss decomposition logged per epoch for diagnostics

### Tables
- **Table 3.3**: All loss terms, formulas, weights, and activation stage (big table — important reference)

---

## Section 3.8: Hybrid Inference Pipeline (1–2 pages)

### What to write
- LBEADS-NET forward pass → initial (x_hat, f_hat)
- Optional: adaptive lowpass denoising on peak estimate
- Optional: classical BEADS refinement using learned parameters as initialization
- Quality scoring: evaluate each stage output (raw LBEADS, denoised, refined)
  - Metrics: baseline_hf_ratio, residual_hf_rms
  - Select stage with best quality score
- Why hybrid: LBEADS provides good initialization, classical BEADS provides convergence guarantees

### Key details from your work
- v5 introduced hybrid pipeline with `hybrid_infer_1d()` and `HybridConfig`
- v6 diagnostics show "short_refine" (classical refinement) selected for all test samples
- This suggests LBEADS alone doesn't fully converge — hybrid is necessary
- Quality metrics defined in analysis/hybrid_diagnostics.py
- Real chromatogram results use hybrid pipeline exclusively

### Figures & Algorithm Boxes
- **Fig 3.4**: Hybrid inference pipeline diagram
- **Algorithm 3.3**: Hybrid inference with quality scoring pseudocode

---

## Evolution Story (for writing context, not directly in text)

| Problem Encountered | Version | Solution | Loss Term / Architectural Change |
|---------------------|---------|----------|--------------------------------|
| No ground truth for training | v1→v2 | Synthetic data generator | SyntheticDataGenerator |
| Peaks not sparse enough | v2→v3 | Sparsity losses | L1, TV, non-negativity |
| Peak energy leaks into baseline | v3→v4 | Baseline supervision | L_baseline (masked MSE) |
| Doesn't work on real data | v4→v5 | Hybrid inference | Quality-scored stage selection |
| CG solver kills gradients | v5→v6 | ISTA-style splitting | Learned step sizes |
| Baseline over-estimation | v6→v7 | Asymmetric penalty | L_asym_baseline (9:1 ratio) |
| Peaks and baseline co-activate | v6→v7 | Orthogonality | ||x odot f||_1 |
| Baseline exceeds local min | v7 | Envelope constraint | L_envelope |
| Spectral bleedthrough | v7 | Frequency separation | L_freq (FFT-based) |
| Negative peak values | v7 | Softplus constraint | F.softplus(x, beta=5) |
| Optimization instability | v7 | Curriculum training | 3-stage: A→B→C |
