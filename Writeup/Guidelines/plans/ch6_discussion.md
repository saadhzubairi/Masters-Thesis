# Chapter 6 — Discussion & Limitations (5–7 pages)

File: `discussion/discussion.tex` (CREATE this directory + files)

**TURN WEAKNESSES INTO THESIS STRENGTH — ANALYZE, DON'T JUST REPORT**

This chapter is what separates a good thesis from a great one. Examiners value honest, analytical discussion of failures more than inflated success claims. Each limitation should be explained mechanistically: why it happens, what you tried, why it persists, and what it teaches the community.

---

## Section 6.1: Baseline Leakage in Dense-Peak Regions (1.5 pages)

### What to write

**The Observation**
- When peaks are densely packed (many overlapping peaks, few zero-valued gaps), the baseline estimate absorbs peak energy — "leakage"
- Visually: baseline rises at peak locations, peaks are underestimated
- This was the central challenge across v3→v7

**Why It Happens — Mechanistic Analysis**
- The BEADS formulation assumes peaks are sparse: ||x||_1 should be small
- Sparsity assumption ≡ most of x should be zero (peaks occupy few samples)
- Dense peaks violate this: x has large support → L1 penalty pushes energy into f
- The model finds a lower-loss solution by splitting energy between x and f
- Fundamental: L1-based sparsity has an inherent bias in dense regions

**What You Tried (the mitigation journey)**
| Version | Mitigation | Outcome |
|---------|-----------|---------|
| v4 | Baseline supervision (masked MSE) | Helped significantly but not fully |
| v6 | Orthogonality penalty (gradient-based) | Modest improvement |
| v7 | Asymmetric baseline loss (9:1 ratio) | Reduced over-estimation |
| v7 | Element-wise orthogonality ||x⊙f||_1 | Better mutual exclusivity |
| v7 | Envelope constraint (f ≤ local min) | Prevents baseline excursions |
| v7 | Frequency separation (FFT-based) | Enforces spectral roles |

**Why It Persists**
- The fundamental issue is the sparsity assumption, not the loss design
- All mitigations are regularizers on top of a sparsity-biased formulation
- To fully solve: need a non-sparsity-based decomposition (different paradigm)

**Broader Insight**
- Any algorithm-unrolled network that inherits sparsity assumptions will face this in dense regimes
- This is not specific to BEADS or chromatography — it's a property of L1-based decomposition

### Figures
- **Fig 6.1**: Dense-peak leakage example — show ground truth vs LBEADS output with visible leakage

---

## Section 6.2: Training/Inference Length Mismatch (1.5 pages)

### What to write

**The Problem**
- Training: N = 1024 (all synthetic signals)
- Inference: real chromatograms often N = 4096 or longer
- The network has never seen signals this long during training

**What Changes With Length**
- BAfilt matrices change dimension: A, B become NxN → different filter behavior
- Spectral properties shift: longer signals have different frequency content
- More peaks per signal: longer signals typically have more analyte responses
- Different baseline shapes: longer runs have different drift patterns

**Which Components Are Length-Sensitive vs Length-Invariant**
| Component | Length Sensitivity | Why |
|-----------|-------------------|-----|
| BAfilt (B, A matrices) | HIGH — matrices change with N | Filter coefficients depend on signal length |
| Learned lambdas | MODERATE — learned for N=1024 distribution | Optimal regularization may differ at other lengths |
| Pointwise shrinkage | LOW — operates element-wise | Independent of signal length |
| TV penalty | LOW — local differences | Mostly local operation |
| Frequency separation | MODERATE — FFT resolution changes | Different freq bin widths at different N |

**Potential Fixes**
- Train on variable lengths (N ∈ {512, 1024, 2048, 4096})
- Sliding-window inference: break long signal into 1024-length windows, process each, stitch
- Convolutional architecture: inherently length-agnostic
- Fine-tune on target length before inference

### Key details from your work
- fc=0.006 may be suboptimal at N=4096 — different Nyquist-relative position
- demo_chromatogram.py likely pads or truncates real signals to N=1024
- This is documented but not solved in current implementation

### Figures
- **Fig 6.2**: Same model applied to N=1024 vs N=4096 signal — show performance difference

---

## Section 6.3: Softplus vs. Hard Threshold — Train/Inference Mismatch (1 page)

### What to write

**The Setup**
- softplus(x, beta=5) used during training for non-negativity
- softplus(x) ≈ ReLU(x) for |x| >> 1/beta, but softplus(0) = ln(2)/beta ≈ 0.139
- During training: smooth gradients enable learning through zero-crossing
- During inference: softplus introduces small positive bias in near-zero regions

**The Mismatch**
- Training: model learns to output slightly negative values relying on softplus to map them to ~0
- Inference: these near-zero regions become small positive values instead of true zeros
- Effect: baseline and peak separation has a small positive floor
- Magnitude: ~0.000016 minimum value (from verification results) — small but nonzero

**Broader Lesson**
- Any algorithm-unrolled network using approximate activation functions faces this
- Learned proximal operators (softplus, smooth L1, etc.) introduce approximation artifacts
- The community should be aware: train-time smoothness ≠ inference-time exactness
- This is a general insight about differentiable relaxations in unrolled networks

### Connection to future work
- Exact thresholding at inference time (straight-through estimator during training)
- Learnable activation functions that bridge the gap

---

## Section 6.4: The CG Variant's Failure (1.5 pages)

### ANALYTICAL CONTENT EXAMINERS VALUE

### What to write

**The Original Design**
- CG-based f-update: run K_inner CG steps inside each BEADSLayer
- Motivation: CG gives a more accurate solution to the banded linear system
- Implementation: `_cg_solve_fixed()` in v6's lbeads_net.py

**Why It Failed — Gradient Analysis**
- Backpropagation depth: K_inner CG steps × K_outer unrolled layers
  - Example: 10 CG steps × 8 layers = 80 sequential operations
  - Gradient must flow backward through all 80
- Vanishing gradient problem: gradient norms decay exponentially
  - Layer 1 receives gradient ~ (decay_rate)^80 × original gradient
  - For decay_rate = 0.95, this is 0.95^80 ≈ 0.017 — 98% loss
- CG convergence criterion: ||r||_2 < tolerance
  - Not differentiable: hard threshold creates zero-gradient regions
  - Different samples may converge in different number of steps → ragged computation graph

**Empirical Evidence**
- v6 phases P1, P2, P3 experimented with CG-based training
- Gradient magnitudes measured per layer show exponential decay
- Early layer parameters barely change during training

**The ISTA Fix**
- Replace K_inner CG steps with 1 proximal gradient step
- Backprop depth: 1 × K_outer = 8 operations (manageable)
- Trade: less accurate per-layer → more layers needed → but gradients flow

**Lesson for the Algorithm Unrolling Community**
- Not all iterative algorithms unroll equally well
- The inner solver's gradient properties determine trainability
- Simpler per-layer computation + more layers beats complex per-layer + fewer layers
- Connection to implicit differentiation: exact inner solve → implicit function theorem → better gradients (but harder to implement)

### Figures
- **Fig 6.3**: Gradient magnitude per layer — CG variant (exponential decay curve) vs ISTA variant (stable/slowly decaying)

---

## Section 6.5: fc Non-Learnability (1 page)

### What to write

**The Problem**
- fc (cutoff frequency) controls the Butterworth filter in BAfilt
- BAfilt is implemented via `scipy.signal.butter` → NumPy/SciPy operations
- PyTorch autograd cannot compute gradients through scipy
- Therefore: d(loss)/d(fc) = 0 — fc has no gradient

**Consequences**
- fc is a hyperparameter, not a learnable parameter
- fc=0.006 used for all experiments — may be suboptimal for some signals
- A truly adaptive system would learn fc per signal or per layer
- This limits generalization: signals with different baseline frequency characteristics require different fc

**Why This Is Fundamental**
- The butterworth filter design involves transcendental functions (poles, zeros in complex plane)
- Even with a PyTorch reimplementation, butterworth pole computation is non-trivial to differentiate
- This is a general challenge for any unrolled algorithm that uses non-differentiable subroutines

**Potential Solutions (connect to Ch. 7 future work)**
1. Differentiable IIR filter design: parameterize filter response directly in frequency domain
2. Replace butterworth with learnable 1D convolution kernels (gives up principled filter structure)
3. Surrogate gradient: finite-difference approximation of d(loss)/d(fc)
4. Meta-learning: train a separate network to predict fc from signal features
5. Differentiable DSP libraries (e.g., torchaudio) for filter design

---

## Writing Guidance for This Chapter

### Tone
- Analytical, not apologetic. Don't say "unfortunately, our method fails at X."
- Instead: "We observe degraded performance in regime X. Analysis reveals this stems from the L1 sparsity assumption inherent to the BEADS formulation, which..."
- Frame each limitation as an insight: "This observation provides guidance for..."

### Structure for each section
1. State the observation clearly (what goes wrong)
2. Explain the mechanism (why it happens — mathematically if possible)
3. Describe mitigation attempts (what you tried)
4. Explain why the limitation persists (fundamental vs fixable)
5. Extract a generalizable lesson (value to the community)

### What makes this chapter strong
- You have FIVE well-characterized failure modes — this is thorough
- Each has a mechanistic explanation, not just empirical observation
- The CG failure and fc non-learnability are contributions in themselves
- These insights are valuable to anyone doing algorithm unrolling
