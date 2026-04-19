# PROMPT: Write Chapter 6 — Discussion & Limitations (5–7 pages)

## ROLE
You are an expert academic writer producing the discussion chapter of an NYU Tandon MS thesis. This chapter turns weaknesses into thesis strength through rigorous mechanistic analysis. Write analytically, not apologetically. Each limitation = insight for the community. LaTeX output.

## OUTPUT
Produce the full LaTeX content for `discussion/discussion.tex`. Begin with `\chapter{Discussion and Limitations}`. Contains Sections 6.1–6.5.

## SOURCE FILES TO READ

**Critical — read for analytical depth:**
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Section 1 (three failure modes of BEADS), Section 2 (loss functions tried), Section 4 (how arPLS handles dense peaks differently — "elevation" vs "leakage" distinction), Section 7 (diagnostic metrics). This is the primary source for Section 6.1.
- `Writeup/Guidelines/baseline-leakage/resources.md` — citations for Gharbi et al. (2024), DIRAS+ (2025), CAE+ (2024) — these validate that leakage is a community-wide problem
- `Implementations/7. LBEADS_NETv7 [Stronger]/train.py` — Read the loss function implementations to understand exactly what mitigations were tried (asymmetric loss, orthogonality, envelope, frequency separation)
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — softplus(x, beta=5) implementation details for Section 6.3; filter construction for Section 6.5

**For CG variant analysis:**
- `Implementations/6. LBEADS_NETv6 [Strong]/lbeads_net.py` — `_cg_solve_fixed()` implementation. Understand the CG algorithm to explain why gradients vanish.
- `Implementations/6. LBEADS_NETv6 [Strong]/P1/`, `P2/`, `P3/` — phase experiments that showed CG underperformance

**For version evolution:**
- `Implementations/3. LBEADS_NETv3 [Sparsity based loss function]/train.py` — the starting point (5-term loss, no baseline supervision)
- `Implementations/4. LBEADS_NETv4 [Baseline fix]/train.py` — when baseline supervision was added (the breakthrough)

**Reference:**
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 6 specification

## SECTION SPECIFICATIONS

### Section 6.1: Baseline Leakage in Dense-Peak Regions (1.5–2 pages)

**Structure**: observation → mechanism → mitigations tried → why it persists → broader insight

1. **Observation**: In dense-peak regions (many overlapping peaks), the baseline estimate absorbs peak energy. Visually: baseline rises at peak locations, peaks underestimated.

2. **Mechanism** — three failure modes (from leakage doc Section 1):
   - **Sparsity assumption violation**: BEADS's L1 penalty assumes ||x||₁ is small (peaks occupy few samples). Dense peaks violate this. L1 over-shrinks → surplus energy attributed to baseline.
   - **Low-pass filter bandwidth mismatch**: overlapping peak tails create sustained elevation that looks low-frequency to the BAfilt filter → interpreted as baseline content.
   - **Uniform regularization**: same λ₀, λ₁, λ₂ across all spatial locations. Dense regions need different regularization than sparse regions. No mechanism to adapt spatially.

3. **Mitigation journey** (v3→v7) — tell this as a systematic engineering narrative:

   | Version | Mitigation | Result | Why Insufficient |
   | v3 | Sparsity losses (L1, TV) | Made peaks sparser | L1 IS the cause of leakage, not the cure |
   | v4 | Baseline supervision | MAJOR improvement | Only works where ground truth available; masking imperfect |
   | v6 | Gradient-based orthogonality | Modest improvement | Indirect — penalizes derivatives, not values |
   | v7 | Asymmetric baseline loss (9:1) | Reduced over-estimation | Penalizes symptom, not cause |
   | v7 | Element-wise orthogonality | Better mutual exclusivity | Leakage persists at peak/baseline transitions |
   | v7 | Envelope constraint | Prevents baseline excursions | Local minimum wrong in dense regions (elevated) |
   | v7 | Frequency separation | Enforces spectral roles | Dense peaks have low-freq content too |
   | v7 | Softplus non-negativity | Eliminates negative-peak artifacts | Doesn't address root cause |

4. **Why it persists**: All mitigations are regularizers on top of a sparsity-biased formulation. The BEADS optimization inherently assumes sparsity. No amount of loss engineering fully overcomes this when the assumption is violated. Fundamental bias-variance tradeoff.

5. **Broader insight**: Any algorithm-unrolled network that inherits L1 sparsity assumptions will face this in dense regimes. Not specific to BEADS or chromatography. Confirmed by Gharbi et al. (MLSP 2024) and DIRAS+ (Analytical Chemistry, 2025).

6. **The arPLS comparison** (key insight from leakage doc): arPLS has NO sparsity assumption. Its failure mode is "elevation" (baseline too high but smooth), not "leakage" (baseline follows peak shapes). Elevation is less damaging for quantification. A truly adaptive system would learn to behave like arPLS in dense regions and BEADS in sparse regions → signal-adaptive regularization (Ch. 7).

**Fig 6.1**: Dense-peak leakage example showing ground truth vs LBEADS output with visible baseline bulge at peak locations.

### Section 6.2: Training/Inference Length Mismatch (1.5 pages)

1. **The problem**: trained on N=1024, real chromatograms often N=4096 or longer.

2. **What changes with length** — analyze component by component:
   | Component | Length Sensitivity | Reason |
   | BAfilt (A, B matrices) | HIGH | Matrices are N×N; change completely |
   | Learned λ values | MODERATE | Optimal regularization may differ |
   | Pointwise shrinkage | LOW | Element-wise, length-independent |
   | TV penalty | LOW | Local differences |
   | Frequency separation | MODERATE | FFT bin width changes |

3. **Distribution shift**: longer signals have different spectral properties, more peaks per signal, different baseline shapes, different noise characteristics.

4. **Potential fixes**:
   - Train on variable lengths: N ∈ {512, 1024, 2048, 4096}
   - Sliding-window inference: process overlapping 1024-sample windows, stitch results
   - Convolutional architecture: inherently length-invariant
   - Fine-tune on target length

**Fig 6.2**: Same model applied to N=1024 vs N=4096 — show performance difference (if data available, otherwise describe expected degradation).

### Section 6.3: Softplus vs. Hard Threshold — Train/Inference Mismatch (1 page)

1. **Setup**: softplus(x, β=5) used for non-negativity during training. softplus(0) = ln(2)/β ≈ 0.139 → not exactly zero.

2. **The mismatch**:
   - Training: smooth gradients through zero-crossing enable learning. Model learns to output slightly negative values, relying on softplus to map them near zero.
   - Inference: near-zero regions become small positive values instead of true zeros. Creates a positive floor.
   - Measured magnitude: ~0.000016 minimum value (from verification) — small but nonzero.

3. **Broader lesson**: Any algorithm-unrolled network using approximate activation functions faces this. Differentiable relaxations (softplus, smooth L1, sigmoid approximations to step functions) introduce train-time smoothness that doesn't match inference-time behavior. The community should be aware: learned proximal operators face this whenever the activation is an approximation.

4. **Potential fixes**: straight-through estimator during training (exact threshold at inference), learnable β parameter, post-training thresholding.

### Section 6.4: The CG Variant's Failure (1.5 pages)
**ANALYTICAL CONTENT EXAMINERS VALUE — this is a contribution in itself.**

1. **The original design**: Conjugate Gradient for f-update. Run K_inner CG steps inside each BEADSLayer. Motivation: more accurate per-layer solution → should converge faster.

2. **Why it failed** — rigorous gradient analysis:
   - Backpropagation must flow through: K_inner CG iterations × K_outer unrolled layers
   - Example: 10 CG steps × 8 layers = 80 sequential operations in the computation graph
   - Vanishing gradients: if each operation has gradient scaling ~0.95, then gradient at layer 1 = 0.95^80 ≈ 0.017 of gradient at layer 8 → 98% loss
   - Early layers receive negligible gradients → parameters barely update → no learning in early layers
   - CG convergence criterion: ||residual||₂ < tolerance. This is a HARD threshold — not differentiable. Creates zero-gradient regions.
   - Different samples may converge in different numbers of CG steps → ragged computation graph → inconsistent gradient paths

3. **Empirical evidence**: v6 phases P1, P2, P3 experimented with CG. Gradient magnitudes per layer showed exponential decay. Parameters in early layers remained near initialization.

4. **The ISTA fix**: Replace K_inner CG steps with 1 proximal gradient step per layer.
   - Backprop depth: 1 × 8 = 8 operations
   - Gradients flow cleanly through all layers
   - Learned step_size compensates: the network learns how much progress to make per layer
   - Trade: per-layer accuracy for end-to-end trainability

5. **Generalizable lesson**: Not all iterative algorithms unroll equally well. The inner solver's gradient properties determine whether the unrolled network is trainable. Design principle: prefer simple, differentiable operations per layer over accurate but gradient-blocking operations.

6. **Connection to theory**: implicit differentiation (Bai et al., 2019) offers an alternative — compute gradients through the fixed-point equation rather than through the unrolled iterations. This avoids the depth problem but requires convergence.

**Fig 6.3**: Gradient magnitude per layer — CG variant (exponential decay curve) vs ISTA variant (stable or slowly decaying). If actual gradient data is available, plot it. Otherwise, create a schematic illustration.

### Section 6.5: fc Non-Learnability (1 page)

1. **The problem**: fc controls Butterworth filter cutoff in BAfilt. BAfilt uses `scipy.signal.butter` → NumPy/SciPy operations OUTSIDE PyTorch autograd. ∂loss/∂fc = 0.

2. **Consequence**: fc = 0.006 fixed for all experiments. This means the baseline/peak frequency boundary cannot adapt:
   - Signals with different baseline drift frequencies may need different fc
   - The model cannot specialize fc per layer (the way it does with λ values)
   - This limits generalization to signals with different spectral characteristics

3. **Why fundamental**: Butterworth filter design involves transcendental functions (computing poles/zeros in the complex s-plane, bilinear transform to z-plane). Even reimplementing in PyTorch, differentiating through these operations is nontrivial.

4. **Potential solutions** (connect to Ch. 7):
   - Differentiable IIR filter design: parameterize filter coefficients directly in PyTorch
   - Replace Butterworth with learnable 1D convolution kernels (trade principled structure for learnability)
   - Surrogate gradients: finite-difference approximation of ∂loss/∂fc
   - Meta-learning: separate network predicts fc from signal features
   - Differentiable DSP libraries (torchaudio filter operations)

## WRITING TONE GUIDANCE

**DO write**: "We observe degraded performance in dense-peak regions. Analysis reveals this stems from the L1 sparsity assumption inherent to the BEADS formulation, which presumes that peaks occupy a small fraction of the signal. When this assumption is violated..."

**DO NOT write**: "Unfortunately, our method fails in dense-peak regions." or "A limitation of our approach is that..."

**For each section, follow the structure**:
1. State the observation clearly (WHAT goes wrong)
2. Explain the mechanism (WHY it happens — mathematically where possible)
3. Describe mitigation attempts (WHAT you tried)
4. Explain why the limitation persists (fundamental vs fixable)
5. Extract a generalizable lesson (VALUE to the community)

**These 5 failure modes are thesis STRENGTHS, not weaknesses.** Examiners value honest, analytical treatment of limitations far more than inflated success claims.
