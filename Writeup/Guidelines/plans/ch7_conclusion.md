# Chapter 7 — Conclusion & Future Work (3–4 pages)

File: `conclusions/conclusions.tex`

---

## Section 7.1: Summary of Contributions (1.5 pages)

### What to write

Restate contributions crisply — one paragraph per contribution:

**1. First Algorithm-Unrolled Network for Baseline Correction**
- LBEADS-NET converts the iterative BEADS algorithm into a K-layer trainable architecture
- Each BEADSLayer performs one BEADS iteration with learnable lambda_0, lambda_1, lambda_2, r
- Log-parameterization ensures positivity; ISTA-style splitting ensures gradient flow
- K=8 layers with per-layer parameter specialization

**2. Multi-Stage Curriculum Training with Composite Loss**
- 11-term composite loss function covering reconstruction, sparsity, baseline supervision, orthogonality, asymmetry, envelope, frequency separation
- Three-stage curriculum (MSE → structured → full loss) enables stable optimization
- Intermediate supervision at every unrolled layer guides early-stage learning
- This training strategy is general — applicable to other unrolled networks with complex losses

**3. Hybrid Inference Pipeline**
- LBEADS-NET output serves as initialization for optional classical BEADS refinement
- Quality-scored stage selection automatically picks the best output
- Bridges learned and classical methods — practical robustness for real data
- Demonstrates that learned initialization + classical refinement > either alone

**4. Systematic Analysis of Failure Modes**
- CG variant failure: gradient vanishing through nested iterations → design guidance for inner solvers
- fc non-learnability: autograd boundary at scipy → need differentiable DSP
- Baseline leakage: sparsity assumption limitation in dense regimes → L1 bias analysis
- Train/inference length mismatch: distribution shift for signal properties
- Softplus train/test mismatch: smooth relaxation artifacts
- These insights are valuable beyond BEADS — applicable to the algorithm unrolling community

### Key details to weave in
- v0→v7 evolution demonstrates systematic engineering of the solution
- Each version addressed a specific failure mode from the previous
- The final system (v7) incorporates lessons from all iterations

---

## Section 7.2: Generalization Argument (0.5 pages)

### What to write

- The unrolling methodology is NOT specific to BEADS or chromatography
- Any iterative signal decomposition algorithm with clear per-iteration structure is a candidate:
  - Identify per-iteration parameters → make learnable
  - Design domain-specific loss function → enforce decomposition properties
  - Use curriculum training → stabilize complex loss optimization
- LBEADS-NET demonstrates the template:
  1. Choose a well-structured iterative algorithm
  2. Map each iteration to a trainable layer
  3. Parameterize algorithm parameters in learnable form (log-space for positivity)
  4. Design composite loss with domain knowledge
  5. Use staged training to manage loss complexity
  6. Add hybrid inference for practical robustness

- Candidate algorithms for this template:
  - arPLS → LarPLS-NET
  - ADMM-based decomposition → LADMM-NET
  - Iterative morphological operations
  - Any optimization-based signal processing pipeline

---

## Section 7.3: Future Directions (1.5–2 pages)

### What to write

**1. Learned Proximal Operators**
- Current: hand-designed shrinkage/thresholding operators
- Future: replace with small learned neural networks (2-3 layer MLPs)
- Connection to deep equilibrium models (DEQ): find fixed point of learned operator
- Would allow richer per-layer transformations while maintaining unrolled structure
- Reference: DEQ literature (Bai et al., 2019)

**2. Making fc Differentiable**
- Current: fc fixed at 0.006, outside autograd
- Option A: Differentiable IIR filter design — parameterize poles/zeros in PyTorch
- Option B: Replace butterworth with learnable 1D convolution (trade structure for learnability)
- Option C: Frequency-domain parameterization — learn filter response directly
- Impact: would make the network fully learnable, no hyperparameters except architecture

**3. Signal-Adaptive Parameter Prediction**
- Current: learned parameters are fixed after training (same for all signals)
- Future: condition parameters on input signal features
  - Hypernetwork: small network predicts lambda_0, lambda_1, etc. from signal statistics
  - Meta-learning: MAML-style adaptation to new signal types with few examples
- Would handle the "different signals need different parameters" problem directly

**4. ADMM Reformulation**
- Current: MM/ISTA-based unrolling of BEADS
- Alternative: reformulate BEADS as ADMM problem, then unroll ADMM
- ADMM advantages: natural handling of equality constraints, potentially better convergence
- ADMM unrolling is well-studied → can leverage existing theory
- Reference: your Optimization Notes/ADMM.pdf

**5. Alternative Architectures**
- Conformer: attention + convolution for sequence processing
  - Attention captures long-range baseline dependencies
  - Convolution handles local peak structure
- Wave-U-Net: multi-resolution processing for baseline estimation
  - Encoder captures baseline at coarse resolution
  - Decoder refines peak details at fine resolution
- These abandon algorithm unrolling but may offer better performance

**6. Extension to Other Signal Types**
- Raman spectroscopy: similar baseline correction problem, different signal characteristics
- Mass spectrometry: baseline from chemical noise, similar peak structure
- ECG baseline wander correction: quasi-periodic baseline, different from chromatographic drift
- The synthetic data generator could be adapted for each domain
- Cross-domain transfer learning: train on one, fine-tune on another

**7. Variable-Length Training**
- Train on mixed signal lengths: N ∈ {512, 1024, 2048, 4096}
- Sliding-window inference for arbitrary-length signals
- Would address the train/inference length mismatch (Section 6.2)

---

## Writing Guidance

### Tone
- Confident but measured. You've made real contributions — state them clearly.
- Don't oversell: "We present the first algorithm-unrolled network for baseline correction" is factual and strong.
- Future work should be concrete and actionable, not vague "more work is needed."

### What NOT to write
- Don't apologize for limitations (those are in Ch. 6)
- Don't introduce new results or analysis
- Don't repeat method details — reference chapters
- Don't make the conclusion longer than 4 pages

### Structure
- 7.1 should feel like a confident summary — "we did X, Y, Z"
- 7.2 should feel like a vision — "this approach generalizes"
- 7.3 should feel like a roadmap — "here's what comes next"
