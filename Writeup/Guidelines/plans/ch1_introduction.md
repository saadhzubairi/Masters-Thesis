# Chapter 1 — Introduction (5–7 pages)

File: `introduction/introduction.tex`

---

## Section 1.1: The Problem — Baseline Drift in Chromatography

### What to write
- Open with 1–2 sentences defining chromatography (separation technique for chemical mixtures)
- Define baseline drift: low-frequency signal corruption from column bleed, detector aging, temperature gradients, mobile phase composition changes
- Explain why it matters: corrupts peak area integration → wrong analyte concentrations → unreliable quantitative analysis
- Introduce the signal model: y = x + f + w (observed = peaks + baseline + noise) — first appearance, elaborated in Ch. 2

### Key details from your work
- Your synthetic data generator creates realistic drift via polynomial (2-4 degree) + sinusoidal combinations
- Real chromatograms tested in demo_chromatogram.py show significant drift
- Signal length: N=1024 for training, potentially N=4096+ for real data

### Figure
- **Fig 1.1**: Motivating example — a raw chromatogram showing visible drift alongside the "true" corrected baseline. Use output from `demo_chromatogram.py` or generate a clean example from `demo.py`

### Tone
- Accessible to a reader who knows signal processing but not chromatography specifically
- Don't write a chromatography tutorial — just enough context to motivate the problem

---

## Section 1.2: Limitations of Existing Approaches

### What to write
- Classical methods (BEADS, arPLS, airPLS, SNIP, AsLS) require per-signal parameter tuning
  - BEADS needs: lambda_0, lambda_1, lambda_2, r, fc, d — 6 parameters
  - arPLS, airPLS each have their own tuning knobs
  - No single parameter set generalizes across signal types (different analytes, instruments, conditions)
- Deep learning alternatives exist but are black-box — no interpretability of what the network learns
- **Gap**: No method combines learned generalization (adapt to different signals automatically) with algorithmic interpretability (understand what parameters mean)

### Key details from your work
- v0 (classical BEADS) demonstrated the tuning problem — different signals need different fc, lambda values
- v1 showed that making these parameters learnable is viable
- The entire v1→v7 progression is motivated by this gap

### References to cite
- BEADS: Ning et al. (2014)
- arPLS: Baek et al.
- airPLS: Zhang et al.
- Algorithm unrolling survey: Monga et al.

---

## Section 1.3: Contribution — LBEADS-NET

### What to write
- Clear statement: "We propose LBEADS-NET, an algorithm-unrolled neural network that converts the iterative BEADS algorithm into a trainable architecture with learnable parameters per layer."
- Bulleted contributions:

1. **Algorithm unrolling of BEADS into differentiable layers**: Each BEADS iteration becomes a BEADSLayer with learnable log_lambda_0, log_lambda_1, log_lambda_2, log_r (log-parameterized for positivity). K=8 layers stacked sequentially.

2. **Multi-stage curriculum training with composite loss**: Three-stage training (A: MSE only → B: baseline supervision + asymmetric + orthogonality → C: full 11-term composite loss). Prevents early divergence and enables stable optimization.

3. **Hybrid inference pipeline with quality-scored stage selection**: LBEADS-NET output → optional classical BEADS refinement, with automatic quality scoring to select best output stage. Bridges learned and classical methods.

4. **Systematic analysis of failure modes**: CG variant gradient vanishing, fc non-learnability (scipy autograd boundary), baseline leakage in dense-peak regions, train/inference length mismatch, softplus approximation artifacts. These insights are valuable to the algorithm unrolling community.

### Key details from your work
- v1 established contributions (1)
- v3-v4 established contribution (2) — the leakage fix was the critical turning point
- v5-v6 established contribution (3) — hybrid inference for robustness
- v6-v7 together establish contribution (4) — systematic iteration on failure modes
- Contribution (4) is what turns weaknesses into thesis strength — examiners value this

---

## Section 1.4: Thesis Outline

### What to write
- One paragraph mapping chapters 2–7:
  - Ch. 2: Background on chromatogram processing, BEADS algorithm, algorithm unrolling
  - Ch. 3: LBEADS-NET architecture, training, loss design
  - Ch. 4: Experimental setup — synthetic data, metrics, baselines
  - Ch. 5: Results on synthetic and real data
  - Ch. 6: Discussion of limitations and failure modes
  - Ch. 7: Conclusion and future directions

---

## Successes to Highlight
- The algorithm unrolling approach is novel for baseline correction — first of its kind
- Learned parameters specialize across layers (early: coarse, later: refined)
- Hybrid pipeline provides practical robustness
- Systematic failure analysis provides community value

## Failures/Challenges to Acknowledge (briefly, details in Ch. 6)
- Dense-peak baseline leakage not fully solved
- fc remains a hyperparameter
- Train/inference length mismatch exists
