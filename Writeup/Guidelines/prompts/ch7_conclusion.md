# PROMPT: Write Chapter 7 — Conclusion & Future Work (3–4 pages)

## ROLE
You are an expert academic writer producing the conclusion chapter of an NYU Tandon MS thesis. Write with confidence — this is a real contribution. Keep it concise (3–4 pages max). LaTeX output.

## OUTPUT
Produce the full LaTeX content for `conclusions/conclusions.tex`. Begin with `\chapter{Conclusion and Future Work}`. Contains Sections 7.1–7.3.

## SOURCE FILES TO READ

**Primary — skim for summary material:**
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Sections VII (evolution story) and II (Chapter 7 spec)
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — Sections 3 and 5 for future work directions (learned proximal operators, Conformer, DEQ, Mamba, signal-adaptive params, ADMM reformulation, WaveNet dilated convolutions, peak-aware MAE)
- `Writeup/Guidelines/baseline-leakage/resources.md` — references for future work citations

**For accuracy — verify claims against:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — architecture details to summarize accurately
- `Implementations/7. LBEADS_NETv7 [Stronger]/train.py` — training details to summarize accurately

## SECTION SPECIFICATIONS

### Section 7.1: Summary of Contributions (1.5 pages)

Restate the four contributions clearly. One paragraph per contribution.

**Contribution 1: First Algorithm-Unrolled Network for Baseline Correction**
- LBEADS-NET converts K iterations of the BEADS algorithm into K=8 trainable neural network layers
- Each BEADSLayer has learnable λ₀, λ₁, λ₂, r (log-parameterized for positivity)
- ISTA-style splitting ensures gradient flow through all layers
- Learned parameters specialize across layers: early layers perform coarse separation, later layers refine
- This demonstrates that algorithm unrolling is viable for baseline correction

**Contribution 2: Multi-Stage Curriculum Training with Composite Loss**
- 11-term composite loss covering reconstruction, sparsity, baseline supervision, orthogonality, asymmetry, envelope constraint, frequency separation
- Three-stage curriculum (MSE → structured → full loss) enables stable optimization of this complex loss landscape
- Intermediate supervision at every unrolled layer guides early-stage learning
- This training strategy is general and applicable to other unrolled networks with complex loss functions

**Contribution 3: Hybrid Inference Pipeline**
- LBEADS-NET output serves as initialization for optional classical BEADS refinement
- Quality-scored stage selection automatically picks the best output
- Demonstrates that learned initialization + classical refinement > either method alone
- Practical robustness for real chromatographic data where training distribution doesn't perfectly match

**Contribution 4: Systematic Analysis of Failure Modes**
- CG variant failure: gradient vanishing through nested iterations → design guidance for unrolled inner solvers
- fc non-learnability: autograd boundary at scipy → need differentiable DSP
- Baseline leakage: sparsity assumption limitation in dense regimes → fundamental L1 bias analysis
- Train/inference length mismatch: distribution shift for varying signal lengths
- Softplus approximation artifacts: smooth relaxation train/test mismatch
- These insights extend beyond BEADS — they are relevant to anyone designing algorithm-unrolled networks

### Section 7.2: Generalization Argument (0.5 pages)

- The unrolling methodology is NOT specific to BEADS or chromatography
- Any iterative signal decomposition algorithm with clear per-iteration structure is a candidate for unrolling
- LBEADS-NET demonstrates the template:
  1. Choose a well-structured iterative algorithm with interpretable per-iteration parameters
  2. Map each iteration to a trainable layer with learnable parameters
  3. Parameterize parameters for positivity (log-space) and gradient flow (ISTA-style splitting)
  4. Design composite loss function with domain knowledge (sparsity, smoothness, supervision)
  5. Use staged curriculum to manage loss complexity
  6. Add hybrid inference for practical robustness
- Candidate algorithms: arPLS → L-arPLS-NET, ADMM-based decomposition, iterative morphological operations, any optimization-based signal processing pipeline

### Section 7.3: Future Directions (1.5–2 pages)

Present 7–9 concrete, well-cited future directions. Each should be 1 paragraph with: what it is, why it would help, and a reference.

1. **Learned Proximal Operators**: Replace fixed soft-thresholding with small 1D CNNs (3 layers, 32 channels, residual structure). Follows ISTA-Net+ (Zhang & Ghanem, CVPR 2018). Hybrid ISTA (Zheng et al., TPAMI 2022) proved convergence guarantees are maintained. Key advantage: can learn to NOT shrink in dense-peak regions, directly addressing the leakage limitation.

2. **Making fc Differentiable**: Differentiable IIR filter design — parameterize Butterworth coefficients directly in PyTorch. Alternatively, replace Butterworth with learnable 1D convolution kernels. Would make the network fully learnable with no fixed hyperparameters except architecture depth.

3. **Signal-Adaptive Parameter Prediction**: Hypernetwork/encoder that predicts per-signal adjustments to λ₀, λ₁, λ₂ from input signal features. Follows ISTA-Net++ (You et al., TIP 2021). DIRAS+ (Analytical Chemistry, 2025) demonstrated CNN+XGBoost for per-spectrum parameter prediction. Would allow the network to relax sparsity in dense regions automatically.

4. **ADMM Reformulation**: Unroll ADMM instead of MM/ISTA. ADMM variable splitting provides natural separation between peak and baseline estimation substeps. ADMM-Net (Yang et al., NeurIPS 2016) showed piecewise-linear shrinkage is more flexible than soft-thresholding.

5. **Conformer-Based Architecture**: Replace proximal operator with Conformer blocks (Gulati et al., Interspeech 2020) combining depthwise convolution (local peak structure) with self-attention (global baseline trends). Linear attention (DF-Conformer) for O(N) complexity.

6. **Deep Equilibrium Model (DEQ)**: Convert LBEADS-NET to DEQ formulation (Bai et al., NeurIPS 2019). Iterate until convergence per signal — dense-peak signals that need more iterations get them automatically. O(1) memory via implicit differentiation. MsDC-DEQ-Net demonstrated for compressive sensing.

7. **State Space Models (Mamba)**: Linear O(N) complexity with data-dependent selection (Gu & Dao, 2023). No published application to chromatography baseline correction — novel research direction. Selection mechanism could learn to distinguish peak from baseline regions.

8. **Extension to Other Signal Types**: Raman spectroscopy (similar baseline problem, different signal characteristics), mass spectrometry (chemical noise baseline), ECG baseline wander correction (quasi-periodic). The synthetic data generator could be adapted per domain. Cross-domain transfer learning experiments.

9. **Variable-Length Training and Inference**: Train on N ∈ {512, 1024, 2048, 4096}. Sliding-window inference for arbitrary-length signals. Would address the train/inference length mismatch.

## STYLE CONSTRAINTS
- Confident but measured tone
- Do NOT apologize for limitations (that was Chapter 6)
- Do NOT introduce new results or analysis
- Do NOT repeat method details — reference chapters
- Keep to 3–4 pages total — this should be crisp
- Every future direction needs at least one citation
- Structure: 7.1 = confident summary, 7.2 = vision, 7.3 = concrete roadmap
