# PROMPT: Write Chapter 1 — Introduction (5–7 pages)

## ROLE
You are an expert academic writer producing a chapter for an NYU Tandon MS thesis in Mechatronics & Robotics. Write in formal academic English, third person, present tense for established facts and past tense for completed work. LaTeX output using the NYU Tandon thesis template.

## OUTPUT
Produce the full LaTeX content for `introduction/introduction.tex`. The file should begin with `\chapter{Introduction}` and contain Sections 1.1–1.4.

## SOURCE FILES TO READ

**Primary (read these fully):**
- `Implementations/7. LBEADS_NETv7 [Stronger]/README.md` — latest architecture summary
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Section II, Chapter 1 specification
- `Writeup/Guidelines/baseline-leakage/resources.md` — references to cite

**For context (skim):**
- `Implementations/0. BEADS/WithNumpy/beads.py` — classical BEADS to understand the tuning problem (6 parameters: λ₀, λ₁, λ₂, r, fc, d)
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — scan class definitions to accurately describe contributions
- `Literature/Research Papers/BEADS.pdf` — original BEADS paper for citation

**For Figure 1.1:**
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo_chromatogram.py` — generates real chromatogram output; reference an existing output image or describe what the figure should show

## SECTION SPECIFICATIONS

### Section 1.1: The Problem — Baseline Drift in Chromatography (1–1.5 pages)
- Open with 1–2 sentences defining chromatography as a chemical separation technique — do NOT write a chromatography tutorial
- Define baseline drift: low-frequency signal corruption from column bleed, detector aging, temperature gradients, mobile phase composition changes
- State why it matters: corrupts peak area integration → wrong analyte concentrations → unreliable quantitative analysis
- Introduce the signal model y = x + f + w (observed = peaks + baseline + noise) — first appearance, developed fully in Ch. 2
- Include `\begin{figure}` placeholder for Fig 1.1: motivating chromatogram showing visible drift vs corrected baseline. Caption should explain what the reader is seeing.

### Section 1.2: Limitations of Existing Approaches (1–1.5 pages)
- Classical methods require per-signal parameter tuning:
  - BEADS (Ning et al., 2014): needs λ₀, λ₁, λ₂, r, fc, d — 6 parameters with nonlinear interactions
  - arPLS, airPLS, SNIP, AsLS: each has own tuning knobs
  - No single parameter set generalizes across instruments, analytes, or conditions
- Deep learning alternatives (autoencoders, U-Nets): learn from data but are black-box
  - No interpretability of what the network has learned
  - Cannot inspect or validate intermediate representations
- State the gap clearly: no existing method combines learned generalization (adapt to different signals without retuning) with algorithmic interpretability (understand what each parameter means and what each layer does)
- Cite: Ning et al. 2014 (BEADS), Baek et al. 2015 (arPLS), Zhang et al. 2010 (airPLS), Monga et al. 2021 (algorithm unrolling survey), Kensert et al. 2021 (neural baseline correction)

### Section 1.3: Contribution — LBEADS-NET (1.5–2 pages)
- Clear statement: "We propose LBEADS-NET, an algorithm-unrolled neural network that converts the iterative BEADS algorithm into a trainable architecture with learnable parameters per layer."
- Four contributions as a numbered list:
  1. **Algorithm unrolling of BEADS**: each BEADS iteration → one BEADSLayer with learnable λ₀, λ₁, λ₂, r (log-parameterized for positivity). K=8 layers stacked sequentially with per-layer parameter specialization.
  2. **Multi-stage curriculum training with composite loss**: 11-term loss function (reconstruction, sparsity, baseline supervision, orthogonality, asymmetry, envelope, frequency separation). Three-stage curriculum (MSE only → structured losses → full loss) for stable optimization.
  3. **Hybrid inference pipeline**: LBEADS-NET output → optional classical BEADS refinement, with quality-scored stage selection. Bridges learned initialization and classical convergence guarantees.
  4. **Systematic analysis of failure modes**: CG variant gradient vanishing, fc non-learnability at scipy/PyTorch boundary, baseline leakage in dense-peak regions, train/inference length mismatch, softplus approximation artifacts. These provide actionable insights for the algorithm unrolling community.

### Section 1.4: Thesis Outline (0.5 pages)
- One paragraph mapping Ch. 2 (background: signal processing, BEADS, algorithm unrolling) → Ch. 3 (LBEADS-NET architecture, training, loss) → Ch. 4 (experimental setup: data, metrics, baselines) → Ch. 5 (results on synthetic and real data) → Ch. 6 (discussion of limitations and failure modes) → Ch. 7 (conclusion and future work)

## STYLE CONSTRAINTS
- Do NOT use first person ("I"); use "we" or passive voice
- Do NOT include code snippets in this chapter
- Equations: use `\begin{equation}` only for the signal model y = x + f + w; all other math is in later chapters
- Citations: use `\cite{}` with placeholder keys that match what will go in thesis.bib (e.g., `\cite{ning2014beads}`, `\cite{gregor2010lista}`)
- Keep it accessible to a reader who knows signal processing but not chromatography
- Do not over-explain chromatography — just enough context to motivate the problem
- The tone should be confident: this is a real contribution, not a homework exercise
