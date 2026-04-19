# LBEADS-NET Thesis — Master Writing Plan

## Thesis Title
"LBEADS-NET: Algorithm Unrolling for Learned Baseline Estimation and Denoising in Chromatographic Signals"

## Status of Repository Work

### Implementation Versions Completed (v0–v7)
| Version | Focus | Key Outcome |
|---------|-------|-------------|
| v0 | Classical BEADS reference | NumPy port of Ning et al. algorithm |
| v1 | Basic algorithm unrolling | First BEADSLayer, learnable lambda params |
| v2 | Synthetic data pipeline | Ground truth generation for supervised training |
| v3 | Sparsity-based loss functions | L1, TV, smoothness, non-negativity penalties |
| v4 | Baseline leakage fix | Direct baseline supervision — critical breakthrough |
| v5 | Hybrid inference pipeline | Post-processing + classical BEADS refinement fallback |
| v6 | Robust "Strong" version | CG solver, multi-phase experiments, diagnostics |
| v7 | Final "Stronger" version | 4 new losses, softplus constraint, 3-stage curriculum, 8 layers |

### Key Source Files for Writing
- `Implementations/7. LBEADS_NETv7 [Stronger]/lbeads_net.py` — architecture
- `Implementations/7. LBEADS_NETv7 [Stronger]/train.py` — training + losses + data gen
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo.py` — synthetic demo
- `Implementations/7. LBEADS_NETv7 [Stronger]/demo_chromatogram.py` — real data demo
- `Implementations/7. LBEADS_NETv7 [Stronger]/analysis/` — diagnostics
- `Implementations/0. BEADS/` — classical reference
- `Literature/` — papers and DSP lecture notes
- `Optimization Notes/` — ADMM, gradient descent references

### Writing Order (from blueprint)
1. Chapter 3 — Method (core contribution, write first)
2. Chapter 4 — Experiments
3. Chapter 5 — Results
4. Chapter 2 — Background (after method is solid)
5. Chapter 6 — Discussion
6. Chapter 1 — Introduction (second-to-last)
7. Chapter 7 — Conclusion (last)
8. Abstract (very last)

### Page Budget
| Chapter | Pages | Status |
|---------|-------|--------|
| 1. Introduction | 5–7 | placeholder |
| 2. Background | 12–15 | placeholder |
| 3. Method | 15–18 | needs creation |
| 4. Experiments | 6–8 | needs creation |
| 5. Results | 12–15 | needs creation |
| 6. Discussion | 5–7 | needs creation |
| 7. Conclusion | 3–4 | placeholder |
| **Total** | **58–74** | |

### LaTeX Infrastructure TODO
- Update thesis.tex: title, author, advisor, \input for chs 3-6
- Create directories: method/, experiments/, results/, discussion/
- Populate definitions.tex with math commands
- Populate thesis.bib
- Create figures/ directory

### Individual Chapter Plans
See the following files for detailed per-chapter plans:
- `ch1_introduction.md`
- `ch2_background.md`
- `ch3_method.md`
- `ch4_experiments.md`
- `ch5_results.md`
- `ch6_discussion.md`
- `ch7_conclusion.md`
