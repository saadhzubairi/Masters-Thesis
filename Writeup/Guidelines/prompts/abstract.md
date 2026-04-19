# PROMPT: Write the Abstract (~300 words)

## ROLE
You are an expert academic writer producing the abstract for an NYU Tandon MS thesis. The abstract is the last thing written — it must accurately reflect the complete thesis. ~300 words, single paragraph or two short paragraphs. LaTeX output.

## OUTPUT
Produce the LaTeX content for `abstract.tex`. Replace the existing \lipsum placeholder.

## SOURCE FILES TO READ
- `Writeup/Guidelines/THESIS_STRUCTURE.md` — Full thesis structure for accuracy
- `Writeup/Guidelines/baseline-leakage/peaks-leaking-into-baseline.md` — For accurate characterization of the leakage problem and what was done

## STRUCTURE (follow this order in ~300 words)

1. **Problem** (2 sentences): Baseline drift in chromatographic signals corrupts quantitative analysis. Existing methods require per-signal parameter tuning (classical) or lack interpretability (deep learning).

2. **Gap** (1 sentence): No method combines learned generalization with algorithmic interpretability.

3. **Method** (3–4 sentences): LBEADS-NET = algorithm-unrolled BEADS. K=8 trainable layers with learnable per-layer parameters. 11-term composite loss with three-stage curriculum training. Hybrid inference with classical BEADS refinement and quality-scored stage selection.

4. **Key results** (2–3 sentences): Learned parameters specialize across layers (early=coarse, late=refined). Competitive with manually-tuned classical methods on synthetic data. Hybrid pipeline provides robustness for real chromatographic data.

5. **Analysis** (1–2 sentences): Systematic analysis of five failure modes (CG gradient vanishing, fc non-learnability, dense-peak leakage, length mismatch, softplus artifacts) provides insights for the algorithm unrolling community.

6. **Significance** (1 sentence): LBEADS-NET demonstrates a template for unrolling any iterative signal decomposition algorithm into a trainable architecture.

## STYLE
- No citations in the abstract
- No equations
- Use full term first, then abbreviation: "LBEADS-NET (Learned BEADS Network)"
- Active voice where possible
- Every sentence must carry information — no filler
