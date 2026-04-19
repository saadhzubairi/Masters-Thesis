# Thesis Writing Prompts

Each file in this folder is a self-contained, production-grade prompt for writing one chapter (or section) of the LBEADS-NET thesis. Feed any of these to an LLM along with the referenced source files to produce a high-quality first draft.

## Usage

1. Open the prompt file for the chapter you want to write
2. Gather the source files listed in the "SOURCE FILES" section of the prompt
3. Feed the prompt + source files to the LLM
4. The LLM produces LaTeX output ready to paste into the corresponding .tex file

## Prompt Order (recommended)

| Order | Prompt File | Chapter | Status |
|-------|-------------|---------|--------|
| 1 | `ch3_method.md` | Proposed Method (15–18 pp) | Core — write first |
| 2 | `ch4_experiments.md` | Experimental Setup (6–8 pp) | Needs method done |
| 3 | `ch5_results.md` | Results & Analysis (12–15 pp) | Needs experiments done |
| 4 | `ch2_background.md` | Background (12–15 pp) | After method is solid |
| 5 | `ch6_discussion.md` | Discussion (5–7 pp) | After results |
| 6 | `ch1_introduction.md` | Introduction (5–7 pp) | Second-to-last |
| 7 | `ch7_conclusion.md` | Conclusion (3–4 pp) | Last |
| 8 | `abstract.md` | Abstract (~300 words) | Very last |

## Master Reference

See `Writeup/Guidelines/THESIS_STRUCTURE.md` for the unified thesis structure with all cross-references, figure/table inventories, and the complete v0→v7 evolution narrative.
