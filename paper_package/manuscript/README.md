# Manuscript Package Notes

- Main manuscript source: `dann_manuscript.tex`
- Compiled PDF: `dann_manuscript.pdf`
- Official Elsevier class in active use: `elsarticle.cls`
- Bibliography style: `elsarticle-num.bst`
- Bibliography databases: `refs.bib` and `refs_expanded.bib`

## Current manuscript status

- The manuscript has been rewritten around an IMRAD-style scientific argument rather than a placeholder skeleton.
- Primary accuracy tables use validation-selected test accuracy reconstructed from per-epoch history files.
- The reduced-data section is explicitly framed as an archived reduced-dataset diagnostic, not as strict fixed-test low-training-data evidence.
- The package includes a 70+ reference bibliography, a recent-studies literature table, generated figures, and the updated author identity block for {\"O}zkan Canay.
- The active class file is the official Elsevier `elsarticle` download documented in `ELSEVIER_TEMPLATE_PROVENANCE.md`.

## Compile

`latexmk` is not available in the current local MiKTeX setup because the required Perl runtime is missing. The manuscript was compiled successfully with the manual sequence below:

```text
pdflatex -interaction=nonstopmode dann_manuscript.tex
bibtex dann_manuscript
pdflatex -interaction=nonstopmode dann_manuscript.tex
pdflatex -interaction=nonstopmode dann_manuscript.tex
```

## Figure and table provenance

- Derived manuscript tables and plots are built from `paper_package/build_manuscript_assets.py`.
- The script reads the archived benchmark histories under `dann_benchmark/runs/` and writes derived CSV outputs into `paper_package/derived/`.
- The manuscript figures used in the PDF are stored under `paper_package/manuscript/figures/`.
- The literature-grounding table is authored directly in `dann_manuscript.tex` and cites entries from `refs_expanded.bib`.

## Remaining submission-time placeholders

- Public repository creation at `https://github.com/canay/dendritic-ann-benchmarks`
- Final target journal choice among the recommended Elsevier options
- Optional journal-specific cover letter and submission metadata

## Submission companions

- `paper_package/submission/highlights.txt`
- `paper_package/submission/cover_letter_neurocomputing.md`
- `paper_package/submission/submission_checklist.md`
- `paper_package/submission/submission_metadata.md`
