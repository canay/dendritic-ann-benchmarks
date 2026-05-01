# Dendritic Artificial Neural Networks Benchmark and Manuscript Package

This repository package contains the code, archived experiment outputs, manuscript sources, and derived evidence used for a controlled study of dendritic artificial neural networks (DANNs).

## Intended public repository

- Repository name: `dendritic-ann-benchmarks`
- Intended URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Contact

- Ozkan Canay
- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: `canay@sakarya.edu.tr`
- ORCID: `0000-0001-7539-6001`
- Web: `https://canay.sakarya.edu.tr/`

## Canonical folders

- `dann_benchmark/`
  Benchmark code, timing code, run scripts, and archived experiment folders.
- `paper_package/`
  Manuscript source, derived tables and figures, literature notes, and submission materials.
- `stats_outputs/`
  Supporting statistical outputs produced during analysis.
- `archive_materials/`
  Legacy notes, earlier drafts, and process artifacts retained for traceability but not required for the canonical submission package.

## Study scope

The manuscript centers on a controlled shallow-classification benchmark with:

- DANN variants: `DANN_LRF`, `DANN_RANDOM`, `DANN_GRF`
- Controls: `NAIVE_BRANCH`, `MLP_PARAM`, `VANN_SAME`
- Full-data tasks: FashionMNIST, KMNIST, CIFAR-10
- Archived reduced-dataset diagnostics
- Separate CPU-only timing runs

The main paper does not claim state-of-the-art image classification. Its focus is architectural fairness, parameter efficiency, and the incremental value of dendritic nonlinearity beyond branching alone.

## Data access

- Archived benchmark outputs used in the manuscript are versioned under `dann_benchmark/runs/`.
- Standard benchmark datasets are documented in `dann_benchmark/DATASETS.md`.
- The local raw dataset cache under `dann_benchmark/data/` is intentionally excluded from GitHub because these datasets are distributed by their original sources and can be re-downloaded through the documented workflow.

## Recommended reading order

1. `paper_package/manuscript/dann_manuscript.pdf`
2. `paper_package/claim_evidence_matrix.md`
3. `paper_package/safe_claims_and_limitations.md`
4. `paper_package/submission/submission_checklist.md`
5. `REPRODUCIBILITY.md`
6. `dann_benchmark/README.md`

## Rebuild path

1. Recreate or inspect benchmark outputs under `dann_benchmark/runs/`.
2. Rebuild manuscript assets from archived histories:

```text
python paper_package/build_manuscript_assets.py
```

3. Compile the manuscript from `paper_package/manuscript/`:

```text
pdflatex -interaction=nonstopmode dann_manuscript.tex
bibtex dann_manuscript
pdflatex -interaction=nonstopmode dann_manuscript.tex
pdflatex -interaction=nonstopmode dann_manuscript.tex
```

## Notes on package status

- The historical reduced-data folders are preserved as diagnostics, not as strict fixed-test low-training-data evidence.
- The primary manuscript tables use validation-selected test accuracy reconstructed from per-epoch histories.
- Some legacy materials are preserved under `archive_materials/` for project traceability, but the canonical research package is the combination of `dann_benchmark/` and `paper_package/`.
- `REPRODUCIBILITY.md` and `paper_package/submission/repository_release_note.md` can be used directly when preparing the public GitHub release.
- The intended public GitHub home for this package is `https://github.com/canay/dendritic-ann-benchmarks`.
- `GITHUB_PUBLISHING.md` contains the exact repository-creation and push steps.
