# Reproducibility Guide

This document describes the canonical evidence path for the DANN manuscript package.

Intended public repository: `https://github.com/canay/dendritic-ann-benchmarks`

## Canonical result folders

The manuscript draws its archived benchmark evidence from `dann_benchmark/runs/`.

Main accuracy folders:

- `runs_fashion_full/fashionmnist`
- `runs_kmnist_full/kmnist`
- `runs_fashion_low02/fashionmnist`
- `runs_fashion_low01/fashionmnist`
- `runs_cifar_full/cifar10`
- `runs_cifar_low02/cifar10`

Timing folders:

- `timing_fashion_full/fashionmnist`
- `timing_cifar_full/cifar10`

## What the manuscript reports

- The main manuscript tables do **not** use the historical `best_test_acc` export as the primary metric.
- Instead, the derived manuscript tables reconstruct **test accuracy at the best validation epoch** from per-epoch history files.
- The reduced-data folders are treated as **archived reduced-dataset diagnostics**, not as strict fixed-test low-training-data evidence.
- Wall-clock timing claims come only from the dedicated CPU timing folders.

## Environment

Benchmark dependencies are listed in `dann_benchmark/requirements.txt`:

- `torch>=2.2`
- `torchvision>=0.17`
- `pandas>=2.0`
- `numpy>=1.26`
- `PyYAML>=6.0`

The local raw dataset cache in `dann_benchmark/data/` is not intended for version control. Reviewers should either let the benchmark code download the datasets automatically or follow `dann_benchmark/DATASETS.md`.

## Rebuild derived manuscript assets

From the repository root:

```text
python paper_package/build_manuscript_assets.py
```

This step rebuilds the derived CSV summaries and manuscript figures under `paper_package/derived/` and `paper_package/manuscript/figures/`.

## Recompile the manuscript

From `paper_package/manuscript/`:

```text
pdflatex -interaction=nonstopmode dann_manuscript.tex
bibtex dann_manuscript
pdflatex -interaction=nonstopmode dann_manuscript.tex
pdflatex -interaction=nonstopmode dann_manuscript.tex
```

The active manuscript class file is the official Elsevier `elsarticle.cls` documented in `paper_package/manuscript/ELSEVIER_TEMPLATE_PROVENANCE.md`.

## Scope notes

- The archived accuracy runs were completed across mixed hardware contexts, so runtime interpretation should come from the dedicated CPU timing folders only.
- The archived reduced-data folders predate the repository-side separation of train-subset and test-subset logic; they are therefore retained as diagnostics.
- The archived outputs remain useful for the current manuscript because the main reported metric is reconstructed from the history files rather than copied from the historical summary exports.
