# Reproducibility Guide

This document describes the canonical evidence path for the public DANN benchmark repository.

Repository URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Canonical result folders

The archived benchmark evidence is stored under `dann_benchmark/runs/`.

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

## What these archived outputs support

- The full-data folders support controlled comparisons among DANN variants, parameter-matched dense baselines, a naive branching control, and a larger dense reference.
- The reduced-data folders are preserved as archived reduced-dataset diagnostics.
- Timing claims should be interpreted only from the dedicated CPU timing folders.

## Environment

Benchmark dependencies are listed in `dann_benchmark/requirements.txt`:

- `torch>=2.2`
- `torchvision>=0.17`
- `pandas>=2.0`
- `numpy>=1.26`
- `PyYAML>=6.0`

The local raw dataset cache in `dann_benchmark/data/` is not intended for version control. Reviewers can let the benchmark code download datasets automatically or follow `dann_benchmark/DATASETS.md`.

## Output structure

Each archived accuracy folder contains:

- `summary_by_seed.csv`
- `summary_by_model.csv`
- `run_config.json`
- `decision_report.json`
- `histories/*.csv`

Each timing folder contains:

- `timing_summary_by_seed.csv`
- `timing_summary_by_model.csv`
- `timing_run_config.json`

## Scope notes

- The archived accuracy runs were completed across mixed hardware contexts, so runtime interpretation should come only from the dedicated CPU timing folders.
- The archived reduced-data folders predate the repository-side separation of train-subset and test-subset logic, so they are best treated as diagnostics rather than as strict fixed-test low-training-data experiments.
- The supporting statistical outputs in `stats_outputs/` are derived from the archived run files in this repository.
