# Reproducibility Guide

This document describes the canonical evidence path of the public repository.

Repository URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Study runs

The evidence of the study is under `runs/`: the families run (F1 to F9, 1,615 training runs), the exploratory
spatial-head CNN arm, the GPU and CPU timing runs, the extension (D1 and D2), the budget arm A18 and the randomness
arm A20. `README.md` lists every run folder, the statistical analysis manifest in `analysis/`, and the commands that
re-aggregate the families run from its public archive and rebuild the result tables and figures with the scripts in
`figures/`. `SHA256SUMS` at the repository root covers every file, and `PUBLIC_FILE_PROVENANCE.csv` records the source
and the public hash of every file under `runs/`, `analysis/` and `figures/`.

## Initial grid

The sections below describe the archived package of the initial 10-seed grid under `dann_benchmark/`, from which the
study started; its paired comparisons are in `stats_outputs/`.

## Canonical result folders

The archived benchmark evidence of the initial grid is stored under `dann_benchmark/runs/`.

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

## Reproduction scope

- Accuracy runs can be regenerated with `dann_benchmark/benchmark.py`.
- Timing runs can be regenerated with `dann_benchmark/timing_benchmark.py`.
- `stats_outputs/paired_tests_validation_selected.csv` is a derived summary of the archived histories.

## Datasets

See `dann_benchmark/DATASETS.md` for the dataset providers and the download behaviour of the benchmark code.
