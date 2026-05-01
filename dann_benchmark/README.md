# Dendritic ANN Benchmark Package

This folder contains the code and archived results for the controlled DANN benchmark used in the manuscript.

Intended public repository: `https://github.com/canay/dendritic-ann-benchmarks`

## Model families

- `dann_lrf`
  Main candidate method with fixed local receptive-field routing.
- `dann_random`
  DANN variant with random feature sampling.
- `dann_grf`
  DANN variant with soma-level grouped receptive fields.
- `naive_branch`
  Branch-structured control without dendrite-level nonlinearity.
- `mlp_param`
  Approximately parameter-matched dense MLP baseline.
- `vann_same`
  Larger dense reference model.

## Folder structure

- `src/`
  Core data loading, model, sampling, and training logic.
- `runs/`
  Archived accuracy and timing outputs used by the manuscript.
- `configs/`
  Supporting benchmark configuration files.
- `benchmark.py`
  Main benchmark runner for accuracy experiments.
- `timing_benchmark.py`
  Dedicated CPU timing runner.
- `RERUN_NOTES.md`
  Notes for optional reruns and fixed-test low-data follow-up.

## Historical archived run set

The manuscript currently relies on the archived folders under `runs/`:

- `runs_fashion_full`
- `runs_kmnist_full`
- `runs_cifar_full`
- `runs_fashion_low02`
- `runs_fashion_low01`
- `runs_cifar_low02`
- `timing_fashion_full`
- `timing_cifar_full`

The low-data folders are preserved as reduced-dataset diagnostics. They are not treated as strict fixed-test low-training-data experiments because the historical subset logic also affected the test loader.

## Environment

Install compatible versions of `torch` and `torchvision` for your platform, then install the remaining Python packages:

```text
pip install pandas numpy pyyaml
```

## Datasets

- Dataset download and provenance notes are documented in `DATASETS.md`.
- The benchmark code can download standard datasets automatically when the local cache is missing.
- The local cache under `data/` is not intended for GitHub version control.

## Example smoke test

```text
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

## Canonical output files

Each archived accuracy folder contains:

- `summary_by_seed.csv`
- `summary_by_model.csv`
- `run_config.json`
- `decision_report.json`
- `histories/*.csv`

Each timing folder contains:

- `timing_summary_by_model.csv`
- `timing_run_config.json`
- timing histories or logs where available

## Benchmark interpretation

This package is designed for a controlled architectural question, not for leaderboard chasing:

> Does dendritic computation add measurable value beyond parameter matching and beyond branching alone?

That is why `naive_branch` is a first-class control throughout the package.
