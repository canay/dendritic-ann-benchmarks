# Dendritic ANN Benchmark

This folder contains the core public benchmark package.

## Included

- `src/`
  Core data loading, model definitions, sampling logic, and train/eval utilities.
- `benchmark.py`
  Main accuracy benchmark runner.
- `timing_benchmark.py`
  Dedicated CPU timing runner.
- `requirements.txt`
  Python dependency list.
- `runs/`
  Archived accuracy and timing outputs.
- `DATASETS.md`
  Dataset provenance and download guidance.

## Excluded from version control

- `data/`
  Local dataset cache downloaded from the original dataset providers.

## Archived run set

The public repository preserves these archived folders under `runs/`:

- `runs_fashion_full`
- `runs_kmnist_full`
- `runs_cifar_full`
- `runs_fashion_low02`
- `runs_fashion_low01`
- `runs_cifar_low02`
- `timing_fashion_full`
- `timing_cifar_full`

The low-data folders are retained as reduced-dataset diagnostics. Runtime interpretation should come only from the dedicated timing folders.

## Smoke test

```text
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

## Output files

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
