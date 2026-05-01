# Dendritic Artificial Neural Networks Benchmark Package

This repository contains the code, archived experiment outputs, and supporting statistical result files for a controlled benchmark study of dendritic artificial neural networks (DANNs).

Repository URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Contact

- Ozkan Canay
- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: `canay@sakarya.edu.tr`
- ORCID: `0000-0001-7539-6001`
- Web: `https://canay.sakarya.edu.tr/`

## Included in this public repository

- `dann_benchmark/`
  Benchmark code, run scripts, timing code, and archived result folders.
- `stats_outputs/`
  Supporting statistical comparison outputs derived from the archived run histories.
- `REPRODUCIBILITY.md`
  Notes on the canonical run folders, metric interpretation, and dataset access workflow.

## Not included in this public repository

- The manuscript source and submission package.
- The local raw dataset cache under `dann_benchmark/data/`.
- Internal archive and planning materials.

## Benchmark scope

The archived benchmark centers on a controlled shallow-classification comparison with:

- DANN variants: `DANN_LRF`, `DANN_RANDOM`, `DANN_GRF`
- Controls: `NAIVE_BRANCH`, `MLP_PARAM`, `VANN_SAME`
- Full-data tasks: FashionMNIST, KMNIST, CIFAR-10
- Archived reduced-dataset diagnostics
- Separate CPU-only timing runs

This repository does not present the models as state of the art image classifiers. Its emphasis is controlled comparison, parameter efficiency, and the incremental effect of dendritic nonlinearity beyond branching alone.

## Data access

- Archived benchmark outputs are versioned under `dann_benchmark/runs/`.
- Dataset provenance and download notes are documented in `dann_benchmark/DATASETS.md`.
- The local dataset cache under `dann_benchmark/data/` is intentionally excluded from GitHub because these datasets are distributed by their original providers and can be re-downloaded through the benchmark code.

## Recommended reading order

1. `dann_benchmark/README.md`
2. `dann_benchmark/DATASETS.md`
3. `REPRODUCIBILITY.md`
4. `stats_outputs/README.md`

## Quick start

From the repository root:

```text
cd dann_benchmark
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

If the local cache is missing, the data-loading logic will download the required dataset files into `dann_benchmark/data/`.
