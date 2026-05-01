# Dendritic Artificial Neural Networks Benchmark Package

This repository contains the minimal public package for a controlled benchmark study of dendritic artificial neural networks (DANNs).

Repository URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Contact

- Ozkan Canay
- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: `canay@sakarya.edu.tr`
- ORCID: `0000-0001-7539-6001`
- Web: `https://canay.sakarya.edu.tr/`

## Public contents

- `dann_benchmark/`
  Benchmark code, dataset-access notes, and archived accuracy and timing outputs.
- `stats_outputs/`
  Supporting statistical summary derived from the archived benchmark histories.
- `REPRODUCIBILITY.md`
  Short guide to the canonical result folders and reproduction scope.

## What is intentionally excluded

- Manuscript source files and submission materials
- Internal planning or archive materials
- Local dataset cache files under `dann_benchmark/data/`
- Legacy helper scripts that are not needed to inspect or rerun the archived benchmark

## Benchmark scope

The public package preserves a controlled comparison among:

- DANN variants: `DANN_LRF`, `DANN_RANDOM`, `DANN_GRF`
- Controls: `NAIVE_BRANCH`, `MLP_PARAM`, `VANN_SAME`
- Full-data tasks: FashionMNIST, KMNIST, CIFAR-10
- Archived reduced-dataset diagnostics
- Separate CPU-only timing runs

The focus is controlled architectural comparison, not state-of-the-art image classification.

## Quick start

```text
cd dann_benchmark
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

If the local cache is missing, the benchmark code will download the required dataset files into `dann_benchmark/data/`.

## Start here

1. `dann_benchmark/README.md`
2. `dann_benchmark/DATASETS.md`
3. `REPRODUCIBILITY.md`
4. `stats_outputs/README.md`
