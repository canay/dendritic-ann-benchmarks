# Dataset Access and Provenance

This benchmark uses standard public image-classification datasets.

## Included in the repository

- Archived experiment outputs under `runs/`
- Dataset-loading code under `src/`
- Download-capable benchmark scripts

## Not included in GitHub version control

- The local raw dataset cache under `data/`

The cache is excluded because the source datasets are already distributed by their original maintainers and can be re-downloaded.

## Datasets used in the manuscript

### FashionMNIST

- Purpose in this study: grayscale fashion-item classification benchmark
- Access path in code: downloaded through `torchvision.datasets.FashionMNIST`
- Official documentation: `https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html`
- Original project page: `https://github.com/zalandoresearch/fashion-mnist`

### KMNIST

- Purpose in this study: Kuzushiji character classification benchmark
- Access path in code: downloaded through the repository data-loading logic
- Official dataset page: `https://codh.rois.ac.jp/kmnist/`

### CIFAR-10

- Purpose in this study: color object classification benchmark
- Access path in code: downloaded through `torchvision.datasets.CIFAR10`
- Official dataset page: `https://www.cs.toronto.edu/~kriz/cifar.html`

## Reviewer-friendly quick start

From the repository root:

```text
cd dann_benchmark
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

If the local cache is missing, the dataset loader will download the required files into `dann_benchmark/data/`.

## Notes for manuscript readers

- The manuscript's primary tables are derived from archived history files, not from the raw dataset cache.
- Reviewers who inspect the repository can access dataset sources from this document and reproduce downloads through the benchmark code.
