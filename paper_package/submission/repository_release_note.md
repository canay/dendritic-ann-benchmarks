# Repository Release Note Draft

## Suggested release title

DANN benchmark and manuscript package for controlled dendritic ANN evaluation

## Repository identity

- Suggested repository name: `dendritic-ann-benchmarks`
- Suggested public URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Suggested release summary

This release packages the code, archived experiment outputs, manuscript source, and derived evidence used in a controlled study of dendritic artificial neural networks (DANNs). The repository is organized around three DANN variants (`DANN_LRF`, `DANN_RANDOM`, `DANN_GRF`), three principal controls (`NAIVE_BRANCH`, `MLP_PARAM`, `VANN_SAME`), full-data image-classification benchmarks, archived reduced-dataset diagnostics, and separate CPU timing runs.

## What is included

- Benchmark code under `dann_benchmark/`
- Archived benchmark outputs under `dann_benchmark/runs/`
- Manuscript source and PDF under `paper_package/manuscript/`
- Derived manuscript assets under `paper_package/derived/`
- Submission support files under `paper_package/submission/`
- Project notes and claim constraints under `dann_codex_md_pack/`

## Important interpretation notes

- The manuscript's primary metric is reconstructed as **test accuracy at the best validation epoch** from history files.
- The historical reduced-data folders are retained as **diagnostic evidence**, not as strict fixed-test low-training-data experiments.
- Timing claims come from separate CPU-only timing runs, not from the mixed-hardware accuracy folders.

## Suggested release checklist

- Add the final public repository URL to the manuscript and submission materials.
- Choose an explicit repository license before publishing.
- Confirm the target journal for the first submission.
- Optionally tag a release after the final PDF and public README are frozen.
