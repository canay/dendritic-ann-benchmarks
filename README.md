# Dendritic Artificial Neural Networks Benchmark Package

This repository holds the code, run manifests, configurations, per-run training histories, processed outputs,
verification records and result-table and result-figure scripts of the study *Branching-Controlled Disentanglement
of Dendritic Nonlinearity from Routing and Capacity in Parameter-Matched Image Classification*, together with the
benchmark code and the archived outputs of the initial 10-seed grid from which the study started.

Repository URL: `https://github.com/canay/dendritic-ann-benchmarks`

## Contact

- Özkan Canay
- Department of Information Systems and Technologies
- Faculty of Computer and Information Sciences
- Sakarya University
- Email: `canay@sakarya.edu.tr`
- ORCID: `0000-0001-7539-6001`
- Web: `https://canay.sakarya.edu.tr/`

## Contents

| Path | Content |
|---|---|
| `runs/2026-10-03_families_f1_f9/` | Families F1 to F9 (main grid, reference priors, sensitivity, randomness sources, shuffled pixels, channel-aware routing, dendrite slope, optimization fairness, convolutional stem): 1,615 training runs |
| `runs/2026-10-04_spatial_head_cnn/` | Exploratory spatial-head CNN arm (A12): 60 runs plus 6 repeats of runs of the families |
| `runs/2026-10-04_timing_gpu/` | F10 timing on the GPU (51 measurement units); its `processed_outputs/` hold the combined CPU and GPU timing analysis |
| `runs/2026-10-04_timing_cpu/` | F10 timing on the CPU (51 measurement units) |
| `runs/2026-10-04_extension_d1_d2/` | Extension: the minimal Transformer front end with three heads (D1) and flattened CIFAR-100 (D2), each at the 30-epoch protocol and in a 100-epoch budget arm: 449 training runs, 9 of them repeats of runs of the families |
| `runs/2026-10-05_budget_arm_a18/` | Budget arm A18: the full-data F1 contrasts of DANN-LRF, Naive-Branch, MLP-Param and DANN-RANDOM continued to 100 epochs, 240 training runs whose first 30 epochs reproduce the stored histories of the families run |
| `runs/2026-10-08_randomness_arm_a20/` | Randomness arm A20: the data seed of F4 replaced by a split seed and a minibatch-order seed, each varied over ten values with DANN-LRF and Naive-Branch on FashionMNIST: 38 training runs plus 8 repeats of runs of the families, whose histories and those of the two value-0 runs reproduce the F4 histories |
| `analysis/R1_ANALYSIS_MANIFEST.json` | Statistical analysis manifest (primary metric, tests, bootstrap, effect sizes, Holm groups, decision labels, replication rule) |
| `analysis/r0_paired_tests_validation_selected.csv` | Paired-test table of the initial 10-seed grid, read by the replication check (Section 5.7 of the paper) |
| `figures/` | `build_r1_assets.py` (result tables and result figures) and `verify_r1_assets.py` (independent re-check), with the generated figure data, fit diagnostics, figure PDFs, PNG carriers and `PROVENANCE.json` in `out/`; `build_tierd_assets.py` and `verify_tierd_assets.py` (extension table) and `build_f1conv_assets.py` and `verify_f1conv_assets.py` (budget-arm block of the contrast table), with their text numbers and provenance in `out/` |
| `dann_benchmark/` | Benchmark code of the initial 10-seed grid, dataset-access notes and its archived accuracy and timing outputs; start with `dann_benchmark/README.md` and `dann_benchmark/DATASETS.md` |
| `stats_outputs/` | Paired comparisons of the initial grid reconstructed from its archived histories |
| `REPRODUCIBILITY.md` | Reproduction scope and the canonical result folders |
| `PUBLIC_FILE_PROVENANCE.csv` | Source path and SHA-256 of every file under `runs/`, `analysis/` and `figures/` before and after the privacy edits described below |
| `SHA256SUMS` | Checksums of every file in the repository (`sha256sum -c SHA256SUMS` from the repository root) |

Each run folder contains:

- `RUN_MANIFEST.json`: command, environment, seeds, resource limits and the planned units of the run;
- `configs/`: the unit plan and `IDENTITY.json`, which binds the plan, the code fingerprint and the data hashes;
- `src/`: the exact code snapshot that the run executed, with SHA-256 hashes in `src/CODE_MANIFEST.json`;
- `processed_outputs/`: the aggregated outputs used by the paper;
- `verification/` (where present): the independent recomputation and the cross-checks;
- `raw_outputs/`: a public archive of the per-unit outputs (`*__public_outputs.tar.gz`), a member list with
  checksums (`*__public_outputs_MEMBERS.csv`) and the delivery verification record of the original archive.

The code snapshots differ only by additions. The timing and spatial-head CNN runs add `r1/f10_timing.py`,
`r1/f10_macs.py` and `r1/check_f10.py`, and extend `r1/r1_models.py` and `r1/r1_plan.py`. All other files are
byte-identical to the snapshot of the families run. The extension and the budget arm ran a third snapshot, which adds
`r1/aggregate_tierd.py` and `r1/check_tierd.py` and extends `r1/r1_data.py`, `r1/r1_models.py` and `r1/r1_plan.py` for the
Transformer front end and CIFAR-100. The budget arm executed it unchanged; its `src/` also holds `r1/aggregate_f1conv.py`,
the analysis fixed before its results, which `src/ANALYSIS_ADDITIONS.json` lists and `src/CODE_MANIFEST.json` does not.
The randomness arm ran a fourth snapshot, which changes only `r1/r1_runner.py` (two optional unit fields, `split_seed` and
`order_seed`, both defaulting to the data seed, so every earlier plan runs as before) and `r1/r1_plan.py` (the plan of the
arm); its `src/` also holds `r1/aggregate_f4split.py`, the analysis fixed before its results, listed in `src/ANALYSIS_ADDITIONS.json`.

The Python package `r1` inside each code snapshot, the file names that carry `r1`, and the `r0_` prefix of the
initial-grid table are the names of the code that ran and of the initial analysis; they are kept unchanged so that the
recorded code hashes stay verifiable.

## Public archives

The public archives keep, with their original member structure, `IDENTITY.json`, `configs/`, the worker environment
and terminal-status records, every unit's `result.json`, and every per-epoch `history.csv` (accuracy runs) or
`progress.json` (timing runs). Heartbeat streams, process logs, launch markers, duplicate `attempt.json` copies and
stderr logs are left out. Every `history.csv` is byte-identical to the delivered member. The member lists give the
SHA-256 of each public member and of the delivered member.

## Privacy edits

Text files that named local or remote paths, the physical name of the experiment machine, the internal run
identifiers or the software that launched the runs were edited with these placeholders, and nothing else was changed:

| Placeholder or edit | Replaces |
|---|---|
| `<remote-home>` | the home directory of the experiment account |
| `<project-root>` | the local project folder |
| `<font-dir>` | the local font folder |
| `<experiment-host>` | the physical host name of the experiment machine |
| descriptive run identifiers (the folder names under `runs/`) | the internal run identifiers, in file and folder names, archive member paths and file contents |
| empty `tool` and `model` fields | the name of the software that launched the runs and wrote the records |

The edited files are marked `True` in the `sanitized` column of `PUBLIC_FILE_PROVENANCE.csv` and `yes` in the
member lists. The public copies of the scripts in `figures/` also change these lines:

- the run folders are read from `runs/` instead of the internal experiment folder;
- the archive of the families run points to `*__public_outputs.tar.gz`;
- the fonts are read from `$R1_FONT_DIR` (default `fonts/`) as `IBMPlexSans-SemiBold.ttf`, `Barlow-Regular.ttf` and `Inter-Regular.otf`;
- the outputs, including the PNG figure carriers, are written to `figures/out/`.

## Reproducing the analysis

Training ran with Python 3.12.3, PyTorch 2.13.0 and torchvision 0.28.0 (CUDA 13.0 builds) under Ubuntu 24.04.
The analysis is pinned to Python 3.12.12, NumPy 2.3.5 and SciPy 1.18.0, and the figures used pandas 2.3.3 and
Matplotlib 3.11.1; the PNG carriers are rendered from the figure PDFs with `pdftoppm` (poppler).

Re-aggregate the families run from its public archive. Run from the repository root:

```bash
mkdir -p work
tar -xzf runs/2026-10-03_families_f1_f9/raw_outputs/2026-10-03_families_f1_f9__public_outputs.tar.gz -C work
python runs/2026-10-03_families_f1_f9/src/r1/aggregate_r1.py \
  --plan runs/2026-10-03_families_f1_f9/configs/plan_frozen_all.json \
  --units-root work/2026-10-03_families_f1_f9 \
  --manifest analysis/R1_ANALYSIS_MANIFEST.json \
  --identity-file runs/2026-10-03_families_f1_f9/configs/IDENTITY.json \
  --r0-paired analysis/r0_paired_tests_validation_selected.csv \
  --out work/processed_outputs
```

Every output equals the file in `processed_outputs/` byte for byte, except `ANALYSIS_RUN.json`, which records the
absolute input paths of the run.

Rebuild the result tables and figures, then re-check them independently. The three fonts are released under the SIL
Open Font License (IBM Plex Sans SemiBold for the emphasised cell values, Inter Regular for the axis titles, tick
labels and colour bars, Barlow Regular for the other text):

```bash
R1_FONT_DIR=/path/to/fonts python figures/build_r1_assets.py
python figures/verify_r1_assets.py
```

The build writes the LaTeX row bodies of the result tables, the figure data, the two result figures as PDF and their
PNG carriers into `figures/out/`. The files in `figures/out/` were built from this repository; the tables, the figure
data and the PNG carriers are byte-identical to the files used in the manuscript, the figure PDFs differ only in their
creation timestamp, and the verification re-read 607 values with no failure.

### Extension and budget arm

The processed outputs of the extension and of the budget arm were produced by `src/r1/aggregate_tierd.py` and
`src/r1/aggregate_f1conv.py` from the delivered archives, which both scripts check against their transfer receipts; the
delivered archives and the receipts are not part of this repository. The table rows are rebuilt from the processed
outputs and re-checked against the per-unit histories of the public archives. Run from the repository root:

```bash
mkdir -p work
tar -xzf runs/2026-10-04_extension_d1_d2/raw_outputs/2026-10-04_extension_d1_d2__public_outputs.tar.gz -C work
tar -xzf runs/2026-10-05_budget_arm_a18/raw_outputs/2026-10-05_budget_arm_a18__public_outputs.tar.gz -C work
python figures/build_tierd_assets.py
python figures/verify_tierd_assets.py check work/2026-10-04_extension_d1_d2
python figures/build_f1conv_assets.py
python figures/verify_f1conv_assets.py check work/2026-10-05_budget_arm_a18
```

The build scripts need pandas (2.3.3 was used). The rebuilt rows and text numbers are byte-identical to the files used
in the manuscript, and the two verifications re-read 604 and 422 values with no failure. The `verification/` folder of
each run holds the independent recomputation of every label from the per-unit histories.

The processed outputs of the randomness arm were produced by `src/r1/aggregate_f4split.py` (Python standard library only)
from the extracted delivered archive. Its reproduction gate compares the histories of the eight repeated runs and of the
two value-0 runs with the delivered SHA-256 of the F4 histories, which the member list of the families archive also
records; all ten are equal. The arm is descriptive (no test and no label), and `verification/independent_recompute.json`
holds its independent recomputation (0 disagreements).

Re-running a training plan needs the datasets described in `dann_benchmark/DATASETS.md`. Every `result.json`
records SHA-256 hashes of the training and test tensors (`data_hashes`), so a re-run can be checked against the same
inputs. The command of each run is in its `RUN_MANIFEST.json`.

## Initial grid

`dann_benchmark/` holds the benchmark code of the initial 10-seed grid (DANN-LRF, DANN-RANDOM, DANN-GRF, Naive-Branch,
MLP-Param and VANN-Same on FashionMNIST, KMNIST and CIFAR-10, full and reduced data, with separate CPU timing runs) and
its archived outputs under `dann_benchmark/runs/`; `stats_outputs/` holds the paired comparisons reconstructed from those
histories. The replication check of the study compares the families run with the paired-test table of that grid in
`analysis/r0_paired_tests_validation_selected.csv`, which differs from `stats_outputs/paired_tests_validation_selected.csv`
in its six reduced-data rows; the check reads the table in `analysis/`.

Smoke test of the initial-grid code:

```text
cd dann_benchmark
python benchmark.py --dataset fashionmnist --epochs 3 --seeds 0 --models dann_lrf naive_branch mlp_param vann_same
```

If the local cache is missing, the benchmark code downloads the required dataset files into `dann_benchmark/data/`.

## What is intentionally excluded

- Manuscript source files and submission materials
- Internal planning or archive materials
- Local dataset cache files under `dann_benchmark/data/`
- The delivered (pre-publication) archives and transfer receipts of the runs; the public archives and member lists
  carry the checksums of every kept member

## Start here

1. `README.md` (this file) and `REPRODUCIBILITY.md`
2. `runs/2026-10-03_families_f1_f9/RUN_MANIFEST.json`
3. `figures/build_r1_assets.py`
4. `dann_benchmark/README.md` and `dann_benchmark/DATASETS.md`
