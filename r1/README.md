# Revision (R1) materials

This folder holds the code, run manifests, configurations, per-run training histories, processed outputs,
verification records and result-table and result-figure scripts of the revised study *Branching-Controlled
Disentanglement of Dendritic Nonlinearity from Routing and Capacity in Parameter-Matched Image Classification*.
The package of the submitted version at the repository root is unchanged. The replication check of the
revision compares the new runs with the paired-test table of the submitted analysis, which is included as
`analysis/r0_paired_tests_validation_selected.csv`. That table differs from
`stats_outputs/paired_tests_validation_selected.csv` in its six reduced-data rows, and the replication check reads
the table in `analysis/`.

## Contents

| Path | Content |
|---|---|
| `runs/2026-10-03_claude_mta_cuda_r1_frozen/` | Families F1 to F9 (main grid, reference priors, sensitivity, randomness sources, shuffled pixels, channel-aware routing, dendrite slope, optimization fairness, convolutional stem): 1,615 training runs |
| `runs/2026-10-04_claude_mta_cuda_r1_cnnflat/` | Exploratory spatial-head CNN arm: 60 runs plus 6 repeats of frozen runs |
| `runs/2026-10-04_claude_mta_cuda_r1_f10_gpu/` | F10 timing on the GPU (51 measurement units); its `processed_outputs/` hold the combined CPU and GPU timing analysis |
| `runs/2026-10-04_claude_mta_cuda_r1_f10_cpu/` | F10 timing on the CPU (51 measurement units) |
| `analysis/R1_ANALYSIS_MANIFEST.json` | Statistical analysis manifest (primary metric, tests, bootstrap, effect sizes, Holm groups, decision labels, replication rule) |
| `analysis/r0_paired_tests_validation_selected.csv` | Paired-test table of the submitted analysis, read by the replication check |
| `figures/r1/` | `build_r1_assets.py` (result tables and result figures) and `verify_r1_assets.py` (independent re-check), with the generated figure data, fit diagnostics, figure PDFs and `PROVENANCE.json` in `out/` |
| `PUBLIC_FILE_PROVENANCE.csv` | Source path and SHA-256 of every file before and after the privacy edits described below |
| `SHA256SUMS` | Checksums of every file in this folder (`sha256sum -c SHA256SUMS` from inside `r1/`) |

Each run folder contains:

- `RUN_MANIFEST.json`: command, environment, seeds, resource limits and the planned units of the run;
- `configs/`: the unit plan and `IDENTITY.json`, which binds the plan, the code fingerprint and the data hashes;
- `src/`: the exact code snapshot that the run executed, with SHA-256 hashes in `src/CODE_MANIFEST.json`;
- `processed_outputs/`: the aggregated outputs used by the paper;
- `verification/` (where present): the independent recomputation and the cross-checks;
- `raw_outputs/`: a public archive of the per-unit outputs (`*__public_outputs.tar.gz`), a member list with
  checksums (`*__public_outputs_MEMBERS.csv`) and the delivery verification record of the original archive.

The two code snapshots are the same apart from additions. The F10 and spatial-head CNN runs add `r1/f10_timing.py`,
`r1/f10_macs.py` and `r1/check_f10.py`, and extend `r1/r1_models.py` and `r1/r1_plan.py`. All other files are
byte-identical to the snapshot of the frozen run.

## Public archives

The public archives keep, with their original member paths, `IDENTITY.json`, `configs/`, the worker environment
and terminal-status records, every unit's `result.json`, and every per-epoch `history.csv` (accuracy runs) or
`progress.json` (timing runs). Heartbeat streams, process logs, launch markers, duplicate `attempt.json` copies and
stderr logs are left out. Every `history.csv` is byte-identical to the delivered member. The member lists give the
SHA-256 of each public member and of the delivered member.

## Privacy edits

Text files that named local or remote paths or the physical name of the experiment machine were edited with these
placeholders, and nothing else was changed:

| Placeholder | Replaces |
|---|---|
| `<remote-home>` | the home directory of the experiment account |
| `<project-root>` | the local project folder |
| `<font-dir>` | the local font folder |
| `<experiment-host>` | the physical host name of the experiment machine |

The edited files are marked `True` in the `sanitized` column of `PUBLIC_FILE_PROVENANCE.csv` and `yes` in the
member lists. The public copies of `build_r1_assets.py` and `verify_r1_assets.py` also change these lines:

- the run folders are read from `r1/runs/` instead of `experiments/`;
- the frozen archive name points to `*__public_outputs.tar.gz`;
- the fonts are read from `$R1_FONT_DIR` (default `r1/fonts/`) as `IBMPlexSans-SemiBold.ttf` and `Barlow-Regular.ttf`;
- the PNG figure carriers are written to `figures/r1/out/` next to the PDFs.

## Reproducing the analysis

Training ran with Python 3.12.3, PyTorch 2.13.0 and torchvision 0.28.0 (CUDA 13.0 builds) under Ubuntu 24.04.
The analysis is pinned to Python 3.12.12, NumPy 2.3.5 and SciPy 1.18.0, and the figures used pandas 2.3.3 and
Matplotlib 3.11.1.

Re-aggregate the frozen run from its public archive. Run from `r1/`:

```bash
mkdir -p work
tar -xzf runs/2026-10-03_claude_mta_cuda_r1_frozen/raw_outputs/2026-10-03_claude_mta_cuda_r1_frozen__public_outputs.tar.gz -C work
python runs/2026-10-03_claude_mta_cuda_r1_frozen/src/r1/aggregate_r1.py \
  --plan runs/2026-10-03_claude_mta_cuda_r1_frozen/configs/plan_frozen_all.json \
  --units-root work/2026-10-03_claude_mta_cuda_r1_frozen \
  --manifest analysis/R1_ANALYSIS_MANIFEST.json \
  --identity-file runs/2026-10-03_claude_mta_cuda_r1_frozen/configs/IDENTITY.json \
  --r0-paired analysis/r0_paired_tests_validation_selected.csv \
  --out work/processed_outputs
```

Every output equals the file in `processed_outputs/` byte for byte, except `ANALYSIS_RUN.json`, which records the
absolute input paths of the run.

Rebuild the result tables and figures, then re-check them independently. Both fonts are released under the SIL Open
Font License (IBM Plex Sans SemiBold and Barlow Regular):

```bash
R1_FONT_DIR=/path/to/fonts python figures/r1/build_r1_assets.py
python figures/r1/verify_r1_assets.py
```

The build writes the LaTeX row bodies of the result tables, the figure data and the two result figures into
`figures/r1/out/`. On the author's machine these outputs, built from this folder, were byte-identical to the
files used in the revised manuscript, and the verification re-read 607 values with no failure.

Re-running a training plan needs the datasets described in `dann_benchmark/DATASETS.md`. Every `result.json`
records SHA-256 hashes of the training and test tensors (`data_hashes`), so a re-run can be checked against the same
inputs. The command of each run is in its `RUN_MANIFEST.json`.
