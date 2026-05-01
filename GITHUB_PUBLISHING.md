# GitHub Publishing Steps

Target repository:

- `https://github.com/canay/dendritic-ann-benchmarks`

## What is already prepared locally

- Canonical repository structure
- Root `README.md`
- `LICENSE`
- `CITATION.cff`
- Dataset access note in `dann_benchmark/DATASETS.md`
- Reproducibility guide in `REPRODUCIBILITY.md`
- Manuscript PDF and sources
- Submission support files

## Create the GitHub repository

Create a new empty repository on GitHub with:

- Owner: `canay`
- Name: `dendritic-ann-benchmarks`
- Visibility: your choice
- Do **not** initialize with a README, `.gitignore`, or license, because those are already present locally

## Publish commands

From the local repository root:

```text
git push -u origin main
```

If the remote was not added for any reason:

```text
git remote add origin https://github.com/canay/dendritic-ann-benchmarks.git
git push -u origin main
```

## After the first push

- Confirm that `dann_benchmark/data/` is absent from GitHub
- Confirm that `dann_benchmark/runs/` is present
- Confirm that `paper_package/manuscript/dann_manuscript.pdf` opens correctly
- Copy the live repository URL into any final submission portal fields if needed

## Suggested first release tag

`v1.0.0-manuscript-submission`
