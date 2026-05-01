# Statistical Outputs

This folder contains supporting statistical comparison outputs derived from the archived benchmark runs.

## Files

- `paired_tests.csv`
  Pairwise statistical comparisons based on the historical summary exports.
- `paired_tests_validation_selected.csv`
  Pairwise statistical comparisons based on validation-selected accuracy reconstructed from the archived per-epoch histories.

## Interpretation

- The validation-selected file is the more conservative summary because it avoids relying on the historical per-run `best_test_acc` export.
- These files are provided as supporting result summaries alongside the raw archived benchmark outputs under `dann_benchmark/runs/`.
