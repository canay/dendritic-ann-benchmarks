"""Build the F1 convergence-arm (MC-NEURO-R1-004, A18-A19) rows and prose numbers of NEURO/manuscript-r1 from the verified
outputs of run 2026-10-05_claude_mta_cuda_r1_f1conv.

Date/time: 2026-10-05 21:22 +03:00
Tool: Cowork-Claude
Model, if known: claude-opus-5-5 (max)
Operation ID: neucom-r1-f1conv-delivery-analysis-20261005

Inputs (read-only): experiments/2026-10-05_claude_mta_cuda_r1_f1conv/processed_outputs/ (summary_by_cell.csv,
decision_inputs.csv, contrasts.csv, budget_agreement.csv, budget_change.csv, composite_decisions.csv, unit_table.csv,
prefix_gate.csv, ANALYSIS_RUN.json) and the run's verification/independent_recompute.json, which must carry the verdict PASS
with 9 contrasts, 12 cells, 240 byte-equal prefixes and no disagreement (fail-closed otherwise: exit 2, nothing written).
Nothing is trained or re-estimated; every printed value is a field of the analysis outputs, apart from the descriptive
training-accuracy gaps between DANN-LRF and Naive-Branch, which are differences of two cell means.

Outputs (figures/r1/out/):
  table5_f1conv_block.tex    the 100-epoch block of Table 5 (the 8 columns of Table 5; block header, then DANN-LRF minus
                             Naive-Branch, minus MLP-Param and minus DANN-RANDOM, each on FashionMNIST, KMNIST and CIFAR-10)
  table3_f1conv_row.tex      the Table 3 row of the arm (A18), run count from the unit table
  f1conv_text_numbers.json   every value the prose may cite, keyed by cell and contrast id, with the 30-epoch selection of
                             every unit (frozen_best_val_epoch) beside the 100-epoch one
  PROVENANCE_f1conv.json     input and output hashes
The independent check is figures/r1/verify_f1conv_assets.py (raw histories plus the run's independent recomputation).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[2]
RUN = P / "runs" / "2026-10-05_claude_mta_cuda_r1_f1conv"  # public copy
PO = RUN / "processed_outputs"
RECOMPUTE = RUN / "verification" / "independent_recompute.json"
OUT = P / "figures" / "r1" / "out"
INPUTS: list[Path] = []
OUTPUTS: list[Path] = []

DATASETS = ("fashion", "kmnist", "cifar")
COND_LABEL = {"fashion": "FashionMNIST full", "kmnist": "KMNIST full", "cifar": "CIFAR-10 full"}
MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random")
CONTRAST_ORDER = ("naive_branch", "mlp_param", "dann_random")      # Table 5 order: RQ1 rows, then the RQ2 rows
CONTRAST_LABEL = {"naive_branch": "\\DANNLRF{} $-$ \\NAIVEBRANCH{}", "mlp_param": "\\DANNLRF{} $-$ \\MLPPARAM{}",
                  "dann_random": "\\DANNLRF{} $-$ \\DANNRANDOM{}"}
BLOCK_HEADER = "Budget arm A18: full-data contrasts at 100 epochs"
LABELS = {"supported": "supported", "not_supported": "not supported", "reverse": "reverse"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> pd.DataFrame:
    INPUTS.append(path)
    return pd.read_csv(path)


def fmt(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0.0:
            return s[1:]
        return "$-$" + s[1:]
    return s


def ci(lo: float, hi: float) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    OUTPUTS.append(path)


def main() -> int:
    INPUTS.append(RECOMPUTE)
    rec = json.loads(RECOMPUTE.read_text(encoding="utf-8"))
    if (rec.get("verdict") != "PASS" or rec.get("contrasts") != 9 or rec.get("cells") != 12
            or rec.get("prefix_byte_equal") != 240 or rec.get("disagreements")):
        print("FAIL: the independent recomputation is not a 9-contrast, 12-cell, 240-prefix PASS; nothing written")
        return 2
    INPUTS.append(PO / "ANALYSIS_RUN.json")
    run = json.loads((PO / "ANALYSIS_RUN.json").read_text(encoding="utf-8"))
    summary = read_csv(PO / "summary_by_cell.csv")
    decision = read_csv(PO / "decision_inputs.csv")
    contrasts = read_csv(PO / "contrasts.csv")
    agreement = read_csv(PO / "budget_agreement.csv")
    change = read_csv(PO / "budget_change.csv")
    composite = read_csv(PO / "composite_decisions.csv")
    units = read_csv(PO / "unit_table.csv")
    prefix = read_csv(PO / "prefix_gate.csv")
    assert len(decision) == 9 and len(contrasts) == 9 and len(agreement) == 9 and len(change) == 9
    assert len(summary) == 12 and len(units) == 240 and len(prefix) == 240
    assert run.get("versions_match_pinned") is True and run.get("prefix_gate_pass") is True
    assert bool(prefix.prefix_rows_equal.all()) and bool(prefix.prefix_bytes_equal.all())
    assert set(units.family) == {"MC004_F1_e100"} and set(units.epochs) == {100}

    def cell(ds: str, model: str) -> pd.Series:
        rows = summary[(summary.condition == f"{ds}_full_e100") & (summary.model == model)]
        assert len(rows) == 1, (ds, model, len(rows))
        r = rows.iloc[0]
        assert int(r.n_seeds) == 20, (ds, model, r.n_seeds)
        return r

    numbers: dict[str, object] = {"cells": {}, "contrasts": {}, "train_gap_dann_minus_naive_pct": {},
                                  "selected_last_epoch_ranges": {}}
    for ds in DATASETS:
        for model in MODELS:
            r = cell(ds, model)
            u = units[(units.condition == f"{ds}_full_e100") & (units.model == model)]
            assert len(u) == 20 and sorted(u.seed) == list(range(20)), (ds, model)
            numbers["cells"][f"{ds}|{model}"] = {
                "acc_mean_pp": float(r.acc_mean_pp), "acc_sd_pp": float(r.acc_sd_pp), "acc_min_pp": float(r.acc_min_pp),
                "acc_max_pp": float(r.acc_max_pp), "acc_ci95_pp": [float(r.acc_ci95_low_pp), float(r.acc_ci95_high_pp)],
                "frozen30_acc_mean_pp": float(r.frozen30_acc_mean_pp),
                "acc_gain_over_frozen30_pp": float(r.acc_gain_over_frozen30_pp),
                "train_acc_last_epoch_mean_pct": 100.0 * float(r.train_acc_last_epoch_mean),
                "selected_epoch_mean": float(r.selected_epoch_mean),
                "selected_last_epoch_count": int(r.selected_last_epoch_count),
                "selected_within_first30_count": int(r.selected_within_first30_count),
                "frozen30_selected_last_epoch_count": int((u.frozen_best_val_epoch == 30).sum()),
                "frozen30_selected_epoch_mean": float(u.frozen_best_val_epoch.mean()),
                "effective_params": int(r.effective_params)}
            assert int((u.best_val_epoch == 100).sum()) == int(r.selected_last_epoch_count), (ds, model)
        numbers["train_gap_dann_minus_naive_pct"][ds] = (numbers["cells"][f"{ds}|dann_lrf"]["train_acc_last_epoch_mean_pct"]
                                                         - numbers["cells"][f"{ds}|naive_branch"]["train_acc_last_epoch_mean_pct"])
        last = [numbers["cells"][f"{ds}|{m}"]["selected_last_epoch_count"] for m in MODELS]
        last30 = [numbers["cells"][f"{ds}|{m}"]["frozen30_selected_last_epoch_count"] for m in MODELS]
        numbers["selected_last_epoch_ranges"][ds] = {"e100": [min(last), max(last)], "e30": [min(last30), max(last30)]}
    numbers["selected_last_epoch_totals"] = {
        "e100": int((units.best_val_epoch == 100).sum()), "e30": int((units.frozen_best_val_epoch == 30).sum()),
        "units": int(len(units))}

    dec = decision.set_index("criterion_id")
    con = contrasts.set_index("id")
    agr = agreement.set_index("id")
    chg = change.set_index("id")
    rows: list[str] = ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{BLOCK_HEADER}}}}} \\\\"]
    used: list[str] = []
    for b in CONTRAST_ORDER:
        for ds in DATASETS:
            cid = f"F1e100:{ds}_full_e100:dann_lrf-{b}"
            r = dec.loc[cid]
            assert int(r.budget_epochs) == 100 and int(r.n) == 20, cid
            used.append(cid)
            rows.append(f"{CONTRAST_LABEL[b]} & {COND_LABEL[ds]} & {fmt(float(r.mean_diff_pp))} & "
                        f"{ci(float(r.ci_low_pp), float(r.ci_high_pp))} & {pval(float(r.holm_p))} & {fmt(float(r.d_z))} & "
                        f"{fmt(float(r.rank_biserial))} & {LABELS[r.outcome]} \\\\")
            diffs = [int(x) for x in str(con.loc[cid].differences_count).split(";")]
            assert len(diffs) == 20, cid
            numbers["contrasts"][cid] = {
                "budget_epochs": 100, "mean_diff_pp": float(r.mean_diff_pp),
                "ci95_pp": [float(r.ci_low_pp), float(r.ci_high_pp)], "holm_p": float(r.holm_p),
                "p_two_sided": float(con.loc[cid].p_two_sided), "d_z": float(r.d_z), "rank_biserial": float(r.rank_biserial),
                "wilcoxon_method": str(r.wilcoxon_method), "outcome": str(r.outcome),
                "positive_seeds": sum(1 for x in diffs if x > 0), "negative_seeds": sum(1 for x in diffs if x < 0),
                "frozen_id": str(agr.loc[cid].frozen_id), "frozen30_outcome": str(agr.loc[cid].frozen30_outcome),
                "frozen30_mean_diff_pp": float(agr.loc[cid].frozen30_mean_diff_pp), "agreement": str(agr.loc[cid].agreement),
                "budget_change_mean_pp": float(chg.loc[cid].budget_change_mean_pp),
                "budget_change_ci95_pp": [float(chg.loc[cid].budget_change_ci95_low_pp),
                                          float(chg.loc[cid].budget_change_ci95_high_pp)],
                "seeds_change_positive": int(chg.loc[cid].seeds_change_positive),
                "seeds_change_negative": int(chg.loc[cid].seeds_change_negative)}
    assert sorted(used) == sorted(decision.criterion_id), "the block must carry every contrast of the arm once"
    write(OUT / "table5_f1conv_block.tex", "\n".join(rows) + "\n")

    fam = units.groupby("family").size().to_dict()
    n_runs = int(fam["MC004_F1_e100"])
    assert n_runs == 240, fam
    t3 = (f"A18 & Budget arm of F1 & RQ1, RQ2 & full-data F1 contrasts at a 100-epoch budget & \\DANNLRF{{}}, "
          f"\\DANNRANDOM{{}}, \\NAIVEBRANCH{{}}, \\MLPPARAM{{}} & 3 full & 0--19 & {n_runs} \\\\")
    write(OUT / "table3_f1conv_row.tex", t3 + "\n")

    numbers["composite"] = {r.rule_id: str(r.outcome) for r in composite.itertuples()}
    numbers["budget_differing"] = [cid for cid in used if numbers["contrasts"][cid]["agreement"] != "consistent"]
    numbers["prefix_gate"] = {"pass": bool(run["prefix_gate_pass"]), "rows_equal": int(prefix.prefix_rows_equal.sum()),
                              "bytes_equal": int(prefix.prefix_bytes_equal.sum()), "units": int(len(prefix))}
    numbers["units_per_family"] = {k: int(v) for k, v in fam.items()}
    numbers["recompute"] = {"verdict": rec["verdict"], "label_tally": rec["label_tally"], "script_sha256": rec["script_sha256"],
                            "differing_between_budgets": rec["differing_between_budgets"]}
    numbers["analysis_versions"] = run["versions"]
    write(OUT / "f1conv_text_numbers.json", json.dumps(numbers, indent=1) + "\n")

    prov = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": {"path": str(Path(__file__).relative_to(P)).replace("\\", "/"), "sha256": sha(Path(__file__))},
            "python": platform.python_version(), "libraries": {"pandas": pd.__version__},
            "inputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in INPUTS],
            "outputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in OUTPUTS]}
    (OUT / "PROVENANCE_f1conv.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"outputs": len(OUTPUTS), "inputs": len(INPUTS), "composite": numbers["composite"],
                      "budget_differing": numbers["budget_differing"],
                      "selected_last_epoch_totals": numbers["selected_last_epoch_totals"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
