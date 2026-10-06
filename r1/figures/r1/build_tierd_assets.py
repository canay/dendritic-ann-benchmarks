"""Build the tier D (MC-NEURO-R1-003) table rows and prose numbers of NEURO/manuscript-r1 from the verified outputs of run
2026-10-04_claude_mta_cuda_r1_tierd.

Date/time: 2026-10-05 02:00 +03:00 (measured write time; a first stamp typed ahead of the clock was corrected at 02:05;
written before any result of the run was opened)
Tool: Cowork-Claude
Model, if known: claude-opus-5-5 (max)
Operation ID: neucom-r1-tierd-delivery-analysis-20261005

Inputs (read-only): experiments/2026-10-04_claude_mta_cuda_r1_tierd/processed_outputs/ (summary_by_cell.csv,
decision_inputs.csv, contrasts.csv, composite_decisions.csv, anchors.csv, unit_table.csv, ANALYSIS_RUN.json) and the run's
verification/independent_recompute.json, which must carry the verdict PASS (fail-closed otherwise: exit 2, nothing written).
Nothing is trained or re-estimated; every printed value is a field of the analysis outputs, apart from the descriptive
budget differences of the cells, which are differences of two printed cell means.

Outputs (figures/r1/out/):
  table8_tierd_accuracy.tex   row bodies of the accuracy panel (10 columns: model, parameters, and for each budget the
                              accuracy as mean +- SD, the mean training accuracy at the last epoch, the mean selected epoch
                              and the number of seeds selected at the last epoch)
  table8_tierd_contrasts.tex  row bodies of the contrast panel (the 8 columns of Table 5), 30-epoch block then 100-epoch block
  table3_tierd_rows.tex       the two Table 3 rows of the extension, run counts from the unit table
  tierd_text_numbers.json     every value the prose may cite, keyed by cell and contrast id
  PROVENANCE_tierd.json       input and output hashes
The independent check is figures/r1/verify_tierd_assets.py (raw histories plus the independent recomputation).
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
RUN = P / "runs" / "2026-10-04_claude_mta_cuda_r1_tierd"  # public copy
PO = RUN / "processed_outputs"
RECOMPUTE = RUN / "verification" / "independent_recompute.json"
OUT = P / "figures" / "r1" / "out"
INPUTS: list[Path] = []
OUTPUTS: list[Path] = []

E100 = "_e100"
D1_CONDS = ("fashion_full", "cifar_full")
D2_COND = "cifar100_full"
HEADS = ("tf_stem_dann_lrf", "tf_stem_naive_branch", "tf_stem_mlp")
D2_MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random", "vann_same")
COND_LABEL = {"fashion_full": "FashionMNIST full", "cifar_full": "CIFAR-10 full", "cifar100_full": "CIFAR-100 full"}
MODEL_LABEL = {"tf_stem_dann_lrf": "Transformer + \\DANNLRF{} head", "tf_stem_naive_branch": "Transformer + \\NAIVEBRANCH{} head",
               "tf_stem_mlp": "Transformer + MLP head", "dann_lrf": "\\DANNLRF{}", "naive_branch": "\\NAIVEBRANCH{}",
               "mlp_param": "\\MLPPARAM{}", "dann_random": "\\DANNRANDOM{}", "vann_same": "\\VANNSAME{}"}
GROUPS = [("Transformer front end, FashionMNIST", "fashion_full", HEADS),
          ("Transformer front end, CIFAR-10", "cifar_full", HEADS),
          ("Flattened CIFAR-100", D2_COND, D2_MODELS)]
CONTRAST_LABEL = {"tf_stem_naive_branch": "Transformer: \\DANNLRF{} head $-$ \\NAIVEBRANCH{} head",
                  "tf_stem_mlp": "Transformer: \\DANNLRF{} head $-$ MLP head",
                  "naive_branch": "\\DANNLRF{} $-$ \\NAIVEBRANCH{}", "mlp_param": "\\DANNLRF{} $-$ \\MLPPARAM{}",
                  "dann_random": "\\DANNLRF{} $-$ \\DANNRANDOM{}"}
LABELS = {"supported": "supported", "not_supported": "not supported", "reverse": "reverse", "no_difference": "no difference"}


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


def pm(mean: float, sd: float) -> str:
    return f"{fmt(mean)} $\\pm$ {fmt(sd)}"


def ci(lo: float, hi: float) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    OUTPUTS.append(path)


def contrast_ids() -> list[tuple[str, str, str, str]]:
    """(id, budget block, base condition, B) in the protocol order k = 0-13 (A16 e, A17)."""
    out = []
    for suffix, g1, g2 in (("", "D1", "D2"), (E100, "D1_e100", "D2_e100")):
        for cond in D1_CONDS:
            for b in ("tf_stem_naive_branch", "tf_stem_mlp"):
                out.append((f"{g1}:{cond}{suffix}:tf_stem_dann_lrf-{b}", suffix, cond, b))
        for b in ("naive_branch", "mlp_param", "dann_random"):
            out.append((f"{g2}:{D2_COND}{suffix}:dann_lrf-{b}", suffix, D2_COND, b))
    return out


def main() -> int:
    INPUTS.append(RECOMPUTE)
    rec = json.loads(RECOMPUTE.read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("contrasts") != 14 or rec.get("cells") != 22:
        print("FAIL: the independent recomputation is not a 14-contrast, 22-cell PASS; nothing written")
        return 2
    INPUTS.append(PO / "ANALYSIS_RUN.json")
    run = json.loads((PO / "ANALYSIS_RUN.json").read_text(encoding="utf-8"))
    summary = read_csv(PO / "summary_by_cell.csv")
    decision = read_csv(PO / "decision_inputs.csv")
    contrasts = read_csv(PO / "contrasts.csv")
    composite = read_csv(PO / "composite_decisions.csv")
    anchors = read_csv(PO / "anchors.csv")
    units = read_csv(PO / "unit_table.csv")
    assert len(decision) == 14 and len(contrasts) == 14, (len(decision), len(contrasts))
    assert len(summary) == 22, len(summary)
    assert len(units) == 449, len(units)
    assert len(anchors) == 9, len(anchors)
    assert run.get("versions_match_pinned") is True, run.get("versions")

    def cell(cond: str, model: str) -> pd.Series:
        rows = summary[(summary.condition == cond) & (summary.model == model)]
        assert len(rows) == 1, (cond, model, len(rows))
        r = rows.iloc[0]
        assert int(r.n_seeds) == 20, (cond, model, r.n_seeds)
        return r

    numbers: dict[str, object] = {"cells": {}, "contrasts": {}, "budget_change": {}}
    t8a: list[str] = []
    for gi, (group, cond, models) in enumerate(GROUPS):
        if gi:
            t8a.append("\\midrule")
        t8a.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
        for model in models:
            r30, r100 = cell(cond, model), cell(cond + E100, model)
            p30, p100 = int(r30.effective_params), int(r100.effective_params)
            assert p30 == p100, (cond, model, p30, p100)
            vals = []
            for r in (r30, r100):
                vals += [pm(float(r.acc_mean_pp), float(r.acc_sd_pp)), fmt(100.0 * float(r.train_acc_last_epoch_mean), 1),
                         fmt(float(r.selected_epoch_mean), 1), str(int(r.selected_last_epoch_count))]
            t8a.append(f"{MODEL_LABEL[model]} & {p30:,} & " + " & ".join(vals) + " \\\\")
            for c, r in ((cond, r30), (cond + E100, r100)):
                numbers["cells"][f"{c}|{model}"] = {
                    "acc_mean_pp": float(r.acc_mean_pp), "acc_sd_pp": float(r.acc_sd_pp), "acc_min_pp": float(r.acc_min_pp),
                    "acc_max_pp": float(r.acc_max_pp), "acc_ci95_pp": [float(r.acc_ci95_low_pp), float(r.acc_ci95_high_pp)],
                    "train_acc_last_epoch_mean_pct": 100.0 * float(r.train_acc_last_epoch_mean),
                    "selected_epoch_mean": float(r.selected_epoch_mean),
                    "selected_last_epoch_count": int(r.selected_last_epoch_count), "effective_params": int(r.effective_params),
                    "unit_seconds_mean": float(r.unit_seconds_mean)}
            numbers["budget_change"][f"{cond}|{model}"] = {
                "acc_mean_e100_minus_30_pp": float(r100.acc_mean_pp) - float(r30.acc_mean_pp),
                "train_acc_last_e100_minus_30_pct": 100.0 * (float(r100.train_acc_last_epoch_mean) - float(r30.train_acc_last_epoch_mean))}
    write(OUT / "table8_tierd_accuracy.tex", "\n".join(t8a) + "\n")

    dec = decision.set_index("criterion_id")
    con = contrasts.set_index("id")
    used: list[str] = []
    t8b: list[str] = []
    for block, header in (("", "30-epoch protocol (primary labels)"), (E100, "100-epoch budget arm")):
        if block:
            t8b.append("\\midrule")
        t8b.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{header}}}}} \\\\")
        for cid, suffix, cond, b in contrast_ids():
            if suffix != block:
                continue
            r = dec.loc[cid]
            used.append(cid)
            t8b.append(f"{CONTRAST_LABEL[b]} & {COND_LABEL[cond]} & {fmt(float(r.mean_diff_pp))} & "
                       f"{ci(float(r.ci_low_pp), float(r.ci_high_pp))} & {pval(float(r.holm_p))} & {fmt(float(r.d_z))} & "
                       f"{fmt(float(r.rank_biserial))} & {LABELS[r.outcome]} \\\\")
            numbers["contrasts"][cid] = {
                "budget_epochs": int(r.budget_epochs), "mean_diff_pp": float(r.mean_diff_pp),
                "ci95_pp": [float(r.ci_low_pp), float(r.ci_high_pp)], "holm_p": float(r.holm_p),
                "p_two_sided": float(con.loc[cid].p_two_sided), "d_z": float(r.d_z), "rank_biserial": float(r.rank_biserial),
                "wilcoxon_method": str(r.wilcoxon_method), "outcome": str(r.outcome),
                "positive_seeds": sum(1 for x in str(con.loc[cid].differences_count).split(";") if int(x) > 0),
                "negative_seeds": sum(1 for x in str(con.loc[cid].differences_count).split(";") if int(x) < 0)}
    write(OUT / "table8_tierd_contrasts.tex", "\n".join(t8b) + "\n")
    assert sorted(used) == sorted(decision.criterion_id), "the contrast panel must carry every tier D contrast once"

    # Table 3 rows: run counts per family from the unit table (the nine lineage anchors are named in the notes)
    fam = units.groupby("family").size().to_dict()
    n_d1 = int(fam["MC003_D1_tfstem"]) + int(fam["MC003_D1_tfstem_e100"])
    n_d2 = int(fam["MC003_D2_cifar100"]) + int(fam["MC003_D2_cifar100_e100"])
    assert (n_d1, n_d2, int(fam["MC003_anchor"])) == (240, 200, 9), (n_d1, n_d2, fam)
    t3 = [f"D1 & Transformer front end & RQ1, RQ2 & transfer of the attribution to a minimal Transformer front end & "
          f"Transformer stem with \\DANNLRF{{}}, \\NAIVEBRANCH{{}} and MLP heads & FashionMNIST, CIFAR-10 full & 0--19 & {n_d1} \\\\",
          f"D2 & CIFAR-100 & RQ1, RQ2 & nonlinearity, dense and routing contrasts with 100 classes & \\DANNLRF{{}}, \\NAIVEBRANCH{{}}, "
          f"\\MLPPARAM{{}}, \\DANNRANDOM{{}}, \\VANNSAME{{}} & CIFAR-100 full & 0--19 & {n_d2} \\\\"]
    write(OUT / "table3_tierd_rows.tex", "\n".join(t3) + "\n")

    numbers["composite"] = {r.rule_id: str(r.outcome) for r in composite.itertuples()}
    numbers["budget_differing"] = [x for x in str(composite.set_index("rule_id").loc["budget_agreement"].differing).split(";")
                                   if x and x != "nan"]
    numbers["anchors_pass"] = bool(run["anchors_pass"])
    numbers["anchors"] = [{"unit": r.anchor_unit, "bytes_equal": bool(r.bytes_equal), "params_equal": bool(r.params_equal),
                           "data_equal": bool(r.data_equal)} for r in anchors.itertuples()]
    numbers["units_per_family"] = {k: int(v) for k, v in fam.items()}
    numbers["recompute"] = {"verdict": rec["verdict"], "label_tally": rec["label_tally"], "script_sha256": rec["script_sha256"]}
    numbers["analysis_versions"] = run["versions"]
    write(OUT / "tierd_text_numbers.json", json.dumps(numbers, indent=1) + "\n")

    prov = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "script": {"path": str(Path(__file__).relative_to(P)).replace("\\", "/"), "sha256": sha(Path(__file__))},
            "python": platform.python_version(), "libraries": {"pandas": pd.__version__},
            "inputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in INPUTS],
            "outputs": [{"path": str(p.relative_to(P)).replace("\\", "/"), "sha256": sha(p)} for p in OUTPUTS]}
    (OUT / "PROVENANCE_tierd.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"outputs": len(OUTPUTS), "inputs": len(INPUTS), "composite": numbers["composite"],
                      "anchors_pass": numbers["anchors_pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
