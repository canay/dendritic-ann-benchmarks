"""Write the frozen R1 analysis manifest (protocol Section 7, amendments A5-A9; review PCR-001/003/005/006/008/010).

    python r1/r1_analysis_manifest.py --out <MD/09_audit_revision/R1_FREEZE/R1_ANALYSIS_MANIFEST.json>

Every inferential contrast, every descriptive summary, every composite decision rule, the replication check and the
A4 identity checks are listed here BEFORE any post-freeze result exists; `r1/aggregate_r1.py` computes exactly this
list and nothing else. Deterministic output (no clock, no host data) so the file hash is reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FULL = ("fashion_full", "kmnist_full", "cifar_full")
LOW = ("fashion_low02", "fashion_low01", "cifar_low02")
S20, S10, S5 = list(range(20)), list(range(10)), list(range(5))
BOOT_BASE = 20261003000


def cell(family: str, condition: str, model: str) -> dict:
    return {"family": family, "condition": condition, "model": model}


def build() -> dict:
    contrasts = []

    def add(cid, group, status, a, b, seeds, decision=None, kind="paired", note=""):
        contrasts.append({"id": cid, "holm_group": group, "status": status, "kind": kind, "A": a, "B": b,
                          "seeds": seeds, "decision": decision, "direction": "positive = A better than B (more test images correct)",
                          "bootstrap_seed": BOOT_BASE + len(contrasts) + 1, "note": note})

    # F1 primary contrasts, Holm within each condition (protocol Section 3)
    for c in FULL + LOW:
        others = ("mlp_param", "naive_branch", "dann_random") if c in FULL else ("mlp_param", "naive_branch")
        for m in others:
            add(f"F1:{c}:dann_lrf-{m}", f"F1:{c}", "confirmatory", cell("main_grid", c, "dann_lrf"), cell("main_grid", c, m), S20,
                decision="rq2" if (m == "naive_branch" and c in FULL) else "label")
    # F2 references, Holm over 11 (A5)
    for c in FULL:
        for m in ("compact_cnn", "lc_net", "sparse_mlp"):
            add(f"F2:{c}:dann_lrf-{m}", "F2", "confirmatory", cell("main_grid", c, "dann_lrf"), cell("F2_baselines", c, m), S20, decision="label",
                note="reference bound; cannot change an attribution statement")
    for c in ("cifar_full", "cifar_low02"):
        add(f"F2:{c}:dann_lrf-mlp_matched", "F2", "confirmatory", cell("main_grid", c, "dann_lrf"), cell("F2_baselines", c, "mlp_matched"), S20,
            decision="label", note="reference bound")
    # F5 difference-in-differences + distributional sanity, Holm over 4 (A5, PCR-008)
    for d in ("fashion", "cifar"):
        add(f"F5:{d}:did_lrf-random", "F5", "confirmatory",
            {"minuend": cell("F5_shuffled", f"{d}_full_shuffled", "dann_lrf"), "subtrahend": cell("F5_shuffled", f"{d}_full_shuffled", "dann_random")},
            {"minuend": cell("main_grid", f"{d}_full", "dann_lrf"), "subtrahend": cell("main_grid", f"{d}_full", "dann_random")},
            S20, decision="locality", kind="did",
            note="A-B = (LRF-RANDOM)_permuted - (LRF-RANDOM)_unpermuted; negative supports the locality explanation")
        add(f"F5:{d}:mlp_param_permuted-unpermuted", "F5", "confirmatory", cell("F5_shuffled", f"{d}_full_shuffled", "mlp_param"),
            cell("main_grid", f"{d}_full", "mlp_param"), S20, decision="sanity",
            note="distributional sanity only: MLP-Param is permutation invariant in function class and initial distribution, not per seed")
    # F6, Holm over 2 (A5)
    for c in ("cifar_full", "cifar_low02"):
        add(f"F6:{c}:dann_lrf_channel-dann_lrf", "F6", "confirmatory", cell("F6_channel", c, "dann_lrf_channel"), cell("main_grid", c, "dann_lrf"), S20,
            decision="label")
    # F8b matched initialisation, Holm over 3, RQ2 rule (A5, PCR-003)
    for c in FULL:
        add(f"F8b:{c}:dann_lrf-naive_branch_matched_init", "F8b", "confirmatory", cell("main_grid", c, "dann_lrf"),
            cell("F8_fairness", c, "naive_branch_matched_init"), S20, decision="rq2_controlled")
    # F9 conv-stem triad, Holm over 4 (A5, PCR-001)
    for c in ("cifar_full", "fashion_full"):
        add(f"F9:{c}:stem_dann_lrf-stem_naive_branch", "F9", "confirmatory", cell("F9_convstem", c, "stem_dann_lrf"),
            cell("F9_convstem", c, "stem_naive_branch"), S20, decision="transfer")
        add(f"F9:{c}:stem_dann_lrf-stem_mlp", "F9", "confirmatory", cell("F9_convstem", c, "stem_dann_lrf"), cell("F9_convstem", c, "stem_mlp"), S20,
            decision="label")

    descriptive = []
    for c in [f"{d}_full_K{k}_B{b}" for d in ("fashion", "kmnist") for k in (8, 16, 32) for b in (2, 4, 8)] + [f"{d}_full_P{p}" for d in ("fashion", "kmnist") for p in (2, 6)]:
        descriptive.append({"id": f"F3:{c}:dann_lrf", "kind": "cell", "cell": cell("F3_sensitivity", c, "dann_lrf"), "seeds": S5})
        if c.startswith("kmnist") and "_K" in c:
            descriptive.append({"id": f"F3:{c}:dann_lrf-dann_random", "kind": "pair", "A": cell("F3_sensitivity", c, "dann_lrf"),
                                "B": cell("F3_sensitivity", c, "dann_random"), "seeds": S5, "bootstrap_seed": BOOT_BASE + 500 + len(descriptive)})
    for src in ("rand_data", "rand_init", "rand_routing"):
        descriptive.append({"id": f"F4:fashion_full_{src}", "kind": "sd_across_values", "A": cell("F4_randomness", f"fashion_full_{src}", "dann_lrf"),
                            "B": cell("F4_randomness", f"fashion_full_{src}", "naive_branch"), "seeds": S10,
                            "note": "conditional sensitivity around one background point (PCR-007); the value 0 point is shared by the three sweeps"})
    for c in FULL:
        for a in ("a0", "a0p05", "a0p1", "a0p2", "a0p5", "a1"):
            descriptive.append({"id": f"F7:{c}:dann_lrf_slope_{a}", "kind": "cell", "cell": cell("F7_slope", c, f"dann_lrf_slope_{a}"), "seeds": S10})
    for c in FULL:
        for lr_cond, fam in ((f"{c}_lr3em4", "F8_fairness"), (c, "main_grid"), (f"{c}_lr3em3", "F8_fairness")):
            descriptive.append({"id": f"F8a:{lr_cond if fam == 'F8_fairness' else c + '_lr1em3'}:dann_lrf-naive_branch", "kind": "pair",
                                "A": cell(fam, lr_cond, "dann_lrf"), "B": cell(fam, lr_cond, "naive_branch"), "seeds": S5,
                                "bootstrap_seed": BOOT_BASE + 700 + len(descriptive),
                                "note": "descriptive learning-rate sensitivity; no selection, no equivalence claim at n = 5"})

    composite = [
        {"id": "RQ2_robustness", "inputs": [f"F1:{c}:dann_lrf-naive_branch" for c in FULL],
         "rule": "robust if >= 2 of the 3 full-data contrasts are 'supported'; otherwise not_robust and the abstract, highlights and "
                 "conclusion are narrowed; every 'reverse' contrast is named in the text as a dataset where Naive-Branch is better"},
        {"id": "F8b_vs_F1", "inputs": [[f"F1:{c}:dann_lrf-naive_branch", f"F8b:{c}:dann_lrf-naive_branch_matched_init"] for c in FULL],
         "rule": "per condition: F1 supported and F8b not supported -> the F1 difference in that condition is not attributed to the dendrite "
                 "nonlinearity (initialisation order cannot be excluded); F8b supported and F1 not -> F8b is reported as the controlled estimate; "
                 "the attribution is always worded as the total effect of the dendrite activation under the shared training protocol"},
        {"id": "F9_transfer", "inputs": [f"F9:{c}:stem_dann_lrf-stem_naive_branch" for c in ("cifar_full", "fashion_full")],
         "rule": "transfer observed if both supported; dataset-specific if exactly one supported; not observed otherwise; any reverse is named"},
        {"id": "F5_locality", "inputs": [f"F5:{d}:did_lrf-random" for d in ("fashion", "cifar")],
         "rule": "per dataset: holm_p < 0.05 and ci95_high < 0 -> the LRF advantage over RANDOM shrinks under permutation (locality explanation "
                 "supported); otherwise report the measured change and its CI; 'disappears' is never claimed without an equivalence margin"},
    ]
    a4 = []
    for c in FULL:
        for s in S10:
            a4.append({"id": f"A4:{c}:slope_a0p1=dann_lrf:s{s:02d}", "left": dict(cell("F7_slope", c, "dann_lrf_slope_a0p1"), seed=s),
                       "right": dict(cell("main_grid", c, "dann_lrf"), seed=s)})
            a4.append({"id": f"A4:{c}:slope_a1=nb_matched_init:s{s:02d}", "left": dict(cell("F7_slope", c, "dann_lrf_slope_a1"), seed=s),
                       "right": dict(cell("F8_fairness", c, "naive_branch_matched_init"), seed=s)})
    return {
        "schema_version": 1,
        "artifact_kind": "r1_analysis_manifest",
        "freeze_id": "NEURO-R1-FREEZE-001",
        "change_id": "MC-NEURO-R1-001",
        "primary_metric": "test accuracy at the validation-selected epoch, RE-DERIVED from history.csv (first epoch of minimum val_loss); "
                          "analysed as the integer number of correctly classified test images out of n_test",
        "n_test": 10000,
        "statistics": {
            "unit": "seed; paired within condition; differences are integer counts of correct test images",
            "test": "scipy.stats.wilcoxon(d, zero_method='wilcox', correction=False, alternative='two-sided', method=M) with M='exact' when "
                    "no difference is zero, no absolute difference is tied and n <= 50, otherwise M='approx'; the method used is written per contrast",
            "all_zero": "all differences zero -> p_two_sided = 1.0, outcome 'no_difference', d_z undefined",
            "bootstrap": "paired percentile 95 % CI of the mean integer difference, R = 10000 resamples, numpy.random.default_rng(bootstrap_seed)",
            "effect_sizes": "d_z = mean / sd (ddof = 1) of the integer differences, 'undefined' when sd = 0; matched-pairs rank-biserial "
                            "over |d| with average ranks, zeros dropped",
            "multiplicity": "Holm step-down within each holm_group",
            "reporting_scale": "differences reported in percentage points = 100 * count / n_test",
            "pinned_versions": {"python": "3.12.12", "numpy": "2.3.5", "scipy": "1.18.0"},
        },
        "decision_rule": {
            "supported": "holm_p < 0.05 and ci95_low > 0",
            "reverse": "holm_p < 0.05 and ci95_high < 0",
            "not_supported": "otherwise, including no_difference",
            "missing_or_invalid_unit": "no decision for the affected contrast; the run is incomplete and nothing is promoted",
            "sanity": "F5 MLP-Param contrasts: a 'supported' or 'reverse' label flags an engineering problem to investigate, not a finding",
        },
        "contrasts": contrasts,
        "descriptive": descriptive,
        "composite_rules": composite,
        "replication": {
            "r0_paired_tests": "paper_package/derived/paired_tests_validation_selected.csv",
            "seeds": S10,
            "rule": "for each r0 contrast (condition_key, A, B) recompute the same contrast on R1 seeds 0-9, Holm within condition over the r0 "
                    "contrast set of that condition; report sign agreement and Holm-decision agreement (r0: p_holm_per_condition < 0.05); descriptive",
        },
        "a4_checks": {
            "definition": "equality of the epoch column and the six metric columns (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc) "
                          "of history.csv; dataset and seed columns must also match; model_name is expected to differ and is checked against the "
                          "unit's own model id (PCR-010)",
            "gate": "all pairs equal before any F7 or F8 contrast is interpreted",
            "pairs": a4,
        },
        "pre_freeze_f1_crosscheck": {
            "left_run": "2026-10-03_claude_mta_cuda_r1_main_grid (pre-freeze, blind)",
            "rule": "after the decision artifact exists and the frozen run is delivered: per F1 unit, equality of all history.csv columns; "
                    "a mismatch is a determinism finding and never substitutes old for new results",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    print(f"contrasts {len(doc['contrasts'])} descriptive {len(doc['descriptive'])} a4 {len(doc['a4_checks']['pairs'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
