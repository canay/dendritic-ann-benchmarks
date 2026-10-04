"""Deterministic unit plans for the R1 evidence families (see MD/09_audit_revision/R1_EVIDENCE_PROTOCOL.md).

    python r1/r1_plan.py main_grid --run-id <run id> --out <plan.json>
    python r1/r1_plan.py smoke --run-id <run id> --out <plan.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = {
    "epochs": 30, "batch_size": 256, "lr": 1e-3, "val_fraction": 0.1,
    "soma_units": 128, "branches_per_soma": 4, "sample_size": 16, "patch_h": 4, "patch_w": 4,
}
ALL6 = ["dann_lrf", "dann_random", "dann_grf", "naive_branch", "mlp_param", "vann_same"]
# Condition -> (dataset, training-subset fraction, model set). Reduced-data model sets equal r0's.
MAIN_CONDITIONS = {
    "fashion_full": ("fashionmnist", 1.0, ALL6),
    "kmnist_full": ("kmnist", 1.0, ALL6),
    "cifar_full": ("cifar10", 1.0, ALL6),  # r0 lacked dann_random/dann_grf here (R3-6)
    "fashion_low02": ("fashionmnist", 0.2, ["dann_lrf", "naive_branch", "mlp_param"]),
    "fashion_low01": ("fashionmnist", 0.1, ["dann_lrf", "naive_branch", "mlp_param"]),
    "cifar_low02": ("cifar10", 0.2, ["dann_lrf", "naive_branch", "mlp_param", "vann_same"]),
}


def unit(family: str, condition: str, dataset: str, subset: float, model: str, seed: int, **over):
    u = {"unit_id": f"{condition}__{model}__s{seed:02d}", "family": family, "condition": condition,
         "dataset": dataset, "subset_fraction": subset, "model": model, "seed": seed, "extra": {}}
    u.update(BASE)
    u.update(over)
    return u


def main_grid(seeds):
    units = []
    for seed in seeds:  # seed-major order: every worker receives a mix of conditions
        for condition, (dataset, subset, models) in MAIN_CONDITIONS.items():
            for model in models:
                units.append(unit("main_grid", condition, dataset, subset, model, seed))
    return units


def smoke():
    units = []
    for seed in (0, 1):
        for model in ("dann_lrf", "naive_branch", "mlp_param"):
            units.append(unit("smoke", "smoke_fashion_sub005", "fashionmnist", 0.05, model, seed, epochs=3))
    return units


SHORT = {"fashionmnist": "fashion", "kmnist": "kmnist", "cifar10": "cifar"}


def r1u(family, condition, dataset, subset, label, seed, r1_model=None, extra=None, **over):
    u = unit(family, condition, dataset, subset, label, seed, **over)
    ex = dict(extra or {})
    if r1_model:
        ex["r1_model"] = r1_model
    u["extra"] = ex
    return u


def slope_tag(a: float) -> str:
    return ("a" + f"{a:g}").replace(".", "p")


def controls():
    """F2-F9 of the locked protocol (amendments A1-A4)."""
    units = []
    s20, s10, s5 = range(20), range(10), range(5)
    full = ("fashionmnist", "kmnist", "cifar10")
    # F2 reference baselines
    for seed in s20:
        for ds in full:
            for m in ("compact_cnn", "lc_net", "sparse_mlp"):
                units.append(r1u("F2_baselines", f"{SHORT[ds]}_full", ds, 1.0, m, seed, r1_model=m))
        for cond, sub in (("cifar_full", 1.0), ("cifar_low02", 0.2)):
            units.append(r1u("F2_baselines", cond, "cifar10", sub, "mlp_matched", seed, r1_model="mlp_matched", extra={"widths": [3, 100]}))
    # F3 sensitivity (KMNIST + FashionMNIST repeat on the R1 lineage)
    for ds in ("kmnist", "fashionmnist"):
        for seed in s5:
            for k in (8, 16, 32):
                for b in (2, 4, 8):
                    units.append(r1u("F3_sensitivity", f"{SHORT[ds]}_full_K{k}_B{b}", ds, 1.0, "dann_lrf", seed, sample_size=k, branches_per_soma=b))
                    if ds == "kmnist":
                        units.append(r1u("F3_sensitivity", f"{SHORT[ds]}_full_K{k}_B{b}", ds, 1.0, "dann_random", seed, sample_size=k, branches_per_soma=b))
            for p in (2, 6):
                units.append(r1u("F3_sensitivity", f"{SHORT[ds]}_full_P{p}", ds, 1.0, "dann_lrf", seed, patch_h=p, patch_w=p))
    # F4 randomness decomposition
    for source in ("routing", "init", "data"):
        for v in s10:
            seeds = {"routing_seed": 0, "init_seed": 0, "data_seed": 0}
            seeds[f"{source}_seed"] = v
            for m in ("dann_lrf", "naive_branch"):
                units.append(r1u("F4_randomness", f"fashion_full_rand_{source}", "fashionmnist", 1.0, m, v, extra=seeds))
    # F5 shuffled-pixel control
    for ds in ("fashionmnist", "cifar10"):
        for seed in s20:
            for m in ("dann_lrf", "dann_random", "naive_branch", "mlp_param"):
                units.append(r1u("F5_shuffled", f"{SHORT[ds]}_full_shuffled", ds, 1.0, m, seed, extra={"pixel_permutation_seed": 1000 + seed}))
    # F6 channel-aware routing
    for cond, sub in (("cifar_full", 1.0), ("cifar_low02", 0.2)):
        for seed in s20:
            units.append(r1u("F6_channel", cond, "cifar10", sub, "dann_lrf_channel", seed, r1_model="dann_lrf_channel"))
    # F7 dendrite-slope dose-response
    for ds in full:
        for a in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            for seed in s10:
                units.append(r1u("F7_slope", f"{SHORT[ds]}_full", ds, 1.0, f"dann_lrf_slope_{slope_tag(a)}", seed,
                                 r1_model="dann_lrf_slope", extra={"dendrite_slope": a}))
    # F8 optimisation fairness
    for ds in full:
        for lr, tag in ((3e-4, "lr3em4"), (3e-3, "lr3em3")):
            for m in ("dann_lrf", "naive_branch"):
                for seed in s5:
                    units.append(r1u("F8_fairness", f"{SHORT[ds]}_full_{tag}", ds, 1.0, m, seed, lr=lr))
        for seed in s20:
            units.append(r1u("F8_fairness", f"{SHORT[ds]}_full", ds, 1.0, "naive_branch_matched_init", seed, r1_model="naive_branch_matched_init"))
    # F9 conv-stem triad
    for ds in ("fashionmnist", "cifar10"):
        for seed in s20:
            for m in ("stem_dann_lrf", "stem_naive_branch", "stem_mlp"):
                units.append(r1u("F9_convstem", f"{SHORT[ds]}_full", ds, 1.0, m, seed, r1_model=m))
    return units


PRIORITY = ["main_grid", "F8b", "F9_convstem", "F2_baselines", "F5_shuffled", "F6_channel", "F7_slope",
            "F3_sensitivity", "F4_randomness", "F8a"]


def frozen_all():
    """Amendment A10 (review PCR-009/013): the F1 re-run and F2-F9 as ONE frozen run, in the pre-registered priority
    order used if the compute ceiling forces a pause (stable sort keeps each family's seed-major order)."""
    units = main_grid(range(20)) + controls()

    def rank(u):
        fam = u["family"]
        if fam == "F8_fairness":
            fam = "F8b" if u["model"] == "naive_branch_matched_init" else "F8a"
        return PRIORITY.index(fam)

    return sorted(units, key=rank)


def controls_smoke():
    """Two-epoch engineering smoke of every new model + the A4 identity pairs (not evidence)."""
    units = []
    e = {"epochs": 2}
    for ds in ("fashionmnist", "cifar10"):
        for m in ("compact_cnn", "lc_net", "sparse_mlp", "stem_dann_lrf", "stem_naive_branch", "stem_mlp"):
            units.append(r1u("smoke", f"smk_{SHORT[ds]}", ds, 1.0, m, 0, r1_model=m, **e))
    units.append(r1u("smoke", "smk_cifar", "cifar10", 1.0, "mlp_matched", 0, r1_model="mlp_matched", extra={"widths": [3, 100]}, **e))
    units.append(r1u("smoke", "smk_cifar", "cifar10", 1.0, "dann_lrf_channel", 0, r1_model="dann_lrf_channel", **e))
    units.append(r1u("smoke", "smk_fashion", "fashionmnist", 1.0, "dann_lrf", 0, **e))
    units.append(r1u("smoke", "smk_fashion", "fashionmnist", 1.0, "dann_lrf_slope_a0p1", 0, r1_model="dann_lrf_slope", extra={"dendrite_slope": 0.1}, **e))
    units.append(r1u("smoke", "smk_fashion", "fashionmnist", 1.0, "dann_lrf_slope_a1", 0, r1_model="dann_lrf_slope", extra={"dendrite_slope": 1.0}, **e))
    units.append(r1u("smoke", "smk_fashion", "fashionmnist", 1.0, "naive_branch_matched_init", 0, r1_model="naive_branch_matched_init", **e))
    units.append(r1u("smoke", "smk_fashion_shuf", "fashionmnist", 1.0, "dann_lrf", 0, extra={"pixel_permutation_seed": 1000}, **e))
    units.append(r1u("smoke", "smk_fashion_rand", "fashionmnist", 1.0, "dann_lrf", 3, extra={"routing_seed": 3, "init_seed": 0, "data_seed": 0}, **e))
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["main_grid", "smoke", "controls", "controls_smoke", "frozen_all"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    args = ap.parse_args()
    builders = {"main_grid": lambda: main_grid(args.seeds), "smoke": smoke, "controls": controls, "controls_smoke": controls_smoke,
                "frozen_all": frozen_all}
    units = builders[args.family]()
    plan = {"schema_version": 1, "run_id": args.run_id, "family": args.family,
            "protocol": "MD/09_audit_revision/R1_EVIDENCE_PROTOCOL.md", "base": BASE, "units": units}
    ids = [u["unit_id"] for u in units]
    assert len(ids) == len(set(ids))
    Path(args.out).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.family}: {len(units)} units -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
