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


FULL3 = ("fashionmnist", "kmnist", "cifar10")


def cnn_flat_arm():
    """MC-NEURO-R1-002 (amendment A12, exploratory): six lineage anchors that repeat frozen units (F1 DANN-LRF and F2
    compact CNN, seed 0, each full-data dataset), then the spatial-head CNN at seeds 0-19 on the three full-data datasets."""
    units = []
    for ds in FULL3:
        units.append(unit("MC002_anchor", f"{SHORT[ds]}_full", ds, 1.0, "dann_lrf", 0))
        units.append(r1u("MC002_anchor", f"{SHORT[ds]}_full", ds, 1.0, "compact_cnn", 0, r1_model="compact_cnn"))
    for seed in range(20):
        for ds in FULL3:
            units.append(r1u("MC002_cnn_flat", f"{SHORT[ds]}_full", ds, 1.0, "cnn_flat", seed, r1_model="cnn_flat"))
    return units


def cnn_flat_smoke():
    """Two-epoch engineering smoke of the spatial-head CNN (not evidence)."""
    return [r1u("smoke", f"smk_{SHORT[ds]}", ds, 1.0, "cnn_flat", 0, r1_model="cnn_flat", epochs=2) for ds in FULL3]


# F10 (amendment A11): the protocol's model list plus the exploratory spatial-head CNN of MC-NEURO-R1-002
F10_MODELS = [("dann_lrf", None), ("naive_branch", None), ("mlp_param", None), ("vann_same", None),
              ("compact_cnn", "compact_cnn"), ("lc_net", "lc_net"), ("sparse_mlp", "sparse_mlp"), ("cnn_flat", "cnn_flat")]
F10_SCHEDULE = {"warmup_epochs": 1, "timed_epochs": 5, "warmup_passes": 1, "timed_passes": 5}


def f10(device: str, seeds=(0, 1, 2), schedule=None, datasets=("fashionmnist", "cifar10"), models=None):
    """One timing unit = one (device, dataset, model, seed); seed-major order so drift spreads over the models."""
    sched = dict(schedule or F10_SCHEDULE)
    units = []
    for seed in seeds:
        for ds in datasets:
            cell = list(models or F10_MODELS)
            if ds == "cifar10" and models is None:
                cell.append(("mlp_matched", "mlp_matched"))
            for label, kind in cell:
                extra = {"device": device, "timing": sched}
                if kind == "mlp_matched":
                    extra["widths"] = [3, 100]
                units.append(r1u("F10_timing", f"{device}_{SHORT[ds]}_full", ds, 1.0, label, seed, r1_model=kind, extra=extra,
                                 epochs=sched["warmup_epochs"] + sched["timed_epochs"]))
    return units


def f10_smoke(device: str):
    """Engineering smoke of the timing runner: two models, one seed, a shortened schedule (not evidence)."""
    sched = {"warmup_epochs": 1, "timed_epochs": 2, "warmup_passes": 1, "timed_passes": 2}
    return f10(device, seeds=(0,), schedule=sched, datasets=("fashionmnist",), models=[("dann_lrf", None), ("cnn_flat", "cnn_flat")])


# MC-NEURO-R1-003 (bounded tier D extension; protocol Section 10, amendments A14-A16)
TF_HEADS = ("tf_stem_dann_lrf", "tf_stem_naive_branch", "tf_stem_mlp")
D1_DATASETS = ("fashionmnist", "cifar10")
D2_MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random", "vann_same")
D2_CONDITION = "cifar100_full"
# Lineage anchors: seed-0 units of NEURO-R1-FREEZE-001 repeated with the new code snapshot. The F1 units cover the r0 code
# path of every D2 model on the closest dataset; the F9 units cover the stem dispatch and every head the D1 kinds mirror.
TIERD_ANCHORS = ([("cifar10", m, None) for m in D2_MODELS] + [("fashionmnist", "stem_dann_lrf", "stem_dann_lrf")]
                 + [("cifar10", m, m) for m in ("stem_dann_lrf", "stem_naive_branch", "stem_mlp")])
TIERD_PILOT_SEED = 100


def tierd_arm():
    """A14-A16: nine lineage anchors first, then D1 (Transformer front end with the three F9 heads; FashionMNIST and CIFAR-10)
    and D2 (CIFAR-100; the five core models) in seed-major order, seeds 0-19: 9 + 120 + 100 = 229 units."""
    units = [r1u("MC003_anchor", f"{SHORT[ds]}_full", ds, 1.0, m, 0, r1_model=kind) for ds, m, kind in TIERD_ANCHORS]
    for seed in range(20):
        for ds in D1_DATASETS:
            for m in TF_HEADS:
                units.append(r1u("MC003_D1_tfstem", f"{SHORT[ds]}_full", ds, 1.0, m, seed, r1_model=m))
        for m in D2_MODELS:
            units.append(unit("MC003_D2_cifar100", D2_CONDITION, "cifar100", 1.0, m, seed))
    return units


# DEC-NEURO-020 (amendment A17): the pre-specified 100-epoch budget-sensitivity arm of D1 and D2. Its conditions carry the
# suffix _e100 so that its unit ids never collide with the 30-epoch units of the same cell and seed.
TIERD_BUDGET_EPOCHS = 100


def tierd_run():
    """A14-A17: the 229 units of tierd_arm unchanged and first (anchors, then the 30-epoch primary D1 and D2), then the
    100-epoch arm of D1 and D2 in seed-major order, seeds 0-19: 229 + 120 + 100 = 449 units."""
    e = {"epochs": TIERD_BUDGET_EPOCHS}
    units = tierd_arm()
    for seed in range(20):
        for ds in D1_DATASETS:
            for m in TF_HEADS:
                units.append(r1u("MC003_D1_tfstem_e100", f"{SHORT[ds]}_full_e100", ds, 1.0, m, seed, r1_model=m, **e))
        for m in D2_MODELS:
            units.append(unit("MC003_D2_cifar100_e100", f"{D2_CONDITION}_e100", "cifar100", 1.0, m, seed, **e))
    return units


def tierd_smoke():
    """Two-epoch engineering smoke of every new kind and dataset (not evidence; run twice to show determinism)."""
    e = {"epochs": 2}
    units = [r1u("smoke", f"smk_{SHORT[ds]}", ds, 1.0, m, 0, r1_model=m, **e) for ds in D1_DATASETS for m in TF_HEADS]
    return units + [unit("smoke", "smk_cifar100", "cifar100", 1.0, m, 0, **e) for m in D2_MODELS]


def tierd_pilot():
    """Reference-adequacy pilot (STUDY_DESIGN 'Baselines And Comparators'; pilot_only, never evidence): every D1 and D2
    model at the full training budget with one seed outside the evidence seeds 0-19."""
    s = TIERD_PILOT_SEED
    units = [r1u("MC003_pilot", f"{SHORT[ds]}_full", ds, 1.0, m, s, r1_model=m) for ds in D1_DATASETS for m in TF_HEADS]
    return units + [unit("MC003_pilot", D2_CONDITION, "cifar100", 1.0, m, s) for m in D2_MODELS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["main_grid", "smoke", "controls", "controls_smoke", "frozen_all", "cnn_flat_arm",
                                       "cnn_flat_smoke", "f10_gpu", "f10_cpu", "f10_smoke_gpu", "f10_smoke_cpu",
                                       "tierd_arm", "tierd_smoke", "tierd_pilot", "tierd_run"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    args = ap.parse_args()
    builders = {"main_grid": lambda: main_grid(args.seeds), "smoke": smoke, "controls": controls, "controls_smoke": controls_smoke,
                "frozen_all": frozen_all, "cnn_flat_arm": cnn_flat_arm, "cnn_flat_smoke": cnn_flat_smoke,
                "f10_gpu": lambda: f10("cuda"), "f10_cpu": lambda: f10("cpu"),
                "f10_smoke_gpu": lambda: f10_smoke("cuda"), "f10_smoke_cpu": lambda: f10_smoke("cpu"),
                "tierd_arm": tierd_arm, "tierd_smoke": tierd_smoke, "tierd_pilot": tierd_pilot, "tierd_run": tierd_run}
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
