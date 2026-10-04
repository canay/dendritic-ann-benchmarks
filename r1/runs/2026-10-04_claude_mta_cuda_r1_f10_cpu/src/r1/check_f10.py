"""Pre-run structural checks for amendments A11 (F10) and A12 (spatial-head CNN); engineering only, no training.

    python r1/check_f10.py

* spatial-head CNN: the A12 width search result, the parameter count equal to the search's own count, the head input
  size, the forward output shape, and the A1 width rule (c1 <= c2 <= 4*c1);
* for every F10 model on both F10 datasets: the two MAC methods of r1/f10_macs.py agree exactly (implemented and
  effective), and the trainable parameter count is printed next to the budget;
* the F10 and CNN-arm plans have the planned unit counts and unique unit ids.
Exit 0 only if every check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from r1 import r1_plan  # noqa: E402
from r1.f10_macs import closed_form, counted  # noqa: E402
from r1.r1_models import CompactCNNFlat, search_cnn_flat, target_budget  # noqa: E402
from r1.r1_runner import build_unit_model  # noqa: E402
from src.sampling import DATASET_SPECS  # noqa: E402
from src.train_eval import set_seed  # noqa: E402


class Info:
    def __init__(self, input_dim: int, num_classes: int) -> None:
        self.input_dim, self.num_classes = input_dim, num_classes


def main() -> int:
    fails = []
    budget = target_budget(128, 4, 16, 10)
    for ds in ("fashionmnist", "kmnist", "cifar10"):
        spec = DATASET_SPECS[ds]
        c1, c2, p = search_cnn_flat(spec, 10, budget)
        set_seed(0)
        m = CompactCNNFlat(spec, 10, c1, c2)
        count = sum(t.numel() for t in m.parameters() if t.requires_grad)
        out = m(torch.zeros(2, spec.channels * spec.height * spec.width))
        ok = (count == p and c1 <= c2 <= 4 * c1 and m.head_inputs == c2 * (spec.height // 4) * (spec.width // 4)
              and tuple(out.shape) == (2, 10))
        print(f"cnn_flat {ds}: c1={c1} c2={c2} params={count} (search {p}, budget {budget}, gap {count - budget:+d}) "
              f"head_inputs={m.head_inputs} ok={ok}")
        if not ok:
            fails.append(f"cnn_flat {ds}")
    plan_units = r1_plan.f10("cpu")
    seen = set()
    for u in plan_units:
        key = (u["dataset"], u["model"])
        if key in seen:
            continue
        seen.add(key)
        spec = DATASET_SPECS[u["dataset"]]
        info = Info(spec.channels * spec.height * spec.width, 10)
        set_seed(0)
        model = build_unit_model(u, info, 0)
        a = closed_form(u, info.input_dim, info.num_classes)
        b = counted(model, info.input_dim)
        params = sum(t.numel() for t in model.parameters() if t.requires_grad)
        ok = a == b
        print(f"MACs {u['dataset']:<12} {u['model']:<13} closed_form={a} counted={b} params={params} agree={ok}")
        if not ok:
            fails.append(f"MACs {u['dataset']} {u['model']}")
    checks = {"f10_gpu": (r1_plan.f10("cuda"), 51), "f10_cpu": (plan_units, 51), "cnn_flat_arm": (r1_plan.cnn_flat_arm(), 66),
              "cnn_flat_smoke": (r1_plan.cnn_flat_smoke(), 3), "f10_smoke_cpu": (r1_plan.f10_smoke("cpu"), 2)}
    for name, (units, expected) in checks.items():
        ids = [u["unit_id"] for u in units]
        ok = len(ids) == expected and len(set(ids)) == len(ids)
        print(f"plan {name}: {len(ids)} units (expected {expected}), unique={len(set(ids)) == len(ids)}")
        if not ok:
            fails.append(f"plan {name}")
    print("FAILED:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
