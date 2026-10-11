"""Pre-training structural checks of the R1 models (engineering, no training; review PCR-002 and PCR-004).

    python r1/check_models.py [--seeds 20]

* random-sparse MLP (amendment A6): for every seed and both input sizes, every first-layer unit has inputs and at least two
  output edges, every second-layer unit has inputs, no duplicate edges, and the effective parameter count equals the
  DANN budget;
* conv-stem triad (amendment A7): for every seed, the stem, the routing indices and the six head tensors of
  stem_naive_branch equal those of stem_dann_lrf before training, built with the runner's own seeding sequence.
Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from r1.r1_models import SparseMLP, build_r1_model, target_budget  # noqa: E402
from src.train_eval import set_seed  # noqa: E402


class Info:
    def __init__(self, input_dim, num_classes):
        self.input_dim, self.num_classes = input_dim, num_classes


def unit(dataset, model, seed):
    return {"dataset": dataset, "model": model, "seed": seed, "soma_units": 128, "branches_per_soma": 4, "sample_size": 16,
            "patch_h": 4, "patch_w": 4, "extra": {"r1_model": model}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    fails = []
    budget = target_budget(128, 4, 16, 10)
    for dim, name in ((784, "28x28x1"), (3072, "32x32x3")):
        allocs = set()
        for s in range(args.seeds):
            m = SparseMLP(dim, 10, budget, mask_seed=s)
            m1, m2 = m.fc1.mask, m.fc2.mask
            ok = (int((m1.sum(1) > 0).sum()) == 512 and int((m2.sum(0) >= 2).sum()) == 512 and int((m2.sum(1) > 0).sum()) == 128
                  and m.effective_params == budget and float(m1.max()) == 1.0 and float(m2.max()) == 1.0
                  and int(m1.sum()) + int(m2.sum()) == budget - (512 + 128) - 129 * 10
                  and int(((m.fc1.weight != 0) & (m1 == 0)).sum()) == 0 and int(((m.fc2.weight != 0) & (m2 == 0)).sum()) == 0)
            allocs.add(str(m.allocation))
            if not ok:
                fails.append(f"sparse_mlp {name} seed {s}")
        print(f"sparse_mlp {name}: {args.seeds} seeds checked; allocations seen: {sorted(allocs)[:2]}{' ...' if len(allocs) > 2 else ''}")
    for dataset, dim in (("fashionmnist", 784), ("cifar10", 3072)):
        for s in range(args.seeds):
            set_seed(s)
            a = build_r1_model(unit(dataset, "stem_dann_lrf", s), Info(dim, 10), s)
            set_seed(s)
            b = build_r1_model(unit(dataset, "stem_naive_branch", s), Info(dim, 10), s)
            set_seed(s)
            c = build_r1_model(unit(dataset, "stem_mlp", s), Info(dim, 10), s)
            stem_eq = all(torch.equal(x, y) for x, y in zip(a.stem.state_dict().values(), b.stem.state_dict().values()))
            stem_eq_mlp = all(torch.equal(x, y) for x, y in zip(a.stem.state_dict().values(), c.stem.state_dict().values()))
            ha, hb = a.head, b.head
            heads = [(ha.dendritic.synaptic_weights, hb.synaptic_weights), (ha.dendritic.synaptic_bias, hb.synaptic_bias),
                     (ha.dendritic.cable_weights, hb.cable_weights), (ha.dendritic.soma_bias, hb.soma_bias),
                     (ha.classifier.weight, hb.classifier.weight), (ha.classifier.bias, hb.classifier.bias)]
            head_eq = all(torch.equal(x, y) for x, y in heads)
            idx_eq = torch.equal(ha.dendritic.dendrite_indices, hb.dendrite_indices)
            if not (stem_eq and stem_eq_mlp and head_eq and idx_eq):
                fails.append(f"stem {dataset} seed {s}: stem={stem_eq} stem_mlp={stem_eq_mlp} head={head_eq} idx={idx_eq}")
        print(f"conv-stem {dataset}: {args.seeds} seeds checked (stem, routing, six head tensors)")
    print("FAILED:", fails[:10] if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
