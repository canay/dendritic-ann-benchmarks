"""Forward multiply-accumulate counts per image for the F10 table (protocol amendment A11).

Method A (``closed_form``) derives the count from the architecture hyperparameters and the deterministic width
searches. Method B (``counted``) counts on the instantiated module: matrix and convolution products through
``torch.utils.flop_counter.FlopCounterMode`` (two FLOPs per MAC) plus the gathered-dendrite and cable products, which
are element-wise operations the FLOP counter does not see, read from the dendrite layers' own buffers and parameters.
Both exclude biases, activations and pooling. "implemented" is what the code computes per image; "effective" counts
only the non-zero weights; the two differ only for the random-sparse MLP, which is computed as a dense masked product.
A timing unit is valid only when both methods agree exactly.
"""
from __future__ import annotations

import copy
from typing import Dict

import torch

from r1.r1_models import (MaskedLinear, lc_positions, search_cnn, search_cnn_flat, search_lc, sparse_mlp_allocation,
                          target_budget)
from src.models import FixedMaskedDendriteLayer, NaiveBranchedLinear, estimate_param_matched_width
from src.sampling import DATASET_SPECS


def closed_form(unit: Dict, input_dim: int, num_classes: int) -> Dict[str, int]:
    extra = unit.get("extra") or {}
    kind = extra.get("r1_model") or unit["model"]
    S, B, K = int(unit["soma_units"]), int(unit["branches_per_soma"]), int(unit["sample_size"])
    C, D = int(num_classes), int(input_dim)
    n = S * B
    spec = DATASET_SPECS[unit["dataset"]]
    H, W, cin = spec.height, spec.width, spec.channels
    target = target_budget(S, B, K, C)
    if kind in ("dann_lrf", "dann_random", "dann_grf", "naive_branch"):
        m = n * K + n + S * C
        return {"implemented": m, "effective": m}
    if kind == "vann_same":
        m = D * n + n * S + S * C
        return {"implemented": m, "effective": m}
    if kind == "mlp_param":
        w = estimate_param_matched_width(target, D, C)
        m = D * w + w * w + w * C
        return {"implemented": m, "effective": m}
    if kind == "mlp_matched":
        dims = [D] + [int(v) for v in extra["widths"]]
        m = sum(a * b for a, b in zip(dims[:-1], dims[1:])) + dims[-1] * C
        return {"implemented": m, "effective": m}
    if kind == "compact_cnn":
        c1, c2, _ = search_cnn(cin, C, target)
        m = H * W * c1 * cin * 9 + (H // 2) * (W // 2) * c2 * c1 * 9 + c2 * C
        return {"implemented": m, "effective": m}
    if kind == "cnn_flat":
        c1, c2, _ = search_cnn_flat(spec, C, target)
        m = H * W * c1 * cin * 9 + (H // 2) * (W // 2) * c2 * c1 * 9 + c2 * (H // 4) * (W // 4) * C
        return {"implemented": m, "effective": m}
    if kind == "lc_net":
        stride, ch, hidden, _ = search_lc(spec, C, target)
        positions = lc_positions(H, 4, stride) * lc_positions(W, 4, stride)
        m = positions * ch * 16 * cin + positions * ch * hidden + hidden * C
        return {"implemented": m, "effective": m}
    if kind == "sparse_mlp":
        n1, n2 = sparse_mlp_allocation(D, C, target)
        return {"implemented": D * 512 + 512 * 128 + 128 * C, "effective": n1 + n2 + 128 * C}
    raise ValueError(f"no closed-form MAC count for {kind}")


def counted(model: torch.nn.Module, input_dim: int) -> Dict[str, int]:
    from torch.utils.flop_counter import FlopCounterMode

    probe = copy.deepcopy(model).to("cpu").eval()
    x = torch.zeros(1, int(input_dim))
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        probe(x)
    flops = int(counter.get_total_flops())
    if flops % 2:
        raise RuntimeError("odd FLOP total; the counter did not count multiply-accumulates")
    implemented = flops // 2
    for mod in probe.modules():
        if isinstance(mod, (FixedMaskedDendriteLayer, NaiveBranchedLinear)):
            implemented += int(mod.dendrite_indices.numel()) + int(mod.cable_weights.numel())
    effective = implemented
    for mod in probe.modules():
        if isinstance(mod, MaskedLinear):
            effective -= int(mod.weight.numel()) - int(mod.mask.sum().item())
    return {"implemented": int(implemented), "effective": int(effective)}
