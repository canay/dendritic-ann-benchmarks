"""R1 reference baselines and controls (protocol MC-NEURO-R1-001, families F2 and F5-F9).

Every model takes the flattened input ``[batch, C*H*W]`` produced by the r0 data
pipeline. Parameter matching is an explicit, deterministic search against the
DANN budget of the same configuration (``target_budget``); the chosen settings
are returned by ``describe_r1_model`` and written into each unit result.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import DendriticANN, FixedMaskedDendriteLayer, FlatMLP, NaiveBranchedLinear, count_parameters, estimate_param_matched_width
from src.sampling import DATASET_SPECS, ImageSpec, build_dendrite_indices


# ------------------------------------------------------------------ budget
def target_budget(soma_units: int, branches: int, sample_size: int, num_classes: int) -> int:
    n = soma_units * branches
    return n * (sample_size + 1) + soma_units * (branches + 1) + (soma_units + 1) * num_classes


# ------------------------------------------------------------------ compact CNN (F2)
def cnn_params(cin: int, c1: int, c2: int, num_classes: int) -> int:
    return (9 * cin + 1) * c1 + (9 * c1 + 1) * c2 + (c2 + 1) * num_classes


def search_cnn(cin: int, num_classes: int, target: int) -> Tuple[int, int, int]:
    """Protocol amendment A1: non-decreasing channel width, at most x4 per stage (c1 <= c2 <= 4*c1)."""
    best = None
    for c1 in range(4, 65):
        for c2 in range(c1, 4 * c1 + 1):
            p = cnn_params(cin, c1, c2, num_classes)
            key = (abs(p - target), c1, c2)
            if best is None or key < best[0]:
                best = (key, c1, c2, p)
    return best[1], best[2], best[3]


class CompactCNN(nn.Module):
    def __init__(self, spec: ImageSpec, num_classes: int, c1: int, c2: int) -> None:
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(spec.channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec
        x = x.view(x.size(0), s.channels, s.height, s.width)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.mean(dim=(2, 3)))


# ------------------------------------------------------------------ spatial-head CNN (MC-NEURO-R1-002, exploratory)
def cnn_flat_params(spec: ImageSpec, c1: int, c2: int, num_classes: int) -> int:
    positions = (spec.height // 4) * (spec.width // 4)
    return (9 * spec.channels + 1) * c1 + (9 * c1 + 1) * c2 + (c2 * positions + 1) * num_classes


def search_cnn_flat(spec: ImageSpec, num_classes: int, target: int) -> Tuple[int, int, int]:
    """Amendment A12: the A1 width rule (c1 <= c2 <= 4*c1, c1 in 4..64) with a head that keeps every position."""
    best = None
    for c1 in range(4, 65):
        for c2 in range(c1, 4 * c1 + 1):
            p = cnn_flat_params(spec, c1, c2, num_classes)
            key = (abs(p - target), c1, c2)
            if best is None or key < best[0]:
                best = (key, c1, c2, p)
    return best[1], best[2], best[3]


class CompactCNNFlat(nn.Module):
    """The two conv stages of CompactCNN followed by a flattening linear head (no global average pooling)."""

    def __init__(self, spec: ImageSpec, num_classes: int, c1: int, c2: int) -> None:
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(spec.channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.head_inputs = c2 * (spec.height // 4) * (spec.width // 4)
        self.fc = nn.Linear(self.head_inputs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec
        x = x.view(x.size(0), s.channels, s.height, s.width)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.reshape(x.size(0), -1))


# ------------------------------------------------------------------ locally connected network (F2)
def lc_positions(size: int, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


def lc_params(spec: ImageSpec, stride: int, channels: int, hidden: int, num_classes: int, kernel: int = 4) -> int:
    p = lc_positions(spec.height, kernel, stride) * lc_positions(spec.width, kernel, stride)
    fan = kernel * kernel * spec.channels
    return p * channels * (fan + 1) + (p * channels + 1) * hidden + (hidden + 1) * num_classes


def search_lc(spec: ImageSpec, num_classes: int, target: int) -> Tuple[int, int, int, int]:
    """Protocol amendment A2: the dense hidden layer is never a bottleneck below 2 x num_classes."""
    best = None
    for stride in (1, 2, 3, 4):
        for channels in (1, 2, 3, 4):
            for hidden in range(2 * num_classes, 1025):
                p = lc_params(spec, stride, channels, hidden, num_classes)
                key = (abs(p - target), stride, channels, hidden)
                if best is None or key < best[0]:
                    best = (key, stride, channels, hidden, p)
    return best[1], best[2], best[3], best[4]


class LocallyConnectedNet(nn.Module):
    """Unshared 4x4 receptive fields on a regular grid -> LeakyReLU -> dense hidden -> LeakyReLU -> linear."""

    def __init__(self, spec: ImageSpec, num_classes: int, stride: int, channels: int, hidden: int, kernel: int = 4, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.spec, self.kernel, self.stride, self.channels = spec, kernel, stride, channels
        self.positions = lc_positions(spec.height, kernel, stride) * lc_positions(spec.width, kernel, stride)
        fan = kernel * kernel * spec.channels
        self.weight = nn.Parameter(torch.empty(self.positions, channels, fan))
        self.bias = nn.Parameter(torch.empty(self.positions, channels))
        bound = 1.0 / math.sqrt(fan)  # nn.Linear default rule, per local unit
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        self.hidden = nn.Linear(self.positions * channels, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec
        x = x.view(x.size(0), s.channels, s.height, s.width)
        patches = F.unfold(x, kernel_size=self.kernel, stride=self.stride)  # [B, fan, P]
        y = torch.einsum("bkp,pck->bpc", patches, self.weight) + self.bias
        y = F.leaky_relu(y, self.negative_slope).reshape(x.size(0), -1)
        y = F.leaky_relu(self.hidden(y), self.negative_slope)
        return self.out(y)


# ------------------------------------------------------------------ random-sparse MLP (F2)
def _balanced_degrees(total: int, slots: int, generator: torch.Generator) -> List[int]:
    """`slots` integer degrees summing to `total`, each floor or ceil of total/slots; the +1 slots are random."""
    base, extra = divmod(int(total), int(slots))
    deg = [base] * slots
    for i in torch.randperm(slots, generator=generator)[:extra].tolist():
        deg[i] += 1
    return deg


def fan_in_mask(fan_in: int, fan_out: int, nonzeros: int, generator: torch.Generator) -> torch.Tensor:
    """[fan_out, fan_in] mask: every output row gets floor/ceil(nonzeros/fan_out) distinct random inputs."""
    mask = torch.zeros(fan_out, fan_in)
    for row, d in enumerate(_balanced_degrees(nonzeros, fan_out, generator)):
        mask[row, torch.randperm(fan_in, generator=generator)[:d]] = 1.0
    return mask


def bipartite_mask(fan_in: int, fan_out: int, nonzeros: int, generator: torch.Generator, max_rounds: int = 10000) -> torch.Tensor:
    """[fan_out, fan_in] mask with balanced out-degree per INPUT unit and balanced in-degree per OUTPUT unit, no
    duplicate edges (configuration model with random swaps to remove duplicates)."""
    out_deg = _balanced_degrees(nonzeros, fan_in, generator)  # edges leaving each input unit
    in_deg = _balanced_degrees(nonzeros, fan_out, generator)  # edges entering each output unit
    src = [i for i, d in enumerate(out_deg) for _ in range(d)]
    dst = [j for j, d in enumerate(in_deg) for _ in range(d)]
    perm = torch.randperm(len(dst), generator=generator).tolist()
    dst = [dst[k] for k in perm]
    for _ in range(max_rounds):
        seen, dup = set(), []
        for k, edge in enumerate(zip(src, dst)):
            if edge in seen:
                dup.append(k)
            else:
                seen.add(edge)
        if not dup:
            break
        for k in dup:
            other = int(torch.randint(0, len(dst), (1,), generator=generator).item())
            dst[k], dst[other] = dst[other], dst[k]
    else:
        raise RuntimeError("bipartite_mask: duplicates not resolved")
    mask = torch.zeros(fan_out, fan_in)
    for i, j in zip(src, dst):
        mask[j, i] = 1.0
    assert int(mask.sum().item()) == int(nonzeros)
    return mask


class MaskedLinear(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, mask: torch.Tensor) -> None:
        super().__init__()
        nonzeros = int(mask.sum().item())
        self.register_buffer("mask", mask)
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias = nn.Parameter(torch.empty(fan_out))
        row_fan = mask.sum(dim=1).clamp(min=1.0)
        bound = (1.0 / row_fan.sqrt()).unsqueeze(1)  # nn.Linear rule with the effective fan-in of each row
        with torch.no_grad():
            self.weight.uniform_(-1.0, 1.0)
            self.weight.mul_(bound).mul_(mask)
            self.bias.uniform_(-1.0, 1.0)
            self.bias.mul_(bound.squeeze(1))
        self.nonzeros = int(nonzeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


def sparse_mlp_allocation(input_dim: int, num_classes: int, target: int, h1: int = 512, h2: int = 128, min_out_degree: int = 2) -> Tuple[int, int]:
    """Amendment A6: proportional split of the non-zero budget, with the second layer raised to at least
    `min_out_degree` edges per first-layer unit so that every unit reaches the output (no dead units, no tree)."""
    dense1, dense2 = input_dim * h1, h1 * h2
    budget = target - (h1 + h2) - (h2 + 1) * num_classes
    n2 = max(int(round(budget * dense2 / (dense1 + dense2))), min_out_degree * h1)
    return budget - n2, n2


class SparseMLP(nn.Module):
    """Two hidden layers (512, 128) with fixed, connected, degree-balanced random masks; dense classifier.

    Layer 1: every first-layer unit draws floor/ceil(n1/512) distinct random inputs. Layer 2: every first-layer unit
    sends floor/ceil(n2/512) >= 2 edges and every second-layer unit receives floor/ceil(n2/128) edges, without
    duplicates. Every hidden unit therefore lies on an input-to-output path (amendment A6, review PCR-004)."""

    def __init__(self, input_dim: int, num_classes: int, target: int, mask_seed: int, h1: int = 512, h2: int = 128, negative_slope: float = 0.1) -> None:
        super().__init__()
        n1, n2 = sparse_mlp_allocation(input_dim, num_classes, target, h1, h2)
        g = torch.Generator().manual_seed(int(mask_seed))
        m1 = fan_in_mask(input_dim, h1, n1, g)
        m2 = bipartite_mask(h1, h2, n2, g)
        assert int((m1.sum(dim=1) > 0).sum()) == h1 and int((m2.sum(dim=0) >= 2).sum()) == h1 and int((m2.sum(dim=1) > 0).sum()) == h2
        self.fc1 = MaskedLinear(input_dim, h1, m1)
        self.fc2 = MaskedLinear(h1, h2, m2)
        self.fc3 = nn.Linear(h2, num_classes)
        self.negative_slope = negative_slope
        self.effective_params = n1 + n2 + h1 + h2 + (h2 + 1) * num_classes
        self.allocation = {"layer1_nonzeros": n1, "layer2_nonzeros": n2,
                           "layer1_fan_in": [int(m1.sum(dim=1).min()), int(m1.sum(dim=1).max())],
                           "layer2_out_degree": [int(m2.sum(dim=0).min()), int(m2.sum(dim=0).max())],
                           "layer2_in_degree": [int(m2.sum(dim=1).min()), int(m2.sum(dim=1).max())]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x), self.negative_slope)
        x = F.leaky_relu(self.fc2(x), self.negative_slope)
        return self.fc3(x)


# ------------------------------------------------------------------ unequal-width MLP (F2, CIFAR-10)
class UnequalMLP(nn.Module):
    def __init__(self, input_dim: int, widths: List[int], num_classes: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        dims = [input_dim] + list(widths)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], num_classes)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = F.leaky_relu(layer(x), self.negative_slope)
        return self.out(x)


# ------------------------------------------------------------------ dendrite-slope variant (F7)
class SlopeDendriteLayer(FixedMaskedDendriteLayer):
    """r0 layer with the DENDRITE slope set separately; soma slope and init gain stay at negative_slope."""

    def __init__(self, *args, dendrite_slope: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dendrite_slope = float(dendrite_slope)

    def forward(self, x: torch.Tensor):
        gathered = x[:, self.dendrite_indices]
        dendritic_preact = (gathered * self.synaptic_weights.unsqueeze(0)).sum(dim=-1) + self.synaptic_bias
        dendritic_act = F.leaky_relu(dendritic_preact, negative_slope=self.dendrite_slope)
        dendritic_act = dendritic_act.view(x.size(0), self.soma_units, self.branches_per_soma)
        soma_preact = (dendritic_act * self.cable_weights.unsqueeze(0)).sum(dim=-1) + self.soma_bias
        soma_act = F.leaky_relu(soma_preact, negative_slope=self.negative_slope)
        return soma_act, dendritic_act


class SlopeDANN(nn.Module):
    def __init__(self, input_dim, num_classes, soma_units, branches, sample_size, idx, dendrite_slope: float) -> None:
        super().__init__()
        # construction order identical to DendriticANN: dendritic layer first, classifier second
        self.dendritic = SlopeDendriteLayer(input_dim=input_dim, soma_units=soma_units, branches_per_soma=branches,
                                            sample_size=sample_size, dendrite_indices=idx, negative_slope=0.1,
                                            dendrite_slope=dendrite_slope)
        self.classifier = nn.Linear(soma_units, num_classes)

    def forward(self, x):
        soma, _ = self.dendritic(x)
        return self.classifier(soma)


# ------------------------------------------------------------------ channel-aware routing (F6)
def channel_aware_lrf_indices(spec: ImageSpec, soma_units: int, branches: int, sample_size: int, seed: int,
                              patch_h: int = 4, patch_w: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    rows = []
    for _ in range(soma_units * branches):
        c = int(torch.randint(0, spec.channels, (1,), generator=g).item())
        top = int(torch.randint(0, spec.height - patch_h + 1, (1,), generator=g).item())
        left = int(torch.randint(0, spec.width - patch_w + 1, (1,), generator=g).item())
        cand = [c * spec.height * spec.width + h * spec.width + w
                for h in range(top, top + patch_h) for w in range(left, left + patch_w)]
        if len(cand) >= sample_size:
            perm = torch.randperm(len(cand), generator=g)[:sample_size].tolist()
        else:
            perm = torch.randint(0, len(cand), (sample_size,), generator=g).tolist()
        rows.append([cand[i] for i in perm])
    return torch.tensor(rows, dtype=torch.long)


# ------------------------------------------------------------------ conv-stem triad (F9)
class ConvStem(nn.Module):
    def __init__(self, spec: ImageSpec) -> None:
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(spec.channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.out_spec = ImageSpec(channels=16, height=spec.height // 4, width=spec.width // 4)

    def forward(self, x):
        s = self.spec
        x = x.view(x.size(0), s.channels, s.height, s.width)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x.reshape(x.size(0), -1)  # (C, H, W) order, as the LRF index map expects


def stem_lrf_indices(spec: ImageSpec, soma_units: int, branches: int, sample_size: int, seed: int,
                     patch_h: int = 4, patch_w: int = 4) -> torch.Tensor:
    """The r0 LRF sampler applied to the stem feature map (patches span all 16 stem channels)."""
    from src import sampling

    key = f"_stem_{spec.channels}x{spec.height}x{spec.width}"
    sampling.DATASET_SPECS.setdefault(key, spec)
    return build_dendrite_indices(key, soma_units, branches, sample_size, mode="lrf", seed=seed, patch_h=patch_h, patch_w=patch_w)


class StemModel(nn.Module):
    def __init__(self, stem: ConvStem, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.head = head

    def forward(self, x):
        return self.head(self.stem(x))


# ------------------------------------------------------------------ Transformer front end (MC-NEURO-R1-003, D1)
def tf_stem_params(spec: ImageSpec, dim: int = 16, mlp_hidden: int = 32, patch: int = 4) -> int:
    """Closed form: patch embedding, positional embedding, three LayerNorms, qkv and output projections, MLP."""
    tokens = (spec.height // patch) * (spec.width // patch)
    return ((patch * patch * spec.channels + 1) * dim + tokens * dim + 3 * 2 * dim + 4 * dim * (dim + 1)
            + (dim + 1) * mlp_hidden + (mlp_hidden + 1) * dim)


class TransformerStem(nn.Module):
    """Drop-in replacement of ConvStem under the F9 heads (amendment A14).

    Non-overlapping 4x4 patches embedded linearly to d = 16 (a convolution with kernel = stride = 4), a learned positional
    embedding over the (H/4) x (W/4) token grid, one pre-LayerNorm encoder block (self-attention with 2 heads of size 8 and
    a residual connection; MLP d -> 2d -> d with GELU and a residual connection) and a final LayerNorm. The token grid is
    returned as a (d, H/4, W/4) map flattened in (C, H, W) order, the shape and order of ConvStem, so the F9 heads and
    stem_lrf_indices are reused unchanged. Attention uses explicit matrix products and softmax (no fused attention kernel),
    so the runner's deterministic-algorithm setting covers it. Initialisation: the PyTorch default of every layer; the
    positional embedding, which has no PyTorch default, is drawn from N(0, 0.02^2) as in torchvision's VisionTransformer.
    """

    def __init__(self, spec: ImageSpec, dim: int = 16, heads: int = 2, mlp_hidden: int = 32, patch: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0 and spec.height % patch == 0 and spec.width % patch == 0
        self.spec, self.dim, self.heads, self.head_dim, self.patch = spec, dim, heads, dim // heads, patch
        self.grid_h, self.grid_w = spec.height // patch, spec.width // patch
        # construction order fixes the global-RNG draws, so one seed gives one stem under all three heads
        self.patch_embed = nn.Conv2d(spec.channels, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.empty(1, self.grid_h * self.grid_w, dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.out_spec = ImageSpec(channels=dim, height=self.grid_h, width=self.grid_w)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)  # each [b, heads, t, head_dim]
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(b, t, d))

    def forward(self, x):
        s = self.spec
        x = x.view(x.size(0), s.channels, s.height, s.width)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed  # [B, tokens, d], tokens in row-major grid order
        x = x + self.attention(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        x = self.norm_out(x)
        return x.transpose(1, 2).reshape(x.size(0), -1)  # (C, H, W) order with C = d, as the LRF index map expects


# ------------------------------------------------------------------ dispatch
def _copy_dann_init_into_nb(dann: DendriticANN, nb: NaiveBranchedLinear) -> None:
    with torch.no_grad():
        nb.synaptic_weights.copy_(dann.dendritic.synaptic_weights)
        nb.synaptic_bias.copy_(dann.dendritic.synaptic_bias)
        nb.cable_weights.copy_(dann.dendritic.cable_weights)
        nb.soma_bias.copy_(dann.dendritic.soma_bias)
        nb.classifier.weight.copy_(dann.classifier.weight)
        nb.classifier.bias.copy_(dann.classifier.bias)
        assert torch.equal(nb.dendrite_indices, dann.dendritic.dendrite_indices)


def build_r1_model(unit: Dict, info, routing_seed: int) -> nn.Module:
    """Build the R1 model named by ``unit['extra']['r1_model']`` (global RNG already seeded by the runner)."""
    extra = unit.get("extra") or {}
    kind = extra["r1_model"]
    spec = DATASET_SPECS[unit["dataset"]]
    S, B, K = int(unit["soma_units"]), int(unit["branches_per_soma"]), int(unit["sample_size"])
    C = int(info.num_classes)
    D = int(info.input_dim)
    target = target_budget(S, B, K, C)
    ph, pw = int(unit["patch_h"]), int(unit["patch_w"])

    if kind == "compact_cnn":
        c1, c2, _ = search_cnn(spec.channels, C, target)
        return CompactCNN(spec, C, c1, c2)
    if kind == "cnn_flat":
        c1, c2, _ = search_cnn_flat(spec, C, target)
        return CompactCNNFlat(spec, C, c1, c2)
    if kind == "lc_net":
        stride, ch, hidden, _ = search_lc(spec, C, target)
        return LocallyConnectedNet(spec, C, stride, ch, hidden)
    if kind == "sparse_mlp":
        return SparseMLP(D, C, target, mask_seed=routing_seed)
    if kind == "mlp_matched":
        return UnequalMLP(D, list(extra["widths"]), C)
    if kind == "dann_lrf_slope":
        idx = build_dendrite_indices(unit["dataset"], S, B, K, mode="lrf", seed=routing_seed, patch_h=ph, patch_w=pw)
        return SlopeDANN(D, C, S, B, K, idx, float(extra["dendrite_slope"]))
    if kind == "naive_branch_matched_init":
        idx = build_dendrite_indices(unit["dataset"], S, B, K, mode="lrf", seed=routing_seed, patch_h=ph, patch_w=pw)
        dann = DendriticANN(D, C, S, B, K, idx)  # consumes the global RNG exactly like the DANN-LRF unit
        nb = NaiveBranchedLinear(D, C, S, B, K, idx)
        _copy_dann_init_into_nb(dann, nb)
        return nb
    if kind == "dann_lrf_channel":
        idx = channel_aware_lrf_indices(spec, S, B, K, routing_seed, ph, pw)
        return DendriticANN(D, C, S, B, K, idx)
    if kind in ("stem_dann_lrf", "stem_naive_branch", "stem_mlp"):
        stem = ConvStem(spec)  # built first: identical stem initialisation across the three heads of a seed
        fspec = stem.out_spec
        fdim = fspec.channels * fspec.height * fspec.width
        if kind == "stem_mlp":
            width = estimate_param_matched_width(target, fdim, C)
            head = FlatMLP(fdim, C, width)
        else:
            idx = stem_lrf_indices(fspec, S, B, K, routing_seed, ph, pw)
            dann_head = DendriticANN(fdim, C, S, B, K, idx)  # same RNG consumption as the stem_dann_lrf unit
            if kind == "stem_dann_lrf":
                head = dann_head
            else:
                # amendment A7 (review PCR-002): the Naive-Branch head starts from the DANN head's initial tensors
                head = NaiveBranchedLinear(fdim, C, S, B, K, idx)
                _copy_dann_init_into_nb(dann_head, head)
        return StemModel(stem, head)
    if kind in ("tf_stem_dann_lrf", "tf_stem_naive_branch", "tf_stem_mlp"):
        # MC-NEURO-R1-003 D1 (amendment A14): the F9 branch above with the Transformer front end; heads, routing and the
        # parameter matching of the heads are unchanged
        stem = TransformerStem(spec)  # built first: identical stem initialisation across the three heads of a seed
        fspec = stem.out_spec
        fdim = fspec.channels * fspec.height * fspec.width
        if kind == "tf_stem_mlp":
            width = estimate_param_matched_width(target, fdim, C)
            head = FlatMLP(fdim, C, width)
        else:
            idx = stem_lrf_indices(fspec, S, B, K, routing_seed, ph, pw)
            dann_head = DendriticANN(fdim, C, S, B, K, idx)  # same RNG consumption as the tf_stem_dann_lrf unit
            if kind == "tf_stem_dann_lrf":
                head = dann_head
            else:
                # as A7: the Naive-Branch head starts from the DANN head's initial tensors
                head = NaiveBranchedLinear(fdim, C, S, B, K, idx)
                _copy_dann_init_into_nb(dann_head, head)
        return StemModel(stem, head)
    raise ValueError(f"unknown r1_model {kind}")


def describe_r1_model(model: nn.Module) -> Dict:
    """Parameter accounting written into result.json (dense count, effective count, head/stem split)."""
    out = {"dense_trainable_params": count_parameters(model)}
    if isinstance(model, SparseMLP):
        out["effective_trainable_params"] = model.effective_params
        out["sparse_allocation"] = model.allocation
    elif isinstance(model, StemModel):
        out["stem_params"] = count_parameters(model.stem)
        out["head_params"] = count_parameters(model.head)
        out["effective_trainable_params"] = out["dense_trainable_params"]
    else:
        out["effective_trainable_params"] = out["dense_trainable_params"]
    if isinstance(model, CompactCNN):
        out["cnn_channels"] = [model.conv1.out_channels, model.conv2.out_channels]
    if isinstance(model, LocallyConnectedNet):
        out["lc"] = {"stride": model.stride, "channels": model.channels, "hidden": model.hidden.out_features, "positions": model.positions}
    if isinstance(model, CompactCNNFlat):
        out["cnn_channels"] = [model.conv1.out_channels, model.conv2.out_channels]
        out["head_inputs"] = model.head_inputs
    if isinstance(model, StemModel) and isinstance(model.stem, TransformerStem):
        t = model.stem
        out["tf_stem"] = {"dim": t.dim, "heads": t.heads, "head_dim": t.head_dim, "mlp_hidden": t.fc1.out_features,
                          "patch": t.patch, "token_grid": [t.grid_h, t.grid_w]}
    return out
