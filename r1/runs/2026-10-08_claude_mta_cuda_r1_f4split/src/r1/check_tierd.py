"""Pre-run structural checks for MC-NEURO-R1-003 (protocol Section 10, amendments A14-A16); engineering only, no training
and no dataset file.

    python -B r1/check_tierd.py [--seeds 20] [--frozen-plan <plan_frozen_all.json>] [--data-root <dir holding cifar-100-python>]

* D1 Transformer front end, both image sizes: the stem output has ConvStem's shape and out_spec and is laid out in (C, H, W)
  order (embedding dimension as channel, token grid row-major); a 4x4 patch reaches only its own grid cell at the patch
  embedding; the stem count equals 3,312 / 4,064 and the closed form; every D1 model's stem, head and total count, with the
  head counts equal to the F9 heads;
* D1 heads for every seed, built with the runner's seeding sequence: the three heads see an identical stem, the Naive-Branch
  head starts from the DANN-LRF head's six initial tensors (copies, not shared storage), the routing indices are equal and
  equal to the F9 indices of the same seed;
* D2 CIFAR-100: the spec registration; the loader plumbing with a stubbed torchvision (no file, no download): the CIFAR-100
  call equals src/data.py's CIFAR-10 call except the class and download=False, and the cached tensors are the (C, H, W)
  flattening of the images; every model's count through the runner's own dispatch with C = 100 (DANN family 22,244;
  MLP-Param width 7, 22,367; VANN-Same 512 and 128 hidden units);
* every new model: one forward and backward pass on synthetic tensors under deterministic algorithms (finite loss, every
  parameter tensor has a finite, non-zero gradient); two identical seeded constructions give equal state, outputs and
  gradients; three Adam steps from two identical constructions end in equal parameters;
* plans tierd_arm (229 units: 9 anchors first, then seed-major D1 and D2), tierd_smoke (11), tierd_pilot (11) and tierd_run
  (449 units: tierd_arm unchanged and first, then the 100-epoch arm of amendment A17, DEC-NEURO-020, each of its units equal
  to its 30-epoch counterpart except id, condition, family and epochs) from the builders and from the command line; with
  --frozen-plan, every anchor equals its frozen unit except the family;
* with --data-root (on the run host after staging; this check never downloads): the real CIFAR-100 tensors through the r1
  loader (shapes, label range, class balance, value range, tensor SHA-256).
Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from r1 import r1_data, r1_plan  # noqa: E402  (importing r1_data registers the cifar100 spec, as in the runner)
from r1.r1_models import ConvStem, StemModel, TransformerStem, describe_r1_model, target_budget, tf_stem_params  # noqa: E402
from r1.r1_runner import build_unit_model, unit_spec  # noqa: E402
from src.models import count_parameters, estimate_param_matched_width  # noqa: E402
from src.sampling import DATASET_SPECS, ImageSpec  # noqa: E402
from src.train_eval import set_seed  # noqa: E402

# Values fixed by DEC-NEURO-019 and by amendment A3 (F9 head widths); asserted against the built models, never derived from them.
EXPECTED_STEM = {"fashionmnist": 3312, "cifar10": 4064}
EXPECTED_HEAD = {"tf_stem_dann_lrf": 10634, "tf_stem_naive_branch": 10634}
EXPECTED_MLP_HEAD = {"fashionmnist": (13, 10527), "cifar10": (10, 10470)}
EXPECTED_D2 = {"dann_lrf": 22244, "naive_branch": 22244, "dann_random": 22244, "mlp_param": 22367,
               "vann_same": (3072 + 1) * 512 + (512 + 1) * 128 + (128 + 1) * 100}
EXPECTED_MLP_PARAM_WIDTH = 7
D1_CLASSES, D2_CLASSES = 10, 100


class Info:
    def __init__(self, input_dim: int, num_classes: int) -> None:
        self.input_dim, self.num_classes = input_dim, num_classes


def dim_of(ds: str) -> int:
    s = DATASET_SPECS[ds]
    return s.channels * s.height * s.width


def first_unit(units, dataset: str, model: str):
    return next(u for u in units if u["dataset"] == dataset and u["model"] == model and u["family"] != "MC003_anchor")


def build(u, info, seed: int):
    set_seed(seed)  # the runner calls set_seed(init_seed) right before build_unit_model
    return build_unit_model(u, info, seed)


def states_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def head_tensors(stem_model):
    h = stem_model.head
    if hasattr(h, "dendritic"):
        d = h.dendritic
        return [d.synaptic_weights, d.synaptic_bias, d.cable_weights, d.soma_bias, h.classifier.weight, h.classifier.bias]
    return [h.synaptic_weights, h.synaptic_bias, h.cable_weights, h.soma_bias, h.classifier.weight, h.classifier.bias]


def head_indices(stem_model):
    h = stem_model.head
    return h.dendritic.dendrite_indices if hasattr(h, "dendritic") else h.dendrite_indices


# ---------------------------------------------------------------- D1: stem shape, order, locality, count
def check_stems(fails) -> None:
    for ds in r1_plan.D1_DATASETS:
        spec = DATASET_SPECS[ds]
        gh, gw = spec.height // 4, spec.width // 4
        set_seed(0)
        conv = ConvStem(spec)
        set_seed(0)
        tf = TransformerStem(spec)
        g = torch.Generator().manual_seed(11)
        x = torch.rand(3, dim_of(ds), generator=g)
        captured = {}
        hook = tf.norm_out.register_forward_hook(lambda _m, _i, o: captured.__setitem__("tokens", o.detach().clone()))
        with torch.no_grad():
            yc, yt = conv(x), tf(x)
        hook.remove()
        shape_ok = tuple(yt.shape) == tuple(yc.shape) == (3, 16 * gh * gw) and tf.out_spec == conv.out_spec
        order_ok = torch.equal(yt.view(3, 16, gh, gw).permute(0, 2, 3, 1).reshape(3, gh * gw, 16), captured["tokens"])
        maps = []
        hook = tf.patch_embed.register_forward_hook(lambda _m, _i, o: maps.append(o.detach().clone()))
        x2 = x.clone().view(3, spec.channels, spec.height, spec.width)
        x2[:, :, 8:12, 12:16] += 0.5  # the 4x4 patch at grid row 2, column 3
        with torch.no_grad():
            tf(x)
            tf(x2.reshape(3, -1))
        hook.remove()
        changed = (maps[0] - maps[1]).abs().sum(dim=(0, 1)).nonzero().tolist()
        local_ok = changed == [[2, 3]]
        n = count_parameters(tf)
        count_ok = n == EXPECTED_STEM[ds] == tf_stem_params(spec)
        parts = {"patch_embed": count_parameters(tf.patch_embed), "pos_embed": tf.pos_embed.numel(),
                 "layernorms": sum(count_parameters(m) for m in (tf.norm1, tf.norm2, tf.norm_out)),
                 "attention": count_parameters(tf.qkv) + count_parameters(tf.proj), "mlp": count_parameters(tf.fc1) + count_parameters(tf.fc2)}
        print(f"stem {ds:<12} {spec.channels}x{spec.height}x{spec.width}: transformer output {tuple(yt.shape)} = conv output "
              f"{tuple(yc.shape)} (16x{gh}x{gw}) shape_ok={shape_ok} chw_order_ok={order_ok} patch_locality_ok={local_ok}")
        print(f"stem {ds:<12} params {n} (expected {EXPECTED_STEM[ds]}, closed form {tf_stem_params(spec)}) {parts}; "
              f"ConvStem params {count_parameters(conv)}")
        if not (shape_ok and order_ok and local_ok and count_ok):
            fails.append(f"stem {ds}: shape={shape_ok} order={order_ok} locality={local_ok} count={count_ok}")


# ---------------------------------------------------------------- D1: model counts and head reuse
def check_d1_counts(fails, arm) -> None:
    budget = target_budget(128, 4, 16, D1_CLASSES)
    for ds in r1_plan.D1_DATASETS:
        info = Info(dim_of(ds), D1_CLASSES)
        for m in r1_plan.TF_HEADS:
            u = first_unit(arm, ds, m)
            model = build(u, info, 0)
            f9 = build(dict(u, extra={"r1_model": m[3:]}), info, 0)  # the F9 kind with the same head (stem_*)
            desc = describe_r1_model(model)
            stem_n, head_n, total = count_parameters(model.stem), count_parameters(model.head), count_parameters(model)
            if m == "tf_stem_mlp":
                width, exp_head = EXPECTED_MLP_HEAD[ds]
                fdim = 16 * model.stem.grid_h * model.stem.grid_w
                width_ok = model.head.fc1.out_features == width == estimate_param_matched_width(budget, fdim, D1_CLASSES)
            else:
                exp_head, width_ok = EXPECTED_HEAD[m], True
            ok = (isinstance(model, StemModel) and isinstance(model.stem, TransformerStem) and stem_n == EXPECTED_STEM[ds]
                  and head_n == exp_head == count_parameters(f9.head) and total == stem_n + head_n and width_ok
                  and desc["stem_params"] == stem_n and desc["head_params"] == head_n and desc["effective_trainable_params"] == total
                  and desc["tf_stem"]["token_grid"] == [model.stem.grid_h, model.stem.grid_w])
            extra = f" mlp_head_width={model.head.fc1.out_features}" if m == "tf_stem_mlp" else ""
            print(f"D1 {ds:<12} {m:<21} stem={stem_n} head={head_n} total={total}{extra} "
                  f"(F9 {m[3:]}: stem={count_parameters(f9.stem)} head={count_parameters(f9.head)} total={count_parameters(f9)}) ok={ok}")
            if not ok:
                fails.append(f"D1 counts {ds} {m}")


def check_d1_heads(fails, arm, seeds: int) -> None:
    for ds in r1_plan.D1_DATASETS:
        info = Info(dim_of(ds), D1_CLASSES)
        units = {m: first_unit(arm, ds, m) for m in r1_plan.TF_HEADS}
        bad = 0
        for s in range(seeds):
            a = build(dict(units["tf_stem_dann_lrf"], seed=s), info, s)
            b = build(dict(units["tf_stem_naive_branch"], seed=s), info, s)
            c = build(dict(units["tf_stem_mlp"], seed=s), info, s)
            f9 = build(dict(units["tf_stem_dann_lrf"], seed=s, extra={"r1_model": "stem_dann_lrf"}), info, s)
            stem_eq = states_equal(a.stem, b.stem) and states_equal(a.stem, c.stem)
            head_eq = all(torch.equal(x, y) for x, y in zip(head_tensors(a), head_tensors(b)))
            copies = all(x.data_ptr() != y.data_ptr() for x, y in zip(head_tensors(a), head_tensors(b)))
            idx_eq = torch.equal(head_indices(a), head_indices(b)) and torch.equal(head_indices(a), head_indices(f9))
            if not (stem_eq and head_eq and copies and idx_eq):
                bad += 1
                fails.append(f"D1 heads {ds} seed {s}: stem={stem_eq} head={head_eq} copies={copies} idx={idx_eq}")
        print(f"D1 {ds:<12} heads: {seeds} seeds checked (identical stem under the three heads, six Naive-Branch head tensors "
              f"copied from the DANN-LRF head, routing equal to F9's); failures {bad}")


# ---------------------------------------------------------------- D2: spec, loader plumbing, counts
def check_d2_spec(fails) -> None:
    expected = {"fashionmnist": ImageSpec(1, 28, 28), "kmnist": ImageSpec(1, 28, 28), "cifar10": ImageSpec(3, 32, 32),
                "cifar100": ImageSpec(3, 32, 32)}
    ok = all(DATASET_SPECS.get(k) == v for k, v in expected.items())
    print(f"D2 spec registration: cifar100 -> {DATASET_SPECS.get('cifar100')}; r0 specs unchanged; ok={ok}")
    if not ok:
        fails.append("D2 spec registration")


def check_loader_plumbing(fails) -> None:
    calls = []

    class StubToTensor:  # stands in for transforms.ToTensor; the stub datasets return tensors directly
        pass

    def stub_images(classes: int, train: bool):
        g = torch.Generator().manual_seed(7 if train else 8)
        n = 6 if train else 4
        return torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, classes, (n,), generator=g).tolist()

    def stub_dataset(cls_name: str, classes: int):
        class Stub:
            def __init__(self, root, train=True, download=False, transform=None) -> None:
                calls.append({"cls": cls_name, "root": root, "train": train, "download": download, "transform": type(transform).__name__})
                self.x, self.y = stub_images(classes, train)

            def __len__(self) -> int:
                return len(self.y)

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        return Stub

    tv, tv_ds, tv_tf = (types.ModuleType(n) for n in ("torchvision", "torchvision.datasets", "torchvision.transforms"))
    tv_ds.CIFAR10, tv_ds.CIFAR100, tv_tf.ToTensor = stub_dataset("CIFAR10", 10), stub_dataset("CIFAR100", 100), StubToTensor
    tv.datasets, tv.transforms = tv_ds, tv_tf
    keys = ("torchvision", "torchvision.datasets", "torchvision.transforms")
    saved_modules = {k: sys.modules.get(k) for k in keys}
    saved_cache = dict(r1_data._MEMORY_CACHE)
    sys.modules.update(dict(zip(keys, (tv, tv_ds, tv_tf))))
    try:
        r1_data._MEMORY_CACHE.clear()
        c10 = r1_data.load_cached("cifar10", "STUB_ROOT")
        c100 = r1_data.load_cached("cifar100", "STUB_ROOT")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pin_memory without an accelerator
            loaders, info = r1_data.make_cached_dataloaders(c100, "cifar100", 4, 0.1, 1.0, data_seed=0)
            bx, by = next(iter(loaders["test"]))
    finally:
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        r1_data._MEMORY_CACHE.clear()
        r1_data._MEMORY_CACHE.update(saved_cache)
    call10 = [c for c in calls if c["cls"] == "CIFAR10"]
    call100 = [c for c in calls if c["cls"] == "CIFAR100"]
    strip = lambda c: {k: v for k, v in c.items() if k not in ("cls", "download")}  # noqa: E731
    same_call = len(call10) == len(call100) == 2 and [strip(c) for c in call10] == [strip(c) for c in call100]
    downloads = [c["download"] for c in call10] == [True, True] and [c["download"] for c in call100] == [False, False]
    tx, ty = stub_images(100, True)
    ex, _ = stub_images(100, False)
    tensors_ok = (tuple(c100["train_x"].shape) == (6, 3072) and tuple(c100["test_x"].shape) == (4, 3072)
                  and c100["train_x"].dtype == torch.float32 and c100["train_y"].dtype == torch.int64
                  and torch.equal(c100["train_x"], tx.reshape(6, -1)) and torch.equal(c100["test_x"], ex.reshape(4, -1))
                  and c100["train_y"].tolist() == ty and c100["num_classes"] == 100 and c10["num_classes"] == 10)
    loader_ok = (info.name == "cifar100" and info.input_dim == 3072 and info.num_classes == 100 and tuple(bx.shape) == (4, 3072)
                 and by.dtype == torch.int64 and len(loaders["train"].dataset) == 5 and len(loaders["val"].dataset) == 1)
    print(f"D2 loader plumbing (stubbed torchvision, no file): CIFAR-100 calls equal the src CIFAR-10 calls except class and "
          f"download flag: {same_call} (calls {call100}); download flags src/r1 {[c['download'] for c in call10]}/"
          f"{[c['download'] for c in call100]}: {downloads}; (C,H,W) flattening and 100 classes: {tensors_ok}; loaders: {loader_ok}")
    if not (same_call and downloads and tensors_ok and loader_ok):
        fails.append(f"D2 loader plumbing: call={same_call} download={downloads} tensors={tensors_ok} loaders={loader_ok}")


def check_d2_counts(fails, arm) -> None:
    budget = target_budget(128, 4, 16, D2_CLASSES)
    info = Info(3072, D2_CLASSES)
    print(f"D2 DANN budget at C = 100: {budget} (expected 22244)")
    if budget != 22244:
        fails.append("D2 budget")
    for m in r1_plan.D2_MODELS:
        model = build(first_unit(arm, "cifar100", m), info, 0)
        n = count_parameters(model)
        ok = n == EXPECTED_D2[m]
        detail = ""
        if m == "mlp_param":
            w = model.fc1.out_features
            ok = ok and w == EXPECTED_MLP_PARAM_WIDTH == estimate_param_matched_width(budget, 3072, D2_CLASSES)
            detail = f" width={w} gap={n - budget:+d} ({100.0 * (n - budget) / budget:+.2f} %)"
        if m == "vann_same":
            ok = ok and model.fc1.out_features == 512 and model.fc2.out_features == 128
            detail = f" hidden=({model.fc1.out_features}, {model.fc2.out_features})"
        with torch.no_grad():
            out = model(torch.zeros(2, 3072))
        ok = ok and tuple(out.shape) == (2, D2_CLASSES)
        print(f"D2 cifar100     {m:<13} params={n} (expected {EXPECTED_D2[m]}){detail} logits={tuple(out.shape)} ok={ok}")
        if not ok:
            fails.append(f"D2 counts {m}")


# ---------------------------------------------------------------- every new model: forward/backward and determinism
def fwd_bwd(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    return logits.detach().clone(), float(loss.detach()), {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in model.named_parameters()}


def check_passes_and_determinism(fails, arm) -> None:
    cases = [(ds, m, D1_CLASSES) for ds in r1_plan.D1_DATASETS for m in r1_plan.TF_HEADS] + [("cifar100", m, D2_CLASSES) for m in r1_plan.D2_MODELS]
    for ds, m, classes in cases:
        u, info = first_unit(arm, ds, m), Info(dim_of(ds), classes)
        g = torch.Generator().manual_seed(123)
        x = torch.rand(16, info.input_dim, generator=g)
        y = torch.randint(0, classes, (16,), generator=g)
        m1, m2 = build(u, info, 0), build(u, info, 0)
        state_eq = states_equal(m1, m2)
        l1, loss1, g1 = fwd_bwd(m1, x, y)
        l2, loss2, g2 = fwd_bwd(m2, x, y)
        good = sum(1 for v in g1.values() if v is not None and bool(torch.isfinite(v).all()) and float(v.abs().sum()) > 0)
        pass_ok = tuple(l1.shape) == (16, classes) and loss1 == loss1 and abs(loss1) < float("inf") and good == len(g1)
        same_pass = torch.equal(l1, l2) and loss1 == loss2 and all(torch.equal(g1[k], g2[k]) for k in g1)
        trained = []
        for _ in range(2):
            mm = build(u, info, 0)
            opt = torch.optim.Adam(mm.parameters(), lr=1e-3, betas=(0.9, 0.999))
            for _step in range(3):
                opt.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(mm(x), y).backward()
                opt.step()
            trained.append(mm)
        same_steps = states_equal(trained[0], trained[1]) and not states_equal(trained[0], m1)
        print(f"pass {ds:<12} {m:<21} logits={tuple(l1.shape)} loss={loss1:.6f} grads finite and non-zero in "
              f"{good}/{len(g1)} tensors; construction equal={state_eq} "
              f"forward/backward equal={same_pass} three Adam steps equal={same_steps}")
        if not (state_eq and pass_ok and same_pass and same_steps):
            fails.append(f"pass/determinism {ds} {m}: state={state_eq} pass={pass_ok} same={same_pass} steps={same_steps}")


# ---------------------------------------------------------------- plans
BASE_KEYS = {"epochs": 30, "batch_size": 256, "lr": 1e-3, "val_fraction": 0.1, "soma_units": 128, "branches_per_soma": 4,
             "sample_size": 16, "patch_h": 4, "patch_w": 4, "subset_fraction": 1.0}


def check_plans(fails, frozen_plan) -> None:
    arm, smoke, pilot = r1_plan.tierd_arm(), r1_plan.tierd_smoke(), r1_plan.tierd_pilot()
    ids = [u["unit_id"] for u in arm]
    fams = [u["family"] for u in arm]
    n_anchor = fams.count("MC003_anchor")
    body = arm[n_anchor:]
    seeds = [u["seed"] for u in body]
    per_seed = {s: sorted((u["dataset"], u["model"]) for u in body if u["seed"] == s) for s in range(20)}
    want = sorted([(ds, m) for ds in r1_plan.D1_DATASETS for m in r1_plan.TF_HEADS] + [("cifar100", m) for m in r1_plan.D2_MODELS])
    d1 = [u for u in body if u["family"] == "MC003_D1_tfstem"]
    d2 = [u for u in body if u["family"] == "MC003_D2_cifar100"]
    ok = (len(arm) == 229 and len(set(ids)) == len(ids) and n_anchor == 9 and fams[:9] == ["MC003_anchor"] * 9
          and len(d1) == 120 and len(d2) == 100 and len(d1) + len(d2) == len(body) and seeds == sorted(seeds)
          and all(per_seed[s] == want for s in range(20))
          and all(u["extra"] == {"r1_model": u["model"]} and u["condition"] == f"{r1_plan.SHORT[u['dataset']]}_full" for u in d1)
          and all(u["extra"] == {} and u["dataset"] == "cifar100" and u["condition"] == "cifar100_full" for u in d2)
          and all(all(u[k] == v for k, v in BASE_KEYS.items()) for u in arm) and all(u["seed"] == 0 for u in arm[:9]))
    print(f"plan tierd_arm: {len(arm)} units (anchors {n_anchor} first, D1 {len(d1)}, D2 {len(d2)}), unique={len(set(ids)) == len(ids)}, "
          f"seed-major with 11 units per seed, shared hyper-parameters; ok={ok}")
    if not ok:
        fails.append("plan tierd_arm")
    s_ok = (len(smoke) == 11 and len({u["unit_id"] for u in smoke}) == 11 and all(u["epochs"] == 2 and u["seed"] == 0 for u in smoke)
            and sorted((u["dataset"], u["model"]) for u in smoke) == want)
    p_ok = (len(pilot) == 11 and len({u["unit_id"] for u in pilot}) == 11 and all(u["epochs"] == 30 and u["seed"] == r1_plan.TIERD_PILOT_SEED for u in pilot)
            and r1_plan.TIERD_PILOT_SEED not in range(20) and sorted((u["dataset"], u["model"]) for u in pilot) == want)
    print(f"plan tierd_smoke: {len(smoke)} units, 2 epochs, every new kind and dataset; ok={s_ok}")
    print(f"plan tierd_pilot: {len(pilot)} units, seed {r1_plan.TIERD_PILOT_SEED} (outside 0-19), 30 epochs; ok={p_ok}")
    if not (s_ok and p_ok):
        fails.append(f"plan smoke={s_ok} pilot={p_ok}")
    # A17 (DEC-NEURO-020): tierd_run = tierd_arm unchanged and first, then the 100-epoch arm of D1 and D2, seed-major
    run = r1_plan.tierd_run()
    e100 = run[len(arm):]
    strip = ("unit_id", "condition", "family", "epochs")
    counterpart = {(u["dataset"], u["model"], u["seed"]): u for u in body}
    r_seeds = [u["seed"] for u in e100]
    r_per_seed = {s: sorted((u["dataset"], u["model"]) for u in e100 if u["seed"] == s) for s in range(20)}
    pair_ok = all(
        {k: v for k, v in u.items() if k not in strip} == {k: v for k, v in counterpart[(u["dataset"], u["model"], u["seed"])].items() if k not in strip}
        and u["condition"] == counterpart[(u["dataset"], u["model"], u["seed"])]["condition"] + "_e100"
        and u["unit_id"] == f"{u['condition']}__{u['model']}__s{u['seed']:02d}"
        and u["family"] == counterpart[(u["dataset"], u["model"], u["seed"])]["family"] + "_e100"
        for u in e100)
    r_ok = (len(run) == 449 and run[:len(arm)] == arm and len(e100) == 220 and len({u["unit_id"] for u in run}) == 449
            and all(u["epochs"] == r1_plan.TIERD_BUDGET_EPOCHS == 100 for u in e100) and r_seeds == sorted(r_seeds)
            and all(r_per_seed[s] == want for s in range(20)) and pair_ok)
    print(f"plan tierd_run: {len(run)} units = tierd_arm ({len(arm)}) unchanged and first + 100-epoch arm ({len(e100)}: D1 "
          f"{sum(1 for u in e100 if u['family'] == 'MC003_D1_tfstem_e100')}, D2 {sum(1 for u in e100 if u['family'] == 'MC003_D2_cifar100_e100')}), "
          f"seed-major, each 100-epoch unit equal to its 30-epoch counterpart except id, condition, family and epochs; ok={r_ok}")
    if not r_ok:
        fails.append("plan tierd_run")
    with tempfile.TemporaryDirectory() as tmp:
        cli_ok = True
        for fam, units in (("tierd_arm", arm), ("tierd_smoke", smoke), ("tierd_pilot", pilot), ("tierd_run", run)):
            out = Path(tmp) / f"{fam}.json"
            rc = subprocess.run([sys.executable, "-B", str(ROOT / "r1" / "r1_plan.py"), fam, "--run-id", "check", "--out", str(out)],
                                cwd=str(ROOT), capture_output=True, text=True).returncode
            cli_ok = cli_ok and rc == 0 and json.loads(out.read_text(encoding="utf-8"))["units"] == units
    print(f"plan command line (r1_plan.py tierd_arm|tierd_smoke|tierd_pilot|tierd_run) equals the builders: {cli_ok}")
    if not cli_ok:
        fails.append("plan command line")
    if frozen_plan:
        frozen = {u["unit_id"]: u for u in json.loads(Path(frozen_plan).read_text(encoding="utf-8"))["units"]}
        rows = []
        for a in arm[:9]:
            f = frozen.get(a["unit_id"])
            same = f is not None and {k: v for k, v in unit_spec(a).items() if k != "family"} == {k: v for k, v in unit_spec(f).items() if k != "family"}
            fam_ok = f is not None and f["family"] == ("F9_convstem" if a["model"].startswith("stem_") else "main_grid")
            rows.append(same and fam_ok)
            print(f"anchor {a['unit_id']:<32} frozen unit {'found' if f else 'MISSING'} (family {f['family'] if f else '-'}); "
                  f"spec equal except family={same}")
        if not all(rows):
            fails.append("anchors against the frozen plan")


# ---------------------------------------------------------------- real CIFAR-100 on the run host (optional)
def check_real_cifar100(fails, data_root: str) -> None:
    if not (Path(data_root) / "cifar-100-python").is_dir():
        print(f"real CIFAR-100: {Path(data_root) / 'cifar-100-python'} is not staged; this check never downloads")
        fails.append("real CIFAR-100 not staged")
        return
    c = r1_data.load_cached("cifar100", data_root)
    ty, ey = c["train_y"], c["test_y"]
    ok = (tuple(c["train_x"].shape) == (50000, 3072) and tuple(c["test_x"].shape) == (10000, 3072) and c["num_classes"] == 100
          and c["train_x"].dtype == torch.float32 and ty.dtype == torch.int64
          and torch.bincount(ty, minlength=100).tolist() == [500] * 100 and torch.bincount(ey, minlength=100).tolist() == [100] * 100
          and float(c["train_x"].min()) >= 0.0 and float(c["train_x"].max()) <= 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaders, info = r1_data.make_cached_dataloaders(c, "cifar100", 256, 0.1, 1.0, data_seed=0)
    ok = ok and info.num_classes == 100 and len(loaders["train"].dataset) == 45000 and len(loaders["val"].dataset) == 5000
    print(f"real CIFAR-100 at {data_root}: train {tuple(c['train_x'].shape)} test {tuple(c['test_x'].shape)} classes {c['num_classes']} "
          f"fit/val {len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}; tensor sha256 {c['hashes']}; ok={ok}")
    if not ok:
        fails.append("real CIFAR-100")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--frozen-plan", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)  # as the runner's --deterministic
    print(f"torch {torch.__version__}; deterministic algorithms {torch.are_deterministic_algorithms_enabled()}")
    fails = []
    arm = r1_plan.tierd_arm()
    check_stems(fails)
    check_d1_counts(fails, arm)
    check_d1_heads(fails, arm, args.seeds)
    check_d2_spec(fails)
    check_loader_plumbing(fails)
    check_d2_counts(fails, arm)
    check_passes_and_determinism(fails, arm)
    check_plans(fails, args.frozen_plan)
    if args.data_root:
        check_real_cifar100(fails, args.data_root)
    print("FAILED:", fails[:20] if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
