"""Independent second check of the F1 convergence-arm rows and prose numbers of NEURO/manuscript-r1 (MC-NEURO-R1-004).

Date/time: 2026-10-05 21:24 +03:00
Tool: Cowork-Claude
Model, if known: claude-opus-5-5 (max)
Operation ID: neucom-r1-f1conv-delivery-analysis-20261005

Written separately from build_f1conv_assets.py and without pandas or numpy. Checked files (figures/r1/out/):
table5_f1conv_block.tex, table3_f1conv_row.tex, f1conv_text_numbers.json and PROVENANCE_f1conv.json.

- Every unit is recomputed from the raw per-epoch history of the extracted run (each history checked against the SHA-256 in its
  result.json first): the first epoch of minimum validation loss over the 100 epochs and, separately, over epochs 1-30 (the
  prefix gate makes epochs 1-30 the frozen 30-epoch history), the test accuracy there as an integer count out of 10,000, the
  training accuracy at the last epoch, the selected epoch, whether it is the last epoch, and the parameter count.
- Every contrast's mean difference, d_z and matched-pairs rank-biserial (average ranks over |d|, zeros dropped) is recomputed from
  the seed-paired integer differences; its interval, Holm-adjusted p and label are compared with the run's independent
  recomputation (verification/independent_recompute.json, which shares no code with the analysis), the label is checked against
  the A8 rule applied to those values, and the 30-epoch label against the recomputation's frozen30_label.
- Tolerance: half a unit of the last printed digit (exact equality for counts and labels).
- `selftest` copies the checked files, requires the unmodified copy to pass (positive control) and requires each of four planted
  errors to fail with a message that names the planted item (negative controls).

Usage: python verify_f1conv_assets.py check <extracted run folder> [<out dir>]
       python verify_f1conv_assets.py selftest <extracted run folder>
Exit: 0 pass, 1 a disagreement, 2 a usage or input fault.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

P = Path(__file__).resolve().parents[2]
RECOMPUTE = P / "runs" / "2026-10-05_claude_mta_cuda_r1_f1conv" / "verification" / "independent_recompute.json"  # public copy
OUT_DEFAULT = P / "figures" / "r1" / "out"
FAMILY = "MC004_F1_e100"
N_TEST = 10000
N_UNITS = 240
CHECKED_FILES = ("table5_f1conv_block.tex", "table3_f1conv_row.tex", "f1conv_text_numbers.json", "PROVENANCE_f1conv.json")

# independently written label maps (not imported from the builder)
CONTRAST_TO_B = {"\\DANNLRF{} $-$ \\NAIVEBRANCH{}": "naive_branch", "\\DANNLRF{} $-$ \\MLPPARAM{}": "mlp_param",
                 "\\DANNLRF{} $-$ \\DANNRANDOM{}": "dann_random"}
COND_LABEL_TO_DS = {"FashionMNIST full": "fashion", "KMNIST full": "kmnist", "CIFAR-10 full": "cifar"}
BLOCK = "Budget arm A18: full-data contrasts at 100 epochs"
LABEL_WORD = {"supported": "supported", "not supported": "not_supported", "reverse": "reverse"}
MODELS = ("dann_lrf", "naive_branch", "mlp_param", "dann_random")


class Fault(Exception):
    """An input or usage fault (exit 2), never a disagreement."""


def first_min(recs: list[dict], upto: int) -> int:
    best_epoch, best_loss = None, None
    for r in recs[:upto]:
        v = float(r["val_loss"])
        if best_loss is None or v < best_loss:
            best_loss, best_epoch = v, int(r["epoch"])
    return best_epoch


def count_at(recs: list[dict], epoch: int, name: str) -> int:
    x = float(recs[epoch - 1]["test_acc"]) * N_TEST
    c = round(x)
    if abs(x - c) > 1e-6:
        raise Fault(f"test accuracy not on the integer scale: {name} epoch {epoch}")
    return c


def load_units(run: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """(dataset key, model) -> seed -> recomputed unit record."""
    units_dir = run / "units"
    if not units_dir.is_dir():
        raise Fault(f"no units/ folder under {run}")
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    seen = 0
    for udir in sorted(p for p in units_dir.iterdir() if p.is_dir()):
        res = json.loads((udir / "result.json").read_bytes().decode("utf-8"))
        spec = res["unit_spec"]
        seen += 1
        if spec["family"] != FAMILY or int(spec["epochs"]) != 100:
            raise Fault(f"unexpected unit {udir.name}: {spec['family']} {spec['epochs']}")
        raw = (udir / res["history_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != res["history_sha256"].upper():
            raise Fault(f"history hash differs from result.json: {udir.name}")
        recs = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        recs.sort(key=lambda r: int(r["epoch"]))
        if [int(r["epoch"]) for r in recs] != list(range(1, 101)):
            raise Fault(f"epochs incomplete: {udir.name}")
        e100 = first_min(recs, 100)
        e30 = first_min(recs, 30)
        ds = spec["condition"].split("_")[0]
        key = (ds, spec["model"])
        seed = int(spec["seed"])
        if seed in cells.setdefault(key, {}):
            raise Fault(f"duplicate unit {key} seed {seed}")
        cells[key][seed] = {"c100": count_at(recs, e100, udir.name), "e100": e100,
                            "c30": count_at(recs, e30, udir.name), "e30": e30,
                            "train100": float(recs[99]["train_acc"]),
                            "params": int(res["summary"]["trainable_params"])}
    if seen != N_UNITS:
        raise Fault(f"expected {N_UNITS} unit folders, found {seen}")
    return cells


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def rank_biserial(d: list[int]) -> float:
    nz = [x for x in d if x != 0]
    if not nz:
        return 0.0
    mags = sorted(set(abs(x) for x in nz))
    rank_of: dict[int, float] = {}
    pos = 0
    for m in mags:
        k = sum(1 for x in nz if abs(x) == m)
        rank_of[m] = pos + (k + 1) / 2.0      # average of ranks pos+1 .. pos+k
        pos += k
    wp = sum(rank_of[abs(x)] for x in nz if x > 0)
    wn = sum(rank_of[abs(x)] for x in nz if x < 0)
    return (wp - wn) / (wp + wn)


def num(token: str) -> float:
    return float(token.strip().replace("$-$", "-").replace(",", ""))


def split_row(line: str) -> list[str]:
    body = line.rstrip()
    if not body.endswith("\\\\"):
        raise Fault(f"row without a LaTeX line end: {line[:60]}")
    return [c.strip() for c in body[:-2].split(" & ")]


def check(run: Path, out: Path) -> tuple[int, list[str]]:
    fails: list[str] = []
    n = {"values": 0}

    def close(shown: float, value: float, decimals: int, what: str) -> None:
        n["values"] += 1
        if abs(shown - value) > 0.5 * 10 ** (-decimals) + 1e-9:
            fails.append(f"{what}: shown {shown} vs recomputed {value}")

    def equal(shown, value, what: str) -> None:
        n["values"] += 1
        if shown != value:
            fails.append(f"{what}: shown {shown!r} vs recomputed {value!r}")

    cells = load_units(run)
    rec = json.loads(RECOMPUTE.read_bytes().decode("utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("disagreements"):
        raise Fault("the run's independent recomputation is not a clean PASS")
    rows = rec["rows"]

    def seeds_of(ds: str, model: str) -> dict[int, dict]:
        s = cells.get((ds, model), {})
        if sorted(s) != list(range(20)):
            raise Fault(f"cell {ds}|{model} does not hold seeds 0-19")
        return s

    def cell_stats(ds: str, model: str) -> dict:
        s = seeds_of(ds, model)
        acc = [s[i]["c100"] / 100.0 for i in range(20)]
        acc30 = [s[i]["c30"] / 100.0 for i in range(20)]
        params = {s[i]["params"] for i in range(20)}
        if len(params) != 1:
            raise Fault(f"parameter count differs within {ds}|{model}")
        return {"acc_mean_pp": mean(acc), "acc_sd_pp": sd(acc), "acc_min_pp": min(acc), "acc_max_pp": max(acc),
                "frozen30_acc_mean_pp": mean(acc30), "gain": mean(acc) - mean(acc30),
                "train_pct": 100.0 * mean([s[i]["train100"] for i in range(20)]),
                "sel_mean": mean([float(s[i]["e100"]) for i in range(20)]),
                "sel_last": sum(1 for i in range(20) if s[i]["e100"] == 100),
                "within30": sum(1 for i in range(20) if s[i]["e100"] <= 30),
                "sel30_mean": mean([float(s[i]["e30"]) for i in range(20)]),
                "sel30_last": sum(1 for i in range(20) if s[i]["e30"] == 30),
                "params": params.pop()}

    def diffs(ds: str, b: str, which: str) -> list[int]:
        sa, sb = seeds_of(ds, "dann_lrf"), seeds_of(ds, b)
        return [sa[i][which] - sb[i][which] for i in range(20)]

    # ---------------------------------------------------------------- Table 5 block
    seen_ids = set()
    lines = [x for x in (out / "table5_f1conv_block.tex").read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or lines[0].strip() != "\\midrule":
        fails.append("Table 5 block does not open with \\midrule")
    head = [x for x in lines if x.startswith("\\multicolumn")]
    if len(head) != 1 or BLOCK not in head[0]:
        fails.append(f"Table 5 block header differs: {head}")
    for line in lines:
        if line.startswith("\\midrule") or line.startswith("\\multicolumn"):
            continue
        c = split_row(line)
        if len(c) != 8:
            raise Fault(f"contrast row with {len(c)} cells: {line[:60]}")
        b = CONTRAST_TO_B[c[0]]
        ds = COND_LABEL_TO_DS[c[1]]
        cid = f"F1e100:{ds}_full_e100:dann_lrf-{b}"
        if cid in seen_ids:
            fails.append(f"{cid}: row repeated")
        seen_ids.add(cid)
        d = diffs(ds, b, "c100")
        mdiff = mean([float(x) for x in d]) / 100.0
        dz = mean([float(x) for x in d]) / sd([float(x) for x in d])
        close(num(c[2]), mdiff, 2, f"{cid} mean difference")
        close(num(c[5]), dz, 2, f"{cid} d_z")
        close(num(c[6]), rank_biserial(d), 2, f"{cid} rank-biserial")
        r = rows.get(cid)
        if r is None:
            fails.append(f"{cid}: not in the independent recomputation")
            continue
        close(r["mean_pp"], mdiff, 9, f"{cid} independent mean vs raw")      # three-way agreement
        close(r["d_z"], dz, 9, f"{cid} independent d_z vs raw")
        lo_t, hi_t = [num(x) for x in c[3].strip("[]").split(",")]
        close(lo_t, r["ci95_pp"][0], 2, f"{cid} interval low")
        close(hi_t, r["ci95_pp"][1], 2, f"{cid} interval high")
        if c[4] == "$<$0.001":
            n["values"] += 1
            if not r["p_holm"] < 0.001:
                fails.append(f"{cid}: Holm p shown <0.001 but is {r['p_holm']}")
        else:
            close(float(c[4]), r["p_holm"], 3, f"{cid} Holm p")
        shown = LABEL_WORD[c[7]]
        equal(shown, r["label"], f"{cid} label vs independent")
        rule = ("supported" if r["p_holm"] < 0.05 and r["ci95_pp"][0] > 0 else
                "reverse" if r["p_holm"] < 0.05 and r["ci95_pp"][1] < 0 else "not_supported")
        equal(shown, rule, f"{cid} label vs the A8 rule")
    if seen_ids != set(rows):
        fails.append(f"Table 5 block ids differ from the independent recomputation: {sorted(set(rows) ^ seen_ids)}")

    # ---------------------------------------------------------------- Table 3 row
    t3 = [x for x in (out / "table3_f1conv_row.tex").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(t3) != 1:
        fails.append(f"Table 3 file holds {len(t3)} rows, expected 1")
    else:
        c = split_row(t3[0])
        equal(c[0], "A18", "Table 3 row id")
        equal(int(num(c[-1])), sum(len(v) for v in cells.values()), "Table 3 row A18 runs")

    # ---------------------------------------------------------------- prose numbers
    tn = json.loads((out / "f1conv_text_numbers.json").read_bytes().decode("utf-8"))
    if len(tn["cells"]) != 12:
        fails.append(f"text numbers hold {len(tn['cells'])} cells, expected 12")
    for key, v in tn["cells"].items():
        ds, model = key.split("|")
        st = cell_stats(ds, model)
        for field, value in (("acc_mean_pp", st["acc_mean_pp"]), ("acc_sd_pp", st["acc_sd_pp"]),
                             ("acc_min_pp", st["acc_min_pp"]), ("acc_max_pp", st["acc_max_pp"]),
                             ("frozen30_acc_mean_pp", st["frozen30_acc_mean_pp"]), ("acc_gain_over_frozen30_pp", st["gain"]),
                             ("train_acc_last_epoch_mean_pct", st["train_pct"]), ("selected_epoch_mean", st["sel_mean"]),
                             ("frozen30_selected_epoch_mean", st["sel30_mean"])):
            close(float(v[field]), value, 6, f"text numbers {key} {field}")
        equal(int(v["selected_last_epoch_count"]), st["sel_last"], f"text numbers {key} selected_last_epoch_count")
        equal(int(v["selected_within_first30_count"]), st["within30"], f"text numbers {key} selected_within_first30_count")
        equal(int(v["frozen30_selected_last_epoch_count"]), st["sel30_last"],
              f"text numbers {key} frozen30_selected_last_epoch_count")
        equal(int(v["effective_params"]), st["params"], f"text numbers {key} effective_params")
    for ds, v in tn["train_gap_dann_minus_naive_pct"].items():
        close(float(v), cell_stats(ds, "dann_lrf")["train_pct"] - cell_stats(ds, "naive_branch")["train_pct"], 6,
              f"text numbers training-accuracy gap {ds}")
    for ds, v in tn["selected_last_epoch_ranges"].items():
        l100 = [cell_stats(ds, m)["sel_last"] for m in MODELS]
        l30 = [cell_stats(ds, m)["sel30_last"] for m in MODELS]
        equal(list(v["e100"]), [min(l100), max(l100)], f"text numbers last-epoch range {ds} at 100 epochs")
        equal(list(v["e30"]), [min(l30), max(l30)], f"text numbers last-epoch range {ds} at 30 epochs")
    tot = tn["selected_last_epoch_totals"]
    equal(int(tot["e100"]), sum(1 for s in cells.values() for u in s.values() if u["e100"] == 100),
          "text numbers last-epoch total at 100 epochs")
    equal(int(tot["e30"]), sum(1 for s in cells.values() for u in s.values() if u["e30"] == 30),
          "text numbers last-epoch total at 30 epochs")
    equal(int(tot["units"]), sum(len(s) for s in cells.values()), "text numbers unit total")
    for cid, v in tn["contrasts"].items():
        _, cond_s, pair = cid.split(":")
        ds = cond_s.split("_")[0]
        b = pair.split("-")[1]
        d = diffs(ds, b, "c100")
        d30 = diffs(ds, b, "c30")
        close(float(v["mean_diff_pp"]), mean([float(x) for x in d]) / 100.0, 6, f"text numbers {cid} mean")
        close(float(v["d_z"]), mean([float(x) for x in d]) / sd([float(x) for x in d]), 6, f"text numbers {cid} d_z")
        close(float(v["rank_biserial"]), rank_biserial(d), 6, f"text numbers {cid} rank-biserial")
        equal(int(v["positive_seeds"]), sum(1 for x in d if x > 0), f"text numbers {cid} positive seeds")
        equal(int(v["negative_seeds"]), sum(1 for x in d if x < 0), f"text numbers {cid} negative seeds")
        close(float(v["frozen30_mean_diff_pp"]), mean([float(x) for x in d30]) / 100.0, 6, f"text numbers {cid} 30-epoch mean")
        ch = [a - b2 for a, b2 in zip(d, d30)]
        close(float(v["budget_change_mean_pp"]), mean([float(x) for x in ch]) / 100.0, 6, f"text numbers {cid} budget change")
        equal(int(v["seeds_change_positive"]), sum(1 for x in ch if x > 0), f"text numbers {cid} seeds change positive")
        equal(int(v["seeds_change_negative"]), sum(1 for x in ch if x < 0), f"text numbers {cid} seeds change negative")
        r = rows[cid]
        close(float(v["budget_change_mean_pp"]), r["budget_change_pp"], 6, f"text numbers {cid} budget change vs independent")
        close(float(v["holm_p"]), r["p_holm"], 9, f"text numbers {cid} Holm p")
        close(float(v["ci95_pp"][0]), r["ci95_pp"][0], 6, f"text numbers {cid} interval low")
        close(float(v["ci95_pp"][1]), r["ci95_pp"][1], 6, f"text numbers {cid} interval high")
        equal(str(v["outcome"]), r["label"], f"text numbers {cid} label")
        equal(str(v["frozen30_outcome"]), r["frozen30_label"], f"text numbers {cid} 30-epoch label")
        want_agree = "consistent" if r["label"] == r["frozen30_label"] else "budget_dependent"
        equal(str(v["agreement"]), want_agree, f"text numbers {cid} budget agreement")
    equal(sorted(tn["budget_differing"]), sorted(rec["differing_between_budgets"]), "text numbers budget-differing ids")
    for rule_id, outcome in tn["composite"].items():
        if rule_id in rec["composites"]:
            equal(outcome, rec["composites"][rule_id], f"text numbers composite {rule_id}")

    # ---------------------------------------------------------------- provenance of the builder run
    prov = json.loads((out / "PROVENANCE_f1conv.json").read_bytes().decode("utf-8"))
    for item in prov["inputs"]:
        n["values"] += 1
        if hashlib.sha256((P / item["path"]).read_bytes()).hexdigest().upper() != item["sha256"]:
            fails.append(f"provenance input drift: {item['path']}")
    for item in prov["outputs"]:
        n["values"] += 1
        name = Path(item["path"]).name
        if hashlib.sha256((out / name).read_bytes()).hexdigest().upper() != item["sha256"]:
            fails.append(f"provenance output drift: {name}")
    return n["values"], fails


def selftest(run: Path) -> int:
    """Positive control on an unmodified copy; four planted errors must each fail by name."""
    plants = [("table5_f1conv_block.tex", "& 1.50 & [1.25, 1.76]", "& 1.51 & [1.25, 1.76]",
               "F1e100:cifar_full_e100:dann_lrf-naive_branch mean difference"),
              ("table5_f1conv_block.tex", "& 0.36 & [0.10, 0.62] & 0.046 & 0.59 & 0.58 & supported \\\\",
               "& 0.36 & [0.10, 0.62] & 0.046 & 0.59 & 0.58 & not supported \\\\",
               "F1e100:kmnist_full_e100:dann_lrf-dann_random label"),
              ("table3_f1conv_row.tex", "& 0--19 & 240 \\\\", "& 0--19 & 239 \\\\", "Table 3 row A18 runs"),
              ("f1conv_text_numbers.json", "\"frozen30_selected_last_epoch_count\": 5,",
               "\"frozen30_selected_last_epoch_count\": 8,", "fashion|mlp_param frozen30_selected_last_epoch_count")]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        base.mkdir()
        for name in CHECKED_FILES:
            shutil.copy2(OUT_DEFAULT / name, base / name)
        values, fails = check(run, base)
        results.append(("positive control", values, len(fails) == 0, fails[:3]))
        for i, (name, old, new, expect) in enumerate(plants):
            case = Path(tmp) / f"plant{i}"
            shutil.copytree(base, case)
            text = (case / name).read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise Fault(f"planted anchor not unique in {name}: {old!r} ({text.count(old)})")
            (case / name).write_text(text.replace(old, new), encoding="utf-8", newline="\n")
            try:
                _, fails = check(run, case)
            except Fault as exc:
                fails = [f"fault: {exc}"]
            named = any(expect in f for f in fails)
            results.append((f"planted: {expect}", len(fails), named, fails[:2]))
    ok = all(r[2] for r in results)
    print(json.dumps({"selftest": "PASS" if ok else "FAIL",
                      "cases": [{"case": r[0], "count": r[1], "ok": r[2], "first": r[3]} for r in results]}, indent=1))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    try:
        if len(argv) >= 2 and argv[0] == "check":
            out = Path(argv[2]) if len(argv) > 2 else OUT_DEFAULT
            values, fails = check(Path(argv[1]), out)
            print(json.dumps({"checked_values": values, "failures": len(fails)}, indent=1))
            for f in fails[:40]:
                print("FAIL", f)
            return 1 if fails else 0
        if len(argv) == 2 and argv[0] == "selftest":
            return selftest(Path(argv[1]))
        print(__doc__)
        return 2
    except Fault as exc:
        print("FAULT", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
