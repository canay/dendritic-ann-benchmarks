"""Independent second check of the tier D table rows and prose numbers of NEURO/manuscript-r1 (MC-NEURO-R1-003).

Date/time: 2026-10-05 17:17 +03:00
Tool: Cowork-Claude
Model, if known: claude-opus-5-5 (max)
Operation ID: neucom-r1-tierd-integration-draft-20261005

Written separately from build_tierd_assets.py and without pandas or numpy. Checked files (figures/r1/out/):
table8_tierd_accuracy.tex, table8_tierd_contrasts.tex, table3_tierd_rows.tex, tierd_text_numbers.json and
PROVENANCE_tierd.json (the file names predate DEC-NEURO-022, which places these rows in the manuscript's Table 7).

- Every accuracy cell is recomputed from the raw per-epoch histories of the extracted run: the first epoch of minimum
  validation loss, the test accuracy there as an integer count out of 10,000, the training accuracy at the last epoch, the
  selected epoch, the number of seeds selected at the last epoch, and the parameter count from result.json. Each history is
  checked against the SHA-256 recorded in its result.json first.
- Every contrast's mean difference, d_z and matched-pairs rank-biserial (average ranks over |d|, zeros dropped) is recomputed
  from the seed-paired integer differences; its interval, Holm-adjusted p and label are compared with the run's independent
  recomputation (verification/independent_recompute.json, which shares no code with the analysis), and the label is checked
  against the A8 rule applied to those values.
- Tolerance: half a unit of the last printed digit (exact equality for counts and labels).
- `selftest` copies the checked files, requires the unmodified copy to pass (positive control) and requires each of four
  planted errors to fail with a message that names the planted item (negative controls).

Usage: python verify_tierd_assets.py check <extracted run folder> [<out dir>]
       python verify_tierd_assets.py selftest <extracted run folder>
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
RECOMPUTE = P / "runs" / "2026-10-04_claude_mta_cuda_r1_tierd" / "verification" / "independent_recompute.json"  # public copy
OUT_DEFAULT = P / "figures" / "r1" / "out"
FAMILIES = {"MC003_D1_tfstem", "MC003_D1_tfstem_e100", "MC003_D2_cifar100", "MC003_D2_cifar100_e100"}
N_TEST = 10000
CHECKED_FILES = ("table8_tierd_accuracy.tex", "table8_tierd_contrasts.tex", "table3_tierd_rows.tex",
                 "tierd_text_numbers.json", "PROVENANCE_tierd.json")

# independently written label maps (not imported from the builder)
GROUP_TO_COND = {"Transformer front end, FashionMNIST": "fashion_full", "Transformer front end, CIFAR-10": "cifar_full",
                 "Flattened CIFAR-100": "cifar100_full"}
ROW_TO_MODEL = {"Transformer + \\DANNLRF{} head": "tf_stem_dann_lrf",
                "Transformer + \\NAIVEBRANCH{} head": "tf_stem_naive_branch",
                "Transformer + MLP head": "tf_stem_mlp", "\\DANNLRF{}": "dann_lrf", "\\NAIVEBRANCH{}": "naive_branch",
                "\\MLPPARAM{}": "mlp_param", "\\DANNRANDOM{}": "dann_random", "\\VANNSAME{}": "vann_same"}
CONTRAST_TO_PAIR = {"Transformer: \\DANNLRF{} head $-$ \\NAIVEBRANCH{} head": ("tf_stem_dann_lrf", "tf_stem_naive_branch"),
                    "Transformer: \\DANNLRF{} head $-$ MLP head": ("tf_stem_dann_lrf", "tf_stem_mlp"),
                    "\\DANNLRF{} $-$ \\NAIVEBRANCH{}": ("dann_lrf", "naive_branch"),
                    "\\DANNLRF{} $-$ \\MLPPARAM{}": ("dann_lrf", "mlp_param"),
                    "\\DANNLRF{} $-$ \\DANNRANDOM{}": ("dann_lrf", "dann_random")}
COND_LABEL_TO_COND = {"FashionMNIST full": "fashion_full", "CIFAR-10 full": "cifar_full", "CIFAR-100 full": "cifar100_full"}
BLOCK_TO_SUFFIX = {"30-epoch protocol (primary labels)": "", "100-epoch budget arm": "_e100"}
LABEL_WORD = {"supported": "supported", "not supported": "not_supported", "reverse": "reverse"}


class Fault(Exception):
    """An input or usage fault (exit 2), never a disagreement."""


def load_units(run: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """(condition, model) -> seed -> recomputed unit record, for the four tier D families only."""
    units_dir = run / "units"
    if not units_dir.is_dir():
        raise Fault(f"no units/ folder under {run}")
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    seen = 0
    for udir in sorted(p for p in units_dir.iterdir() if p.is_dir()):
        res = json.loads((udir / "result.json").read_bytes().decode("utf-8"))
        spec = res["unit_spec"]
        seen += 1
        if spec["family"] not in FAMILIES:
            continue
        raw = (udir / res["history_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != res["history_sha256"].upper():
            raise Fault(f"history hash differs from result.json: {udir.name}")
        recs = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        recs.sort(key=lambda r: int(r["epoch"]))
        if [int(r["epoch"]) for r in recs] != list(range(1, int(spec["epochs"]) + 1)):
            raise Fault(f"epochs incomplete: {udir.name}")
        best_epoch, best_loss = None, None
        for r in recs:
            v = float(r["val_loss"])
            if best_loss is None or v < best_loss:
                best_loss, best_epoch = v, int(r["epoch"])
        at = recs[best_epoch - 1]
        correct = round(float(at["test_acc"]) * N_TEST)
        if abs(float(at["test_acc"]) * N_TEST - correct) > 1e-6:
            raise Fault(f"test accuracy not on the integer scale: {udir.name}")
        key = (spec["condition"], spec["model"])
        seed = int(spec["seed"])
        if seed in cells.setdefault(key, {}):
            raise Fault(f"duplicate unit {key} seed {seed}")
        cells[key][seed] = {"correct": correct, "selected": best_epoch, "epochs": int(spec["epochs"]),
                            "train_last": float(recs[-1]["train_acc"]),
                            "params": int(res["summary"]["trainable_params"]), "family": spec["family"]}
    if seen != 449:
        raise Fault(f"expected 449 unit folders, found {seen}")
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


def header_text(line: str) -> str | None:
    if line.startswith("\\multicolumn") and "\\textit{" in line:
        return line.split("\\textit{", 1)[1].split("}", 1)[0]
    return None


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
    if rec.get("verdict") != "PASS":
        raise Fault("the run's independent recomputation is not PASS")
    rows = rec["rows"]

    def cell_stats(cond: str, model: str) -> dict:
        seeds = cells.get((cond, model), {})
        if sorted(seeds) != list(range(20)):
            raise Fault(f"cell {cond}|{model} does not hold seeds 0-19")
        acc = [seeds[s]["correct"] / 100.0 for s in range(20)]
        params = {seeds[s]["params"] for s in range(20)}
        if len(params) != 1:
            raise Fault(f"parameter count differs within {cond}|{model}")
        return {"acc_mean_pp": mean(acc), "acc_sd_pp": sd(acc), "acc_min_pp": min(acc), "acc_max_pp": max(acc),
                "train_pct": 100.0 * mean([seeds[s]["train_last"] for s in range(20)]),
                "sel_mean": mean([float(seeds[s]["selected"]) for s in range(20)]),
                "sel_last": sum(1 for s in range(20) if seeds[s]["selected"] == seeds[s]["epochs"]),
                "params": params.pop()}

    # ---------------------------------------------------------------- accuracy panel
    cond = None
    seen_cells = set()
    for line in (out / "table8_tierd_accuracy.tex").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\midrule"):
            continue
        head = header_text(line)
        if head is not None:
            cond = GROUP_TO_COND[head]
            continue
        c = split_row(line)
        if len(c) != 10:
            raise Fault(f"accuracy row with {len(c)} cells: {line[:60]}")
        model = ROW_TO_MODEL[c[0]]
        for offset, suffix in ((2, ""), (6, "_e100")):
            st = cell_stats(cond + suffix, model)
            seen_cells.add((cond + suffix, model))
            m_txt, s_txt = c[offset].split("$\\pm$")
            tag = f"accuracy {cond + suffix}|{model}"
            close(num(m_txt), st["acc_mean_pp"], 2, tag + " mean")
            close(num(s_txt), st["acc_sd_pp"], 2, tag + " SD")
            close(num(c[offset + 1]), st["train_pct"], 1, tag + " training accuracy at the last epoch")
            close(num(c[offset + 2]), st["sel_mean"], 1, tag + " mean selected epoch")
            equal(int(c[offset + 3]), st["sel_last"], tag + " seeds selected at the last epoch")
            equal(int(num(c[1])), st["params"], tag + " parameters")
    if len(seen_cells) != 22:
        fails.append(f"accuracy panel covers {len(seen_cells)} cells, expected 22")

    # ---------------------------------------------------------------- contrast panel
    suffix = None
    seen_ids = set()
    for line in (out / "table8_tierd_contrasts.tex").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\midrule"):
            continue
        head = header_text(line)
        if head is not None:
            suffix = BLOCK_TO_SUFFIX[head]
            continue
        c = split_row(line)
        if len(c) != 8:
            raise Fault(f"contrast row with {len(c)} cells: {line[:60]}")
        a, b = CONTRAST_TO_PAIR[c[0]]
        base = COND_LABEL_TO_COND[c[1]]
        cond_s = base + suffix
        group = ("D1" if a.startswith("tf_stem") else "D2") + suffix
        cid = f"{group}:{cond_s}:{a}-{b}"
        seen_ids.add(cid)
        sa, sb = cells[(cond_s, a)], cells[(cond_s, b)]
        d = [sa[s]["correct"] - sb[s]["correct"] for s in range(20)]
        mdiff = mean([float(x) for x in d]) / 100.0
        dz = mean([float(x) for x in d]) / sd([float(x) for x in d])
        close(num(c[2]), mdiff, 2, f"{cid} mean difference")
        close(num(c[5]), dz, 2, f"{cid} d_z")
        close(num(c[6]), rank_biserial(d), 2, f"{cid} rank-biserial")
        r = rows.get(cid)
        if r is None:
            fails.append(f"{cid}: not in the independent recomputation")
            continue
        close(r["mean_pp"], mdiff, 9, f"{cid} independent mean vs raw")   # three-way agreement
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
        fails.append(f"contrast panel ids differ from the independent recomputation: {sorted(set(rows) ^ seen_ids)}")

    # ---------------------------------------------------------------- Table 3 rows
    fam_count: dict[str, int] = {}
    for seeds in cells.values():
        for u in seeds.values():
            fam_count[u["family"]] = fam_count.get(u["family"], 0) + 1
    want = {"D1": fam_count.get("MC003_D1_tfstem", 0) + fam_count.get("MC003_D1_tfstem_e100", 0),
            "D2": fam_count.get("MC003_D2_cifar100", 0) + fam_count.get("MC003_D2_cifar100_e100", 0)}
    for line in (out / "table3_tierd_rows.tex").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = split_row(line)
        equal(int(num(c[-1])), want[c[0]], f"Table 3 row {c[0]} runs")

    # ---------------------------------------------------------------- prose numbers
    tn = json.loads((out / "tierd_text_numbers.json").read_bytes().decode("utf-8"))
    for key, v in tn["cells"].items():
        cnd, model = key.split("|")
        st = cell_stats(cnd, model)
        for field, value in (("acc_mean_pp", st["acc_mean_pp"]), ("acc_sd_pp", st["acc_sd_pp"]),
                             ("acc_min_pp", st["acc_min_pp"]), ("acc_max_pp", st["acc_max_pp"]),
                             ("train_acc_last_epoch_mean_pct", st["train_pct"]), ("selected_epoch_mean", st["sel_mean"])):
            close(float(v[field]), value, 6, f"text numbers {key} {field}")
        equal(int(v["selected_last_epoch_count"]), st["sel_last"], f"text numbers {key} selected_last_epoch_count")
        equal(int(v["effective_params"]), st["params"], f"text numbers {key} effective_params")
    for key, v in tn["budget_change"].items():
        cnd, model = key.split("|")
        close(float(v["acc_mean_e100_minus_30_pp"]),
              cell_stats(cnd + "_e100", model)["acc_mean_pp"] - cell_stats(cnd, model)["acc_mean_pp"], 6,
              f"text numbers budget change {key}")
    for cid, v in tn["contrasts"].items():
        group, cond_s, pair = cid.split(":")
        a, b = pair.split("-")
        sa, sb = cells[(cond_s, a)], cells[(cond_s, b)]
        d = [sa[s]["correct"] - sb[s]["correct"] for s in range(20)]
        close(float(v["mean_diff_pp"]), mean([float(x) for x in d]) / 100.0, 6, f"text numbers {cid} mean")
        close(float(v["d_z"]), mean([float(x) for x in d]) / sd([float(x) for x in d]), 6, f"text numbers {cid} d_z")
        close(float(v["rank_biserial"]), rank_biserial(d), 6, f"text numbers {cid} rank-biserial")
        equal(int(v["positive_seeds"]), sum(1 for x in d if x > 0), f"text numbers {cid} positive seeds")
        equal(int(v["negative_seeds"]), sum(1 for x in d if x < 0), f"text numbers {cid} negative seeds")
        r = rows[cid]
        close(float(v["holm_p"]), r["p_holm"], 9, f"text numbers {cid} Holm p")
        close(float(v["ci95_pp"][0]), r["ci95_pp"][0], 6, f"text numbers {cid} interval low")
        close(float(v["ci95_pp"][1]), r["ci95_pp"][1], 6, f"text numbers {cid} interval high")
        equal(str(v["outcome"]), r["label"], f"text numbers {cid} label")
    for rule_id, outcome in tn["composite"].items():
        if rule_id in rec["composites"]:
            equal(outcome, rec["composites"][rule_id], f"text numbers composite {rule_id}")

    # ---------------------------------------------------------------- provenance of the builder run
    prov = json.loads((out / "PROVENANCE_tierd.json").read_bytes().decode("utf-8"))
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
    plants = [("table8_tierd_accuracy.tex", "21.40 $\\pm$ 0.37", "21.46 $\\pm$ 0.37", "accuracy cifar100_full|dann_lrf mean"),
              ("table8_tierd_contrasts.tex", "& 0.55 & [0.35, 0.76] & $<$0.001 & 1.14 & 0.88 & supported \\\\",
               "& 0.55 & [0.35, 0.76] & $<$0.001 & 1.14 & 0.88 & not supported \\\\",
               "D2_e100:cifar100_full_e100:dann_lrf-naive_branch label"),
              ("table3_tierd_rows.tex", "& 0--19 & 240 \\\\", "& 0--19 & 239 \\\\", "Table 3 row D1 runs"),
              ("table8_tierd_accuracy.tex", "& 95.9 & 3 \\\\", "& 95.9 & 4 \\\\",
               "accuracy cifar100_full_e100|naive_branch seeds selected at the last epoch")]
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
