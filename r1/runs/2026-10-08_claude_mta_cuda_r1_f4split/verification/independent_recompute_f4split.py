"""Independent recomputation of the MC-NEURO-R1-005 analysis (protocol Section 12, A21). Shares no code with
dann_benchmark/r1/aggregate_f4split.py or the runner: reads the delivered archive directly (tarfile), parses every history
itself, takes the first minimum of the validation loss, rebuilds every summary with its own arithmetic, re-checks the
reproduction gate against the frozen transfer receipt, and compares everything with the analysis outputs.

    python -B independent_recompute_f4split.py <archive.tar.gz> <plan.json> <processed_outputs dir> <frozen receipt.json> <out.json>

Exit 0 = every compared value agrees (floats within 1e-9, integers and flags exactly); 1 = DISAGREE; 3 = TOOL FAULT (the
archive does not hold exactly the planned units, or an input cannot be read). Writes <out.json> in every case it can.
Operation neucom-r1-round-e-20261007 (Cowork-Claude, claude-opus-5-5, xhigh); the verdict carries the clock time of the run.
"""
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

TOL = 1e-9


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def mean_sd(xs):
    n = len(xs)
    m = math.fsum(xs) / n
    return m, math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))


def main(argv) -> int:
    arc_p, plan_p, proc, rec_p, out_p = (Path(a) for a in argv[1:6])
    verdict = {"written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "script_sha256": digest(Path(__file__).read_bytes()),
               "inputs": {}, "disagreements": [], "status": None}

    def finish(status, code):
        verdict["status"] = status
        out_p.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(status, "disagreements", len(verdict["disagreements"]))
        for d in verdict["disagreements"][:10]:
            print("  ", d)
        return code

    try:
        arc_bytes = arc_p.read_bytes()
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        receipt = json.loads(rec_p.read_text(encoding="utf-8"))
        verdict["inputs"] = {"archive": digest(arc_bytes), "plan": digest(plan_p.read_bytes()),
                             "frozen_receipt": digest(rec_p.read_bytes()),
                             "processed": {p.name: digest(p.read_bytes()) for p in sorted(proc.iterdir()) if p.is_file()}}
        with tarfile.open(fileobj=io.BytesIO(arc_bytes), mode="r:gz") as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    except Exception as exc:  # noqa: BLE001
        verdict["tool_fault"] = f"input unreadable: {exc}"
        return finish("TOOL FAULT", 3)

    planned = {u["unit_id"]: u for u in plan["units"]}
    results = {n.split("/")[2]: n for n in files if n.count("/") == 3 and n.split("/")[1] == "units" and n.endswith("/result.json")}
    if set(results) != set(planned):
        verdict["tool_fault"] = (f"archive units {len(results)} != planned {len(planned)}; missing "
                                 f"{sorted(set(planned) - set(results))[:5]}, extra {sorted(set(results) - set(planned))[:5]}")
        return finish("TOOL FAULT", 3)

    correct, hist_sha = {}, {}
    for uid, rname in results.items():
        res = json.loads(files[rname])
        hname = rname.rsplit("/", 1)[0] + "/" + res["history_file"]
        if hname not in files:
            verdict["tool_fault"] = f"history missing for {uid}"
            return finish("TOOL FAULT", 3)
        rows = list(csv.reader(io.StringIO(files[hname].decode("utf-8"))))
        head, body = rows[0], rows[1:]
        iv, it = head.index("val_loss"), head.index("test_acc")
        if len(body) != int(planned[uid]["epochs"]):
            verdict["tool_fault"] = f"{uid}: {len(body)} history rows"
            return finish("TOOL FAULT", 3)
        best = None
        for r in body:
            if best is None or float(r[iv]) < float(best[iv]):
                best = r
        acc = float(best[it])
        correct[uid] = int(round(acc * 10000))
        hist_sha[uid] = digest(files[hname])

    def cell(source, model):
        vals = {}
        for uid, u in planned.items():
            ex = u["extra"]
            if u["model"] != model:
                continue
            if source == "split" and u["family"] == "MC005_F4_split":
                vals[ex["split_seed"]] = correct[uid]
            if source == "order" and ((u["family"] == "MC005_F4_order") or (u["family"] == "MC005_F4_split" and ex["split_seed"] == 0)):
                vals[ex["order_seed"]] = correct[uid]
        assert sorted(vals) == list(range(10)), (source, model, sorted(vals))
        return vals

    def close(a, b):
        return abs(float(a) - float(b)) <= TOL

    def note(what, mine, theirs):
        verdict["disagreements"].append({"what": what, "recomputed": mine, "analysis": theirs})

    # unit table
    for r in csv.DictReader(io.StringIO((proc / "unit_table.csv").read_text(encoding="utf-8"))):
        uid = r["unit_id"]
        if uid not in correct:
            note(f"unit_table unknown unit {uid}", None, uid)
            continue
        if int(r["correct"]) != correct[uid]:
            note(f"unit_table correct {uid}", correct[uid], r["correct"])
        if r["history_sha256"] != hist_sha[uid]:
            note(f"unit_table history_sha256 {uid}", hist_sha[uid], r["history_sha256"])
    # per source and model
    msum = {(r["source"], r["model"]): r for r in csv.DictReader(io.StringIO((proc / "source_model_summary.csv").read_text(encoding="utf-8")))}
    psum = {r["source"]: r for r in csv.DictReader(io.StringIO((proc / "source_paired_summary.csv").read_text(encoding="utf-8")))}
    if len(msum) != 4 or len(psum) != 2:
        note("summary row counts", [4, 2], [len(msum), len(psum)])
    for s in ("split", "order"):
        cells = {m: cell(s, m) for m in ("dann_lrf", "naive_branch")}
        for m, vals in cells.items():
            xs = [vals[v] / 100 for v in range(10)]
            m_, sd_ = mean_sd(xs)
            row = msum.get((s, m), {})
            for key, mine in (("acc_mean", m_), ("acc_sd", sd_), ("acc_min", min(xs)), ("acc_max", max(xs))):
                if key not in row or not close(row[key], mine):
                    note(f"{s} {m} {key}", mine, row.get(key))
        diffs = [cells["dann_lrf"][v] - cells["naive_branch"][v] for v in range(10)]
        xs = [d / 100 for d in diffs]
        m_, sd_ = mean_sd(xs)
        row = psum.get(s, {})
        for key, mine in (("diff_mean", m_), ("diff_sd", sd_), ("diff_min", min(xs)), ("diff_max", max(xs))):
            if key not in row or not close(row[key], mine):
                note(f"{s} paired {key}", mine, row.get(key))
        for key, mine in (("n_positive", sum(d > 0 for d in diffs)), ("n_zero", sum(d == 0 for d in diffs)),
                          ("n_negative", sum(d < 0 for d in diffs))):
            if int(row.get(key, -1)) != mine:
                note(f"{s} paired {key}", mine, row.get(key))
        verdict.setdefault("recomputed", {})[s] = {"diff_mean_pp": m_, "diff_sd_pp": sd_, "n_positive": sum(d > 0 for d in diffs),
                                                  "diffs_correct_images": diffs}
    # gate
    members = receipt["members"]
    grows = {r["unit_id"]: r for r in csv.DictReader(io.StringIO((proc / "reproduction_gate.csv").read_text(encoding="utf-8")))}
    n_gate = 0
    for uid, u in planned.items():
        ex = u["extra"]
        if u["family"] == "MC005_anchor":
            v = u["seed"]
        elif u["family"] == "MC005_F4_split" and ex["split_seed"] == 0:
            v = 0
        else:
            continue
        n_gate += 1
        frozen = f"fashion_full_rand_data__{u['model']}__s{v:02d}"
        want = [members[k]["sha256"].upper() for k in members if f"/units/{frozen}/" in k and k.endswith("/history.csv")]
        mine = len(want) == 1 and want[0] == hist_sha[uid]
        r = grows.get(uid)
        if r is None or (r["equal"] == "True") != mine:
            note(f"gate {uid}", mine, None if r is None else r["equal"])
        verdict.setdefault("gate", {})[uid] = mine
    if n_gate != 10 or len(grows) != 10:
        note("gate row count", n_gate, len(grows))
    verdict["gate_pass"] = all(verdict.get("gate", {}).values()) and n_gate == 10
    return finish("PASS" if not verdict["disagreements"] else "DISAGREE", 0 if not verdict["disagreements"] else 1)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
