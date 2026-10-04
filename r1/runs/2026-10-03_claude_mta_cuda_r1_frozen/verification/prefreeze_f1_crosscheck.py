"""Pre-freeze F1 cross-check (R1_ANALYSIS_MANIFEST.json -> pre_freeze_f1_crosscheck).

    python prefreeze_f1_crosscheck.py <prefreeze_outputs.tar.gz> <frozen_units_root> <frozen_plan.json> <out.json>

Rule (frozen before the frozen run was opened): per F1 unit, equality of all history.csv columns; a mismatch is a
determinism finding and never substitutes old for new results. Units are matched by (condition, model, seed) of
the main_grid family. The pre-freeze archive is read in memory (never extracted). Invariants: both sides hold the
same 560 keys; matched + mismatched == 560; every compared history has at least one row.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

pre_arc, frozen_root, plan_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])


def rows_of(text):
    return list(csv.DictReader(io.StringIO(text)))


pre = {}
with tarfile.open(pre_arc, "r:gz") as tar:
    names = {m.name: m for m in tar.getmembers() if m.isfile()}
    for name, m in names.items():
        if name.endswith("/result.json") and "/units/" in name:
            res = json.loads(tar.extractfile(m).read().decode("utf-8"))
            unit = res.get("unit_spec") or {}
            if not unit:
                print("TOOL FAULT: result without unit_spec", name); sys.exit(3)
            hist_name = name.rsplit("/", 1)[0] + "/" + res["history_file"]
            raw = tar.extractfile(names[hist_name]).read()
            key = (unit.get("condition"), unit.get("model"), int(unit.get("seed")))
            if unit.get("family") != "main_grid":
                continue
            if key in pre:
                print("TOOL FAULT: duplicate pre-freeze key", key); sys.exit(3)
            pre[key] = raw

plan = json.loads(plan_path.read_text(encoding="utf-8"))
new = {}
for u in plan["units"]:
    if u["family"] != "main_grid":
        continue
    udir = frozen_root / "units" / u["unit_id"]
    res = json.loads((udir / "result.json").read_text(encoding="utf-8"))
    # bytes on both sides: Path.read_text would translate CRLF and make every byte comparison fail (measured once)
    new[(u["condition"], u["model"], int(u["seed"]))] = (udir / res["history_file"]).read_bytes()

if set(pre) != set(new) or len(new) != 560:
    print("TOOL FAULT: key sets differ or not 560:", len(pre), len(new), len(set(pre) ^ set(new))); sys.exit(3)

identical_bytes, equal_columns, mismatched, details = 0, 0, 0, []
for key in sorted(new):
    a, b = rows_of(pre[key].decode("utf-8")), rows_of(new[key].decode("utf-8"))
    if not a or not b:
        print("TOOL FAULT: empty history", key); sys.exit(3)
    if pre[key] == new[key]:
        identical_bytes += 1
    cols_a, cols_b = list(a[0].keys()), list(b[0].keys())
    same = cols_a == cols_b and len(a) == len(b) and all(x == y for x, y in zip(a, b))
    if same:
        equal_columns += 1
    else:
        mismatched += 1
        diff_cols = sorted({k for x, y in zip(a, b) for k in set(x) | set(y) if x.get(k) != y.get(k)})
        details.append({"key": list(key), "rows_pre": len(a), "rows_frozen": len(b), "columns_pre": cols_a,
                        "columns_frozen": cols_b, "differing_columns": diff_cols})
assert equal_columns + mismatched == len(new), "TOOL FAULT: arithmetic"
doc = {"schema_version": 1, "rule": "per F1 unit, equality of all history.csv columns",
       "prefreeze_archive": pre_arc.name, "prefreeze_archive_sha256": hashlib.sha256(pre_arc.read_bytes()).hexdigest().upper(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
       "units_compared": len(new), "byte_identical_histories": identical_bytes, "all_columns_equal": equal_columns,
       "mismatched": mismatched, "mismatch_details": details[:50]}
out_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"compared {len(new)}; byte-identical {identical_bytes}; all columns equal {equal_columns}; mismatched {mismatched}")
for d in details[:5]:
    print("MISMATCH", d["key"], d["differing_columns"][:8], d["rows_pre"], d["rows_frozen"])
sys.exit(0 if mismatched == 0 else 2)
