"""Verify a delivered R1 output archive against the receipt written on the remote host BEFORE transfer.

    python r1/verify_delivery.py --archive <run>__outputs.tar.gz --receipt <run>__TRANSFER_RECEIPT_outputs.json
        --out <DELIVERY_VERIFICATION.json> --operation-id <op> --tool <tool> --model <model>

Acceptance (EXPERIMENT_DURABILITY_AND_RECOVERY.md 11.1): (1) local archive sha256 equals the receipt value,
(2) the archive's file-member set equals the receipt's member set exactly, (3) every member's sha256 and size
read from the local archive equal the receipt, (4) administrative members are listed by name, not dropped.
The same check is first shown to REJECT four tampered expectations (negative controls); if any tampered
expectation is accepted the tool reports TOOL FAULT and exits 3. Reads no metric. Exit 0 PASS, 2 FAIL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def archive_members(path: Path) -> dict:
    members = {}
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                data = tar.extractfile(m).read()
                if m.name in members:
                    raise ValueError(f"duplicate member {m.name}")
                members[m.name] = {"sha256": hashlib.sha256(data).hexdigest().upper(), "bytes": len(data)}
    return members


def problems(archive_sha: str, members: dict, receipt: dict) -> list:
    out = []
    if archive_sha != str(receipt.get("archive", {}).get("sha256", "")).upper():
        out.append("archive sha256 differs from the receipt")
    expected = receipt.get("members") or {}
    missing = sorted(set(expected) - set(members))
    extra = sorted(set(members) - set(expected))
    if missing:
        out.append(f"{len(missing)} receipt member(s) absent from the archive, first {missing[0]}")
    if extra:
        out.append(f"{len(extra)} archive member(s) absent from the receipt, first {extra[0]}")
    for name in sorted(set(expected) & set(members)):
        exp, got = expected[name], members[name]
        if str(exp.get("sha256", "")).upper() != got["sha256"] or int(exp.get("bytes", -1)) != got["bytes"]:
            out.append(f"member differs: {name}")
    if int(receipt.get("archive", {}).get("file_members", -1)) != len(members):
        out.append("receipt file_members count differs from the archive")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operation-id", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    archive, receipt_path = Path(args.archive), Path(args.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    archive_sha = sha_file(archive)
    members = archive_members(archive)

    # negative controls: each tampered expectation must be rejected by the same check
    controls = {}
    t = copy.deepcopy(receipt); t["archive"]["sha256"] = "0" * 64
    controls["receipt_archive_sha_tampered"] = bool(problems(archive_sha, members, t))
    t = copy.deepcopy(receipt); victim = sorted(t["members"])[0]; del t["members"][victim]
    controls["receipt_member_removed"] = bool(problems(archive_sha, members, t))
    t = copy.deepcopy(receipt); t["members"]["phantom/never_written.json"] = {"sha256": "0" * 64, "bytes": 1}
    controls["receipt_member_added"] = bool(problems(archive_sha, members, t))
    t = copy.deepcopy(receipt); victim = sorted(t["members"])[-1]; t["members"][victim]["sha256"] = "F" * 64
    controls["receipt_member_hash_tampered"] = bool(problems(archive_sha, members, t))
    if not all(controls.values()):
        print("TOOL FAULT: a tampered expectation was accepted", controls)
        return 3

    found = problems(archive_sha, members, receipt)
    verdict = "PASS" if not found else "FAIL"
    now = datetime.now().astimezone()
    record = {
        "schema_version": 1,
        "verdict": verdict,
        "run_id": receipt.get("run_id"),
        "verified_at_local": now.strftime("%Y-%m-%d %H:%M %z"),
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": args.tool,
        "model": args.model,
        "operation_id": args.operation_id,
        "archive": {"path": archive.name, "sha256": archive_sha, "bytes": archive.stat().st_size},
        "receipt": {"path": receipt_path.name, "sha256": sha_file(receipt_path),
                    "written_at_utc": receipt.get("written_at_utc"), "direction": receipt.get("direction")},
        "checks": {
            "archive_sha256_equals_receipt": archive_sha == str(receipt["archive"]["sha256"]).upper(),
            "member_set_equal": set(members) == set(receipt.get("members") or {}),
            "file_members": len(members),
            "member_bytes_and_sha256_equal": not any(p.startswith("member differs") for p in found),
        },
        "negative_controls_rejected": controls,
        "administrative_members": receipt.get("administrative_members", []),
        "administrative_changes": [],
        "administrative_changes_note": "archive kept compressed and verified in place; nothing was extracted or edited locally",
        "invariants_from_receipt": receipt.get("invariants"),
        "problems": found,
        "metrics_read": False,
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(verdict, "archive", archive_sha[:16], "members", len(members), "negative controls rejected", sum(controls.values()), "/", len(controls))
    for p in found[:10]:
        print(" -", p)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
