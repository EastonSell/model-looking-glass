"""Compare the exact saved farmer prefix across the two closed CPU attempts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

ORIGINAL_SHA256 = "ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05"
CASE = "fresh-02-prior"
CUES = ("never", "rarely", "always", "often")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected():
    events = [{"kind": "clean", "cue": cue} for cue in CUES]
    events.extend(({"kind": "baseline"}, {"kind": "radius"}))
    events.extend({"kind": "self", "site": "cue2", "target": cue} for cue in CUES)
    events.extend({"kind": "transfer", "site": "cue2", "dose": "low", "donor": donor, "target": target}
                  for target in CUES for donor in CUES if donor != target)
    return events[:21]


def farmer_records(path, *, original):
    records = []
    started = not original
    with gzip.open(path, "rb") as stream:
        for line in stream:
            if len(line) > 2 * 1024 * 1024 or not line.endswith(b"\n"):
                raise ValueError("oversized or truncated JSON event")
            event = json.loads(line)
            if original and not started and event.get("case_id") == CASE:
                started = True
            # The original farmer block is the journal suffix; the continuation
            # must start with its 21 consecutive records. Keep one extra original
            # line to detect any unexpected suffix without retaining the journal.
            limit = 22 if original else 21
            if started and len(records) < limit:
                records.append(line)
    return records


def compare(original, continuation):
    schedule = expected()
    reasons = []
    if len(original) != len(schedule):
        reasons.append("original farmer prefix record count differs")
    if len(continuation) < len(schedule):
        reasons.append("continuation farmer prefix record count is insufficient")
    for index, ident in enumerate(schedule):
        if index >= len(original) or index >= len(continuation):
            break
        a, b = json.loads(original[index]), json.loads(continuation[index])
        if (any(a.get(k) != v or b.get(k) != v for k, v in ident.items()) or
                a.get("case_id") != CASE or b.get("case_id") != CASE):
            reasons.append(f"event identity/order differs at index {index}")
        if original[index] != continuation[index]:
            reasons.append(f"serialized event bytes differ at index {index}")
    return {"matched": not reasons, "prefix_records_expected": len(schedule),
            "original_suffix_records_capped_at_22": len(original),
            "continuation_prefix_records_available": len(continuation),
            "reasons": reasons,
            "scope": "Exact saved JSON-line comparison only; not independent execution or runtime attestation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("continuation", type=Path)
    parser.add_argument("--continuation-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.continuation_sha256):
        raise ValueError("invalid continuation journal digest")
    before = (sha256(args.original), sha256(args.continuation))
    if before != (ORIGINAL_SHA256, args.continuation_sha256):
        raise ValueError("journal identity differs")
    report = compare(farmer_records(args.original, original=True),
                     farmer_records(args.continuation, original=False))
    if (sha256(args.original), sha256(args.continuation)) != before:
        raise ValueError("journal changed during comparison")
    report.update({"original_sha256": before[0], "continuation_sha256": before[1]})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
