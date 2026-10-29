"""Compare fixed exposed calibration events across two closed saved journals.

Only previously exposed rows are compared. Exact float32 payload equality is
descriptive across the observed runs; a difference is not a study failure.
This script does not execute a model or establish execution authenticity.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


PREVIOUS = {
    "eval-02-main-prior": "exposed-calibration-prior",
    "eval-02-main-new": "exposed-calibration-new",
}
KINDS = {"clean", "self", "transfer", "repeat"}
ARRAY_FIELDS = ("mask_logits_f32le", "before_f32le", "after_f32le",
                "submitted_f32le", "full_before_f32le", "full_after_f32le")
SCALAR_FIELDS = ("input_ids", "requested_radius", "requested_fraction",
                 "planned_norm", "actual_norm", "relative_error")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path, previous):
    selected = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            old_id = event.get("case_id")
            if previous:
                if old_id not in PREVIOUS:
                    continue
                case_id = PREVIOUS[old_id]
            else:
                if old_id not in PREVIOUS.values():
                    continue
                case_id = old_id
            kind = event.get("kind")
            if kind not in KINDS or (kind == "transfer" and not previous and event.get("dose") != "low"):
                continue
            key = (case_id, kind, event.get("site"), event.get("cue"),
                   event.get("target"), event.get("donor"))
            if key in selected:
                raise ValueError("duplicate calibration event: " + repr(key))
            selected[key] = event
    return selected


def compare(previous, current):
    if previous.keys() != current.keys():
        raise ValueError("calibration event identity sets differ")
    kinds = Counter(key[1] for key in previous)
    if kinds != {"clean": 8, "self": 32, "transfer": 96, "repeat": 8}:
        raise ValueError("unexpected exposed calibration schedule: " + repr(kinds))
    fields_checked = Counter()
    mismatches = []
    for key, before in previous.items():
        after = current[key]
        fields = [name for name in ARRAY_FIELDS + SCALAR_FIELDS if name in before or name in after]
        if "captures_f32le" in before or "captures_f32le" in after:
            left, right = before.get("captures_f32le"), after.get("captures_f32le")
            if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError("capture site identities differ")
            for site in sorted(left):
                name = "captures_f32le." + site
                fields_checked[name] += 1
                if left[site] != right[site]:
                    mismatches.append({"event": list(key), "field": name})
        for name in fields:
            if name not in before or name not in after:
                raise ValueError("required event field missing: " + name)
            fields_checked[name] += 1
            if before[name] != after[name]:
                mismatches.append({"event": list(key), "field": name})
    return {"events": len(previous), "event_kinds": dict(sorted(kinds.items())),
            "fields_checked": dict(sorted(fields_checked.items())),
            "mismatch_count": len(mismatches), "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous")
    parser.add_argument("current")
    parser.add_argument("--previous-sha256", required=True)
    parser.add_argument("--current-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.previous) != args.previous_sha256 or sha256(args.current) != args.current_sha256:
        raise ValueError("journal digest does not match the supplied closed-run pin")
    report = compare(records(args.previous, True), records(args.current, False))
    if sha256(args.previous) != args.previous_sha256 or sha256(args.current) != args.current_sha256:
        raise ValueError("journal changed during comparison")
    report.update({"previous_journal_sha256": args.previous_sha256,
                   "current_journal_sha256": args.current_sha256,
                   "scope": "Exposed calibration only; equality is descriptive saved-byte consistency, not execution attestation."})
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
