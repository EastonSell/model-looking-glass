"""Compare fixed public head cases with a pinned public inventory and pending cases.

No model, protected input, or private membership query is used. The receipt is
limited to complete-row-prefix relations among the explicit public sequences.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

CASES_SHA = "bfc93dbcf33f06c8266b0fabf351dfa9f17ab98d6501ca7380e2cd171bddc8b4"
V43_SHA = "61a80aafd9d4aea075774776d830b017f8e4c5e1a4fd0f2643963d2075a6e4ee"
INVARIANT_SHA = "188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b"
FRANCE_SHA = "c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82"


def read(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("input SHA-256 mismatch: " + path.name)
    return json.loads(raw)


def seq(values):
    if not isinstance(values, list) or not values or any(type(x) is not int or not 0 <= x < 50257 for x in values):
        raise ValueError("invalid token sequence")
    return tuple(values)


def counts(a, b):
    return {"candidate_prefix_of_other": sum(len(x) <= len(y) and x == y[:len(x)] for x in a for y in b),
            "other_prefix_of_candidate": sum(len(y) <= len(x) and y == x[:len(y)] for x in a for y in b)}


def inventory_sequences(inventory):
    rows = set()
    for record in inventory["records"]:
        base = seq(record["token_ids"])
        bases = [base]
        mask = record.get("attention_mask")
        if mask is not None:
            if len(mask) != len(base) or any(type(x) is not int or x not in (0, 1) for x in mask):
                raise ValueError("invalid public attention mask")
            if any(x == 0 for x in mask):
                masked = tuple(t for t, active in zip(base, mask, strict=True) if active)
                if not masked:
                    raise ValueError("empty masked row")
                bases.append(masked)
        choices = []
        if "target_token_id" in record:
            choices.append(record["target_token_id"])
        choices.extend(record.get("candidate_token_ids", []))
        for group in record.get("candidate_token_groups", {}).values():
            choices.extend(group)
        for current in bases:
            rows.add(current)
            for token in choices:
                rows.add(current + seq([token]))
        if "scored_prefix_token_ids" in record:
            prefix = seq(record["scored_prefix_token_ids"])
            rows.add(prefix)
            for token in record["scored_candidate_token_ids"]:
                rows.add(prefix + seq([token]))
        elif "scored_candidate_token_ids" in record:
            raise ValueError("scored candidates without prefix")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "inventory", "pending-invariant", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inventory-sha256", default=V43_SHA)
    parser.add_argument("--france", type=Path)
    parser.add_argument("--france-output", type=Path)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.inventory_sha256) is None:
        raise ValueError("inventory SHA-256 must be lowercase hex")
    if (args.france is None) != (args.france_output is None):
        raise ValueError("France input and output must be provided together")
    cases = read(args.cases, CASES_SHA)["cases"]
    inventory = read(args.inventory, args.inventory_sha256)
    invariant = read(args.pending_invariant, INVARIANT_SHA)
    if len(cases) != 16 or len(invariant["cases"]) != 68 or invariant["answers"] != [318,389,373,547,262,257,287,319]:
        raise ValueError("fixed public case counts or continuations differ")
    candidate = []
    by_case = []
    for case in cases:
        base = seq(case["token_ids"])
        group = [base]
        if len(case["continuations"]) != 4:
            raise ValueError("wrong continuation count")
        for entry, token in zip(case["continuations"], [318,389,373,547], strict=True):
            extended = seq(entry["tokens"])
            if entry["token_id"] != token or extended != base + (token,):
                raise ValueError("invalid head continuation")
            group.append(extended)
        candidate.extend(group)
        by_case.append(group)
    public = inventory_sequences(inventory)
    pending = []
    for case in invariant["cases"]:
        base = seq(case["token_ids"])
        pending.append(base)
        pending.extend(base + (token,) for token in invariant["answers"])
    cross = sum((len(x) <= len(y) and x == y[:len(x)]) or
                (len(y) <= len(x) and y == x[:len(y)])
                for i in range(16) for j in range(i + 1, 16)
                for x in by_case[i] for y in by_case[j])
    result = {"schema":"agreement-head-v4-1-public-overlap-v1", "cases_sha256":CASES_SHA,
              "inventory_sha256":args.inventory_sha256, "pending_invariant_sha256":INVARIANT_SHA,
              "candidate_sequences":len(candidate), "inventory_sequences":len(public), "pending_sequences":len(pending),
              "cross_case_prefix_relations":cross, "inventory_prefix_counts":counts(candidate, public),
              "pending_prefix_counts":counts(candidate, pending),
              "scope":"public bases and declared continuations only; no model or private query"}
    if cross or any(result["inventory_prefix_counts"].values()) or any(result["pending_prefix_counts"].values()):
        raise ValueError("public prefix relation found: " + json.dumps(result, sort_keys=True))
    french_result = None
    if args.france is not None:
        france = read(args.france, FRANCE_SHA)
        rows = france["rows"] + france["projection_basis"]
        if len(france["rows"]) != 27 or len(france["projection_basis"]) != 4:
            raise ValueError("France public case count differs")
        french = []
        for item in rows:
            base = seq(item["token_ids"])
            french.append(base)
            for entry in item["continuations"]:
                extended = seq(entry["appended_token_ids"])
                if extended != base + (entry["token_id"],):
                    raise ValueError("France continuation differs from base")
                french.append(extended)
        compare = counts(candidate, french)
        french_result = {"schema":"agreement-head-v4-1-france-public-overlap-v1", "cases_sha256":CASES_SHA,
                         "france_sha256":FRANCE_SHA, "candidate_sequences":len(candidate),
                         "france_sequences":len(french), "prefix_counts":compare,
                         "scope":"public bases and declared continuations only; no model or private query"}
        if any(compare.values()):
            raise ValueError("France public prefix relation found: " + json.dumps(french_result, sort_keys=True))
    for path, digest in ((args.cases,CASES_SHA), (args.inventory,args.inventory_sha256), (args.pending_invariant,INVARIANT_SHA)):
        read(path,digest)
    if args.france is not None:
        read(args.france,FRANCE_SHA)
    if french_result is not None:
        write(args.france_output, french_result)
    write(args.output, result)


def write(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short write")
        stream.flush()
        os.fsync(stream.fileno())
    print(path.name, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
