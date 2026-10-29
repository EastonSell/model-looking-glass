"""Independently check public planned copying-screen reservation; no model is used."""

import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(r"E:\Codex\model-looking-glass\work\copy-behavior-screen-v1")
V40 = Path(r"E:\Codex\model-looking-glass\work\agreement-invariant-noun-v1\inventory-v40.canonical.json")
V41 = Path(r"E:\Codex\model-looking-glass\work\copy-behavior-inventory-v41.canonical.json")
CASES = ROOT / "experiments/distilgpt2_copy_behavior_screen_cases_v1.json"
REPORT = Path("work/reviews/copy-behavior-v41-independent-v1.json")
V40_SHA = "685391b672c356dd627923681c4a93fc9a310c03a40fdd22e0b29af2ae9f7be3"
V41_SHA = "110ebcf674168e794544c257d158b3c3242f98bcf108375cdc073bb145e2bfad"
CASE_SHA = "96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, f"digest differs: {path}")
    return json.loads(data)


def index(rows):
    full = set(rows)
    return full, {row[:i] for row in full for i in range(1, len(row) + 1)}


def related(row, known):
    full, prefixes = known
    return row in prefixes or any(row[:i] in full for i in range(1, len(row) + 1))


def expanded_public(records):
    rows = []
    for record in records:
        base = tuple(record["token_ids"])
        require(base and all(type(x) is int and 0 <= x < 50257 for x in base), "invalid public input")
        variants = [base]
        if "attention_mask" in record:
            mask = record["attention_mask"]
            require(len(mask) == len(base) and all(type(x) is int and x in (0, 1) for x in mask),
                    "invalid attention mask")
            active = tuple(token for token, flag in zip(base, mask) if flag)
            if active and active != base:
                variants.append(active)
        choices = []
        if "target_token_id" in record:
            choices.append(record["target_token_id"])
        choices.extend(record.get("candidate_token_ids", []))
        for group in record.get("candidate_token_groups", {}).values():
            choices.extend(group)
        require(all(type(x) is int and 0 <= x < 50257 for x in choices), "invalid public continuation")
        for variant in variants:
            rows.append(variant)
            rows.extend(variant + (token,) for token in choices)
        if "scored_prefix_token_ids" in record:
            prefix = tuple(record["scored_prefix_token_ids"])
            scores = record.get("scored_candidate_token_ids", [])
            require(prefix and all(type(x) is int and 0 <= x < 50257 for x in prefix + tuple(scores)),
                    "invalid scored public prefix")
            rows.append(prefix)
            rows.extend(prefix + (token,) for token in scores)
        else:
            require("scored_candidate_token_ids" not in record, "scored tokens lack a prefix")
    return rows


def main():
    previous, current, data = read(V40, V40_SHA), read(V41, V41_SHA), read(CASES, CASE_SHA)
    old, cases = previous["records"], data["cases"]
    require(len(old) == 1251 and len(cases) == 192 and len(current["records"]) == 1443,
            "record count differs")
    require(current["records"][:len(old)] == old, "predecessor records changed")
    support = read(ROOT / "experiments/natural_agreement_original_public_support_v1.json",
                   "5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6")
    successor = read(ROOT / "experiments/natural_agreement_successor_inputs_v1.json",
                     "6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4")
    require(len(support["rows"]) == 128 and len(successor["cases"]) == 256, "public scopes differ")
    public_rows = expanded_public(old)
    public_rows.extend(tuple(row) for row in support["rows"])
    public_rows.extend(tuple(row["input_ids"]) for row in successor["cases"])
    known = index(public_rows)
    expected = copy.deepcopy(previous)
    seen_ids = {r["record_id"] for r in old}
    require(len(seen_ids) == len(old), "duplicate predecessor record ID")
    new_rows = []
    for i, case in enumerate(cases):
        group, variant = i // 8 + 1, i % 8
        require(case["group"] == group and case["variant"] == variant
                and case["rotation"] == variant // 4 and case["shuffle"] == variant % 4
                and case["split"] == ("development" if group % 2 else "held-out"),
                "fixed case order or split differs")
        row, target = tuple(case["token_ids"]), case["target_id"]
        ident = "planned-" + case["case_id"]
        require(ident not in seen_ids and len(row) == 11 and
                all(type(x) is int and 0 <= x < 50257 for x in row + (target,)), "invalid case identity")
        seen_ids.add(ident)
        candidate = (row, row + (target,))
        require(not any(related(value, known) for value in candidate), "public-prefix collision")
        for prior in new_rows:
            prior_index = index(prior)
            require(not any(related(value, prior_index) for value in candidate), "new-case collision")
        new_rows.append(candidate)
        expected["records"].append({"record_id": ident, "category": "planned", "token_ids": list(row),
            "target_token_id": target, "public_text": case["text"],
            "reference_commit": "8efc8e019be0929ca6344eca3f94de1f5c5cbdd7",
            "references": ["docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md",
                           "experiments/distilgpt2_copy_behavior_screen_cases_v1.json"],
            "provenance": "Public copying-behavior proposal; no model forward is asserted. "
                          f"Case SHA-256 {CASE_SHA}."})
    expected["version"] = "distilgpt2-development-exposures-v41"
    expected["predecessor"] = {"version": "distilgpt2-development-exposures-v40", "sha256": V40_SHA}
    expected["update"] = {"planned_copy_behavior_screen_rows": 192}
    require(expected == current, "independent complete payload reconstruction differs")
    reconstructed = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                               allow_nan=False).encode()
    require(hashlib.sha256(reconstructed).hexdigest() == V41_SHA, "canonical digest differs")
    prefix_count = len(index(tuple(r["token_ids"]) for r in expected["records"])[1])
    require(prefix_count == 9011, "prefix count differs")
    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules), "torch loaded")
    report = {"status": "pass", "v40_sha256": V40_SHA, "v41_sha256": V41_SHA,
              "case_sha256": CASE_SHA, "predecessor_records_preserved": len(old),
              "planned_rows": len(new_rows), "input_and_continuation_rows": 2 * len(new_rows),
              "expanded_public_sequences": len(known[0]), "public_support_rows": 128,
              "public_successor_rows": 256, "public_prefix_relations": 0,
              "distinct_case_prefix_relations": 0, "input_prefix_count": prefix_count,
              "scope": "Known public snapshot only; no model, protected cases or pretraining attestation."}
    raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    with REPORT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"report": str(REPORT), "sha256": hashlib.sha256(raw).hexdigest(), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
