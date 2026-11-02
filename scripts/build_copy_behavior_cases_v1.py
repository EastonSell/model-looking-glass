"""Select the frozen public 24-family pool with a tokenizer only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v7 as v7
from looking_glass import pretrained_exposures_v9 as v9
from looking_glass import pretrained_exposures_v12 as v12
from looking_glass import pretrained_exposures_v15 as v15
from looking_glass import pretrained_exposures_v38 as inventory

PROTOCOL = "docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md"
PROTOCOL_SHA = "78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
INVENTORY_SHA = "f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9"
PRIOR = (("experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json",
          "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"),
         ("experiments/distilgpt2_copy_output_dose_cases_v1.json",
          "e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e"))
PERMUTATIONS = ((0,1,2,3,4,5), (4,2,0,3,1,5), (1,3,4,2,0,5), (3,0,4,1,2,5))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def input_sequences(payload):
    """Expand all public inputs, active masked inputs and scored continuations."""
    rows = []
    for record in payload["records"]:
        row = tuple(record["token_ids"])
        if not row or any(type(v) is not int or not 0 <= v < 50257 for v in row):
            raise ValueError("invalid public input token IDs")
        bases = [row]
        if "attention_mask" in record:
            mask = record["attention_mask"]
            if len(mask) != len(row) or any(type(v) is not int or v not in (0, 1) for v in mask):
                raise ValueError("invalid public attention mask")
            active = tuple(v for v, m in zip(row, mask) if m)
            if active and active != row:
                bases.append(active)
        choices = []
        if "target_token_id" in record:
            choices.append(record["target_token_id"])
        choices.extend(record.get("candidate_token_ids", ()))
        choices.extend(v for group in record.get("candidate_token_groups", {}).values() for v in group)
        if any(type(v) is not int or not 0 <= v < 50257 for v in choices):
            raise ValueError("invalid public continuation token")
        for base in bases:
            rows.append(base)
            rows.extend(base + (v,) for v in choices)
        if "scored_prefix_token_ids" in record:
            prefix = tuple(record["scored_prefix_token_ids"])
            scored = record.get("scored_candidate_token_ids", ())
            if (not prefix or any(type(v) is not int or not 0 <= v < 50257 for v in prefix)
                    or any(type(v) is not int or not 0 <= v < 50257 for v in scored)):
                raise ValueError("invalid public scored prefix or candidates")
            rows.append(prefix)
            rows.extend(prefix + (v,) for v in scored)
        elif "scored_candidate_token_ids" in record:
            raise ValueError("scored candidates require a scored prefix")
    return rows


def build(tokenizer_path):
    protocol = (ROOT / PROTOCOL).read_bytes()
    if sha(protocol) != PROTOCOL_SHA or sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("protocol or tokenizer differs")
    pools = re.findall(r"(?m)^\d+\. ([a-z ]+)$", protocol.decode("utf-8"))
    if len(pools) != 8:
        raise ValueError("expected eight frozen pools")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payload, digest = inventory._checked_inventory()
    if digest != INVENTORY_SHA:
        raise ValueError("construction inventory changed")
    known = input_sequences(payload)
    support = json.loads((ROOT / "experiments/natural_agreement_original_public_support_v1.json").read_bytes())
    successor = json.loads((ROOT / "experiments/natural_agreement_successor_inputs_v1.json").read_bytes())
    support_rows = [tuple(row) for row in support["rows"]]
    successor_rows = [tuple(case["input_ids"]) for case in successor["cases"]]
    if len(support_rows) != 128 or len(successor_rows) != 256:
        raise ValueError("public support scope changed")
    known.extend(support_rows)
    known.extend(successor_rows)
    prior_tokens = set(itertools.chain.from_iterable((*v6.PUBLIC_COPY_GROUPS, *v7.PUBLIC_EVALUATION_GROUPS,
        *v9.FOLLOWUP_GROUPS, *v12.FACTORIAL_GROUPS, *v15.PAIRS_DOSE_GROUPS)))
    for path, expected in PRIOR:
        raw = (ROOT / path).read_bytes()
        if sha(raw) != expected:
            raise ValueError("prior case bundle changed")
        prior_tokens.update(itertools.chain.from_iterable(group["token_ids"] for group in json.loads(raw)["groups"]))
    selected_ids, selected_sequences, groups, cases, audits = set(), [], [], [], []
    for pool_index, pool in enumerate(pools):
        words = pool.split()
        seen = set()
        initial = []
        for position, word in enumerate(words):
            ids = tokenizer.encode(" " + word, add_special_tokens=False).ids
            reason = ("repeated-word" if word in seen else "not-single-token" if len(ids) != 1
                      else "prior-copy-token" if ids[0] in prior_tokens else None)
            seen.add(word)
            initial.append((position, word, ids, reason))
        for within in range(3):
            group = 3 * pool_index + within + 1
            entries, eligible = [], []
            for position, word, ids, reason in initial:
                exclusion = reason or ("earlier-selected-token" if ids[0] in selected_ids else None)
                entries.append({"position": position, "word": word, "token_ids": ids, "exclusion": exclusion})
                if exclusion is None:
                    eligible.append((position, word, ids[0]))
            chosen, rejected = None, []
            for combo in itertools.combinations(eligible, 6):
                positions = [item[0] for item in combo]
                local_words, ids = [item[1] for item in combo], [item[2] for item in combo]
                reason = "duplicate-token" if len(set(ids)) != 6 else None
                rows = []
                for rotation in range(2) if reason is None else ():
                    order = tuple(range(6)) if rotation == 0 else (1,2,3,4,5,0)
                    target_index = order[5]
                    for variant, permutation in enumerate(PERMUTATIONS):
                        indices = tuple(order[p] for p in permutation) + order[:5]
                        text = "".join(" " + local_words[j] for j in indices)
                        row = tuple(ids[j] for j in indices)
                        if tokenizer.encode(text, add_special_tokens=False).ids != list(row):
                            reason = "input-tokenization"; break
                        if tokenizer.encode(text + " " + local_words[target_index],
                                            add_special_tokens=False).ids != [*row, ids[target_index]]:
                            reason = "continuation-tokenization"; break
                        rows.append((text, row, row + (ids[target_index],), ids[target_index], rotation, variant))
                    if reason is not None:
                        break
                if reason is None:
                    for i, (_, row, ext, _, _, _) in enumerate(rows):
                        if any(related(x, y) for _, b, bc, _, _, _ in rows[i+1:]
                               for x in (row, ext) for y in (b, bc)):
                            reason = "within-family-relation"; break
                        if any(related(x, prior) for x in (row, ext) for prior in selected_sequences):
                            reason = "earlier-family-relation"; break
                        if any(related(x, prior) for x in (row, ext) for prior in known):
                            reason = "known-public-relation"; break
                if reason is None:
                    chosen = (combo, rows)
                    break
                rejected.append({"positions": positions, "reason": reason})
            if chosen is None:
                raise ValueError(f"construction-ineligible: pool {pool_index + 1}, group {group}")
            combo, rows = chosen
            words_selected, ids = [item[1] for item in combo], [item[2] for item in combo]
            groups.append({"group": group, "pool": pool_index + 1,
                           "split": "development" if group % 2 else "held-out",
                           "words": words_selected, "token_ids": ids})
            audits.append({"group": group, "word_candidates": entries, "rejected_combinations": rejected,
                           "chosen_positions": [item[0] for item in combo]})
            for text, row, ext, target, rotation, variation in rows:
                case_variant = 4 * rotation + variation
                cases.append({"case_id": f"copy-behavior-{group:02d}-{case_variant}", "group": group,
                              "pool": pool_index + 1, "split": "development" if group % 2 else "held-out",
                              "variant": case_variant, "rotation": rotation, "shuffle": variation,
                              "text": text, "token_ids": list(row), "target_id": target})
                selected_sequences.extend((row, ext))
            selected_ids.update(ids)
    if len(selected_ids) != 144 or len(groups) != 24 or len(cases) != 192:
        raise ValueError("unexpected selected family size")
    return {"schema": "distilgpt2-copy-behavior-screen-cases-v1",
            "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "protocol_sha256": PROTOCOL_SHA, "tokenizer_sha256": TOKENIZER_SHA,
            "selection_inventory_version": inventory.INVENTORY_VERSION, "selection_inventory_sha256": digest,
            "selection_inventory_record_count": len(payload["records"]), "public_support_rows": len(support_rows),
            "public_successor_rows": len(successor_rows), "prior_case_sha256": [x[1] for x in PRIOR],
            "model_forward_asserted": False, "permutations": [list(p) for p in PERMUTATIONS],
            "candidate_word_pools": [p.split() for p in pools], "selection_audit": audits,
            "groups": groups, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.tokenizer.is_absolute() or not args.out.is_absolute():
        raise ValueError("absolute tokenizer and output paths are required")
    raw = (json.dumps(build(args.tokenizer), sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": sha(raw), "groups": 24, "cases": 192, "model_forward_asserted": False}))


if __name__ == "__main__":
    main()
