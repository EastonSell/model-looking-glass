"""Construct only fixed public tokenizer cases; never load or run a model."""
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
from looking_glass import pretrained_exposures_v35 as inventory

PROTOCOL = "docs/DISTILGPT2_COPY_OUTPUT_DOSE_V1.md"
PROTOCOL_SHA = "52bea66f131668ded270f95506b5911719dcd4b967dbc83320124f418187e485"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
INVENTORY_SHA = "3a7ecf81eb7f04759ed5b8ffd0607c5955b634b938756e76eb52d2435aef9610"
PRIOR_CASES = "experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json"
PRIOR_CASES_SHA = "b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69"
PERMUTATIONS = ((0, 1, 2, 3, 4, 5), (4, 2, 0, 3, 1, 5),
                (1, 3, 4, 2, 0, 5), (3, 0, 4, 1, 2, 5))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def build(tokenizer_path):
    protocol = (ROOT / PROTOCOL).read_bytes()
    if sha(protocol) != PROTOCOL_SHA or sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("protocol or tokenizer changed")
    pools = re.findall(r"(?m)^\d+\. `([^`]+)`$", protocol.decode("utf-8"))
    if len(pools) != 8:
        raise ValueError("expected exactly eight frozen ordered pools")
    prior_raw = (ROOT / PRIOR_CASES).read_bytes()
    if sha(prior_raw) != PRIOR_CASES_SHA:
        raise ValueError("prior public case bytes changed")
    prior = json.loads(prior_raw)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payload, digest = inventory._checked_inventory()
    if digest != INVENTORY_SHA:
        raise ValueError("selection inventory changed")
    known = []
    for record in payload["records"]:
        row = tuple(record["token_ids"])
        known.append(row)
        if type(record.get("target_token_id")) is int:
            known.append(row + (record["target_token_id"],))
        known.extend(row + (v,) for v in record.get("candidate_token_ids", ()))
        known.extend(row + (v,) for group in record.get("candidate_token_groups", {}).values() for v in group)
    support = json.loads((ROOT / "experiments/natural_agreement_original_public_support_v1.json").read_bytes())
    successor = json.loads((ROOT / "experiments/natural_agreement_successor_inputs_v1.json").read_bytes())
    support_rows = [tuple(x) for x in support["rows"]]
    successor_rows = [tuple(x["input_ids"]) for x in successor["cases"]]
    if len(support_rows) != 128 or len(successor_rows) != 256:
        raise ValueError("public support scope changed")
    known.extend(support_rows)
    known.extend(successor_rows)
    prior_tokens = set(itertools.chain.from_iterable((
        *v6.PUBLIC_COPY_GROUPS, *v7.PUBLIC_EVALUATION_GROUPS, *v9.FOLLOWUP_GROUPS,
        *v12.FACTORIAL_GROUPS, *v15.PAIRS_DOSE_GROUPS,
    )))
    prior_tokens.update(itertools.chain.from_iterable(group["token_ids"] for group in prior["groups"]))
    selected, selected_rows, groups, cases, audit = set(), [], [], [], []
    for family, text in enumerate(pools, 1):
        eligible, word_log, seen = [], [], set()
        for position, word in enumerate(text.split()):
            ids = tokenizer.encode(" " + word, add_special_tokens=False).ids
            reason = ("repeated-word" if word in seen else "not-single-token" if len(ids) != 1
                      else "prior-copy-token" if ids[0] in prior_tokens
                      else "earlier-selected-token" if ids[0] in selected else None)
            seen.add(word)
            word_log.append({"position": position, "word": word, "token_ids": ids, "exclusion": reason})
            if reason is None:
                eligible.append((position, word, ids[0]))
        rejections = []
        choice = None
        for combo in itertools.combinations(eligible, 6):
            positions, words, ids = ([z[i] for z in combo] for i in range(3))
            if len(set(ids)) != 6:
                rejections.append({"positions": positions, "reason": "duplicate-token"})
                continue
            rows = []
            reason = None
            for permutation in PERMUTATIONS:
                order = permutation + (0, 1, 2, 3, 4)
                full_text = "".join(" " + words[j] for j in order)
                row = tuple(ids[j] for j in order)
                if tokenizer.encode(full_text, add_special_tokens=False).ids != list(row):
                    reason = "input-tokenization"; break
                if tokenizer.encode(full_text + " " + words[5], add_special_tokens=False).ids != [*row, ids[5]]:
                    reason = "continuation-tokenization"; break
                rows.append((full_text, row, row + (ids[5],)))
            if reason is None:
                for i, (_, a, ac) in enumerate(rows):
                    if any(related(x, y) for _, b, bc in rows[i + 1:] for x in (a, ac) for y in (b, bc)):
                        reason = "within-family-relation"; break
                    if any(related(x, old) for x in (a, ac) for old in selected_rows):
                        reason = "earlier-family-relation"; break
                    if any(related(x, old) for x in (a, ac) for old in known):
                        reason = "known-public-relation"; break
            if reason is not None:
                rejections.append({"positions": positions, "reason": reason})
                continue
            choice = combo, rows
            break
        if choice is None:
            raise ValueError(f"construction-ineligible: family {family}")
        combo, rows = choice
        words, ids = [x[1] for x in combo], [x[2] for x in combo]
        groups.append({"words": words, "token_ids": ids})
        audit.append({"word_candidates": word_log, "rejected_combinations": rejections,
                      "chosen_positions": [x[0] for x in combo]})
        for variant, (full_text, row, continuation) in enumerate(rows):
            cases.append({"case_id": f"copy-output-dose-{family:02d}-{variant}",
                          "group": family, "variant": variant, "text": full_text,
                          "token_ids": list(row), "target_id": ids[5]})
            selected_rows.extend((row, continuation))
        selected.update(ids)
    if len(selected) != 48 or len(cases) != 32:
        raise ValueError("unexpected selected tokens or cases")
    return {"schema": "distilgpt2-copy-output-dose-cases-v1",
            "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "protocol_sha256": PROTOCOL_SHA, "tokenizer_sha256": TOKENIZER_SHA,
            "selection_inventory_version": inventory.INVENTORY_VERSION, "selection_inventory_sha256": digest,
            "selection_inventory_record_count": len(payload["records"]),
            "public_support_rows": len(support_rows), "public_successor_rows": len(successor_rows),
            "prior_cases_sha256": PRIOR_CASES_SHA, "model_forward_asserted": False,
            "permutations": [list(p) for p in PERMUTATIONS], "candidate_word_pools": [p.split() for p in pools],
            "selection_audit": audit, "groups": groups, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = (json.dumps(build(args.tokenizer), sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.out.open("xb") as output:
        output.write(raw)
    print(json.dumps({"sha256": sha(raw), "groups": 8, "cases": 32}))


if __name__ == "__main__":
    main()
