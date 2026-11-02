"""Build the fixed public copying cases using only the pinned tokenizer.

This program does not import or load a model.  It refuses to overwrite an
existing output, and it records all word-level exclusions and structural
combination rejections needed to reproduce the first qualifying choice.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass import pretrained_exposures_v6 as v6  # noqa: E402
from looking_glass import pretrained_exposures_v7 as v7  # noqa: E402
from looking_glass import pretrained_exposures_v9 as v9  # noqa: E402
from looking_glass import pretrained_exposures_v12 as v12  # noqa: E402
from looking_glass import pretrained_exposures_v15 as v15  # noqa: E402
from looking_glass import pretrained_exposures_v29 as inventory  # noqa: E402

PROTOCOL_SHA = "40d1aac13f0faa82c048db033f8a4152d8c6efb564274afa829b58042bac5d6d"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
INVENTORY_SHA = "f2dc3ff6cc8befc600cc62de68c9c804f8962c7261b87cad3c94ea7744bb7df6"
POOL_STRINGS = (
    "island desert mountain forest prairie marsh dune delta harbor lagoon reef volcano hillside plateau ravine tundra oasis beach jungle basin meadow canyon",
    "apple pear peach plum lemon lime grape melon cherry berry lettuce olive potato bread muffin cereal bacon honey pasta yogurt apricot mango",
    "lion tiger bear fox wolf deer moose raccoon badger otter seal shark trout robin parrot insect ant bee butterfly salmon lizard rabbit",
    "shovel screwdriver pliers drill axe saw chisel hammer wrench rake spade scissor nail screw bolt nut wire pulley lever needle thread",
    "drum harp banjo bass cymbal saxophone accordion opera orchestra festival chorus composer quartet music tempo harmony soprano trumpet cello whistle",
    "airplane bicycle ship boat van sedan helicopter scooter carriage canoe kayak barge cruiser tram minivan sailboat spaceship hovercraft vessel cable",
    "scarf mittens trousers sandal pajama robe bracelet tie necklace ring buckle zipper sleeve collar apron brooch pendant pocket ribbon cardigan",
    "chair table desk sofa couch lamp clock frame cabinet window drawer stove oven kettle skillet fork blanket sheet mug rug vase",
)
PERMUTATIONS = ((0, 1, 2, 3, 4, 5), (4, 2, 0, 3, 1, 5),
                (1, 3, 4, 2, 0, 5), (3, 0, 4, 1, 2, 5))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def build(tokenizer_path: Path):
    if sha((ROOT / "docs/DISTILGPT2_COPY_HEAD_PROMPT_DOSE_V1.md").read_bytes()) != PROTOCOL_SHA:
        raise ValueError("protocol bytes changed")
    if sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
        raise ValueError("tokenizer bytes changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payload, digest = inventory._checked_inventory()
    if digest != INVENTORY_SHA:
        raise ValueError("selection inventory changed")
    known = [tuple(record["token_ids"]) for record in payload["records"]]
    preceding = set(itertools.chain.from_iterable((
        *v6.PUBLIC_COPY_GROUPS, *v7.PUBLIC_EVALUATION_GROUPS,
        *v9.FOLLOWUP_GROUPS, *v12.FACTORIAL_GROUPS, *v15.PAIRS_DOSE_GROUPS,
    )))
    selected_ids, selected_inputs = set(), []
    groups, cases, audits = [], [], []
    for group_index, pool_string in enumerate(POOL_STRINGS):
        eligible, word_audit, seen_words = [], [], set()
        for position, word in enumerate(pool_string.split()):
            token_ids = tokenizer.encode(" " + word, add_special_tokens=False).ids
            reason = None
            if word in seen_words:
                reason = "repeated-word"
            elif len(token_ids) != 1:
                reason = "not-single-token"
            elif token_ids[0] in preceding:
                reason = "prior-copy-token"
            elif token_ids[0] in selected_ids:
                reason = "earlier-selected-token"
            seen_words.add(word)
            word_audit.append({"position": position, "word": word,
                               "token_ids": token_ids, "exclusion": reason})
            if reason is None:
                eligible.append((position, word, token_ids[0]))
        rejected = []
        chosen = None
        for combo in itertools.combinations(eligible, 6):
            positions = [row[0] for row in combo]
            token_ids = [row[2] for row in combo]
            if len(set(token_ids)) != 6:
                rejected.append({"positions": positions, "reason": "duplicate-token"})
                continue
            inputs = [tuple(token_ids[i] for i in permutation + (0, 1, 2, 3, 4))
                      for permutation in PERMUTATIONS]
            if any(related(left, right) for i, left in enumerate(inputs) for right in inputs[i + 1:]):
                rejected.append({"positions": positions, "reason": "within-family-relation"})
                continue
            if any(related(left, right) for left in inputs for right in selected_inputs):
                rejected.append({"positions": positions, "reason": "earlier-family-relation"})
                continue
            if any(related(left, right) for left in inputs for right in known):
                rejected.append({"positions": positions, "reason": "known-input-relation"})
                continue
            chosen = combo, inputs
            break
        if chosen is None:
            raise ValueError(f"construction-ineligible: group {group_index + 1}")
        combo, inputs = chosen
        words = [row[1] for row in combo]
        token_ids = [row[2] for row in combo]
        groups.append({"words": words, "token_ids": token_ids})
        audits.append({"word_candidates": word_audit, "rejected_combinations": rejected,
                       "chosen_positions": [row[0] for row in combo]})
        for variant, (permutation, input_ids) in enumerate(zip(PERMUTATIONS, inputs)):
            indices = permutation + (0, 1, 2, 3, 4)
            text = "".join(" " + words[index] for index in indices)
            if tokenizer.encode(text, add_special_tokens=False).ids != list(input_ids):
                raise ValueError("complete input did not preserve the selected IDs")
            continuation = tokenizer.encode(text + " " + words[5], add_special_tokens=False).ids
            if continuation != [*input_ids, token_ids[5]]:
                raise ValueError("target continuation did not append as one token")
            cases.append({"case_id": f"copy-prompt-dose-{group_index + 1:02d}-{variant}",
                          "group": group_index + 1, "variant": variant,
                          "text": text, "token_ids": list(input_ids), "target_id": token_ids[5]})
        selected_ids.update(token_ids)
        selected_inputs.extend(inputs)
    if len(selected_ids) != 48 or len(selected_inputs) != 32:
        raise ValueError("selected tokens or inputs are not unique")
    support_path = ROOT / "experiments/natural_agreement_original_public_support_v1.json"
    successor_path = ROOT / "experiments/natural_agreement_successor_inputs_v1.json"
    support = [tuple(row) for row in json.loads(support_path.read_bytes())["rows"]]
    successor = [tuple(row["input_ids"]) for row in json.loads(successor_path.read_bytes())["cases"]]
    if len(support) != 128 or len(successor) != 256:
        raise ValueError("public comparison scope changed")
    for name, rows in (("support", support), ("successor", successor)):
        if any(related(left, right) for left in selected_inputs for right in rows):
            raise ValueError(f"construction-ineligible: collision with public {name}")
    return {
        "schema": "distilgpt2-copy-head-prompt-dose-cases-v1",
        "model_id": "distilbert/distilgpt2",
        "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "protocol_revision": "7b4e4b3b6d3c5697133a80fcb8c5777a22b3d174",
        "protocol_sha256": PROTOCOL_SHA,
        "selection_inventory_version": inventory.INVENTORY_VERSION,
        "selection_inventory_sha256": digest,
        "selection_inventory_record_count": len(known),
        "tokenizer_sha256": TOKENIZER_SHA,
        "public_support_sha256": sha(support_path.read_bytes()),
        "public_support_rows": len(support),
        "public_successor_sha256": sha(successor_path.read_bytes()),
        "public_successor_rows": len(successor),
        "model_forward_asserted": False,
        "permutations": [list(p) for p in PERMUTATIONS],
        "candidate_word_pools": [pool.split() for pool in POOL_STRINGS],
        "selection_audit": audits, "groups": groups, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.tokenizer)
    encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with args.out.open("xb") as output:
        output.write(encoded)
    print(json.dumps({"path": str(args.out), "sha256": sha(encoded),
                      "groups": len(result["groups"]), "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
