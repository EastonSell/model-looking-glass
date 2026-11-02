"""Build tokenizer-only fixed answer-format prompts; never load a model."""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from tokenizers import Tokenizer


PROTOCOL_SHA = "f60b240f1303fbf6ca785d2197e10d182aa1d34e6818ff8ed3e2f30a70c9307d"
SCOPE_SHA = "2530ea0517847731d9c82731a17d58d23504738cc3e85b52fe259fe9bf4ed330"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
INVENTORY_SHA = "ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5"
SUPPORT_SHA = "5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6"
SUCCESSOR_SHA = "6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4"
FACTS = (("france", "France", " Paris", " London"),
         ("japan", "Japan", " Tokyo", " Paris"),
         ("italy", "Italy", " Rome", " London"),
         ("uk", "the United Kingdom", " London", " Paris"))
PREFIXES = (("none", ""), ("neutral", "A short geography prompt appears below.\n"),
            ("spain", "The capital of Spain is Madrid.\n"),
            ("sweden", "The capital of Sweden is Stockholm.\n"))
TEMPLATES = {
    "development": {
        "statement": "In this exercise, {country}'s national capital is",
        "question": "What city serves as the national capital of {country}? Answer:",
    },
    "heldout": {
        "statement": "The national capital city belonging to {country} is",
        "question": "Identify {country}'s national capital city. City:",
    },
}
SEEDS = {"development": 20261031, "heldout": 20261101}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, digest):
    raw = Path(path).read_bytes()
    if sha(raw) != digest:
        raise ValueError("pinned input differs: " + str(path))
    return raw


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def build(tokenizer, inventory, support, successor):
    rows = []
    for stage in ("development", "heldout"):
        for fact, country, factual, counterfactual in FACTS:
            for family in ("statement", "question"):
                suffix = TEMPLATES[stage][family].format(country=country)
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
                for prefix_name, prefix in PREFIXES:
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids if prefix else []
                    if prefix and len(prefix_ids) != 8:
                        raise ValueError("a nonempty control prefix is not eight tokens")
                    text = prefix + suffix
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    if ids != prefix_ids + suffix_ids:
                        raise ValueError("prefix/target suffix token boundary differs")
                    labels = {}
                    for label, continuation in (("factual", factual), ("counterfactual", counterfactual),
                                                ("madrid", " Madrid"), ("stockholm", " Stockholm")):
                        token = tokenizer.encode(continuation, add_special_tokens=False).ids
                        full = tokenizer.encode(text + continuation, add_special_tokens=False).ids
                        if len(token) != 1 or full != ids + token:
                            raise ValueError("single-token continuation boundary differs")
                        labels[label] = {"text": continuation, "token_id": token[0], "full_token_ids": full}
                    if len({v["token_id"] for v in labels.values()}) != 4:
                        raise ValueError("factual, counterfactual and example tokens must differ")
                    rows.append({"case_id": f"answer-format-{stage}-{fact}-{family}-{prefix_name}",
                                 "role": stage, "fact_id": fact, "country": country, "family": family,
                                 "prefix": prefix_name, "prefix_text": prefix, "suffix_text": suffix,
                                 "prompt": text, "prefix_token_ids": prefix_ids, "suffix_token_ids": suffix_ids,
                                 "token_ids": ids, "labels": labels})
    inputs = [tuple(r["token_ids"]) for r in rows]
    own = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if related(inputs[i], inputs[j])]
    prior = [(r["record_id"], tuple(r["token_ids"])) for r in inventory["records"]]
    prior_with_candidates = [(name, ids + (candidate,)) for r in inventory["records"]
                             for name, ids in [(r["record_id"], tuple(r["token_ids"]))]
                             for candidate in r.get("candidate_token_ids", [])]
    def conflicts(known):
        return [(rows[i]["case_id"], name) for i, candidate in enumerate(inputs)
                for name, old in known if related(candidate, old)]
    old = conflicts(prior)
    extended = conflicts(prior_with_candidates)
    natural = conflicts([(str(i), tuple(x)) for i, x in enumerate(support["rows"])])
    future = conflicts([(str(i), tuple(x["input_ids"])) for i, x in enumerate(successor["cases"])])
    if own or old or extended or natural or future:
        raise ValueError(json.dumps({"internal": own, "inventory": old, "inventory_appended": extended,
                                     "public_natural_support": natural, "public_successor": future}))
    permutations = {}
    for stage, seed in SEEDS.items():
        order = list(range(32))
        random.Random(seed).shuffle(order)
        permutations[stage] = order
    return {"version": "answer-format-feasibility-cases-v1", "model_forward": False,
            "protocol_commit": "24e47b40b44b83c1d30eaebc135813408e41eef5",
            "scope_commit": "9fc3680f6d0de3907468f94d75cc976721167bb8",
            "protocol_sha256": PROTOCOL_SHA, "scope_sha256": SCOPE_SHA, "tokenizer_sha256": TOKENIZER_SHA,
            "inventory_version": inventory["version"], "inventory_sha256": INVENTORY_SHA,
            "public_support_sha256": SUPPORT_SHA, "successor_inputs_sha256": SUCCESSOR_SHA,
            "seeds": SEEDS, "execution_permutations": permutations,
            "comparison_counts": {"internal": len(rows) * (len(rows) - 1) // 2,
                                  "public_inputs": len(rows) * len(prior),
                                  "public_appended_metadata": len(rows) * len(prior_with_candidates),
                                  "natural_support": len(rows) * len(support["rows"]),
                                  "successor_pool": len(rows) * len(successor["cases"])},
            "known_prefix_conflicts": 0, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "scope", "tokenizer", "inventory", "support", "successor", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    read_pinned(args.protocol, PROTOCOL_SHA)
    read_pinned(args.scope, SCOPE_SHA)
    read_pinned(args.tokenizer, TOKENIZER_SHA)
    inventory = json.loads(read_pinned(args.inventory, INVENTORY_SHA))
    support = json.loads(read_pinned(args.support, SUPPORT_SHA))
    successor = json.loads(read_pinned(args.successor, SUCCESSOR_SHA))
    if len(support["rows"]) != 128 or len(successor["cases"]) != 256:
        raise ValueError("public support sizes differ")
    output = build(Tokenizer.from_file(str(args.tokenizer)), inventory, support, successor)
    raw = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode() + b"\n"
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": sha(raw), "cases": len(output["cases"]),
                      "counts": output["comparison_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

