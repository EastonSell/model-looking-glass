"""Build public tokenizer-only candidates; deliberately does not load a model."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


PROTOCOL_SHA = "36f1a709a6951866d6835802be71c8a0b3ea1a8bbcbfab4c5c3a97be5392299f"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tokenizer", type=Path, required=True, help="pinned tokenizer.json")
    parser.add_argument("--output", type=Path, required=True, help="new output path")
    args = parser.parse_args()
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise RuntimeError("unexpected tokenizer bytes")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def continuation(prompt, text):
        prefix, token, full = encode(prompt), encode(text), encode(prompt + text)
        if len(token) != 1 or full != prefix + token:
            raise RuntimeError(f"invalid continuation boundary: {prompt!r}, {text!r}")
        return {"text": text, "token_id": token[0], "full_token_ids": full}

    facts = [
        ("france", "France", "France's", " Paris", " London"),
        ("japan", "Japan", "Japan's", " Tokyo", " Paris"),
        ("italy", "Italy", "Italy's", " Rome", " London"),
        ("uk", "the United Kingdom", "the United Kingdom's", " London", " Paris"),
    ]
    cases = []
    for fact_id, country, possessive, factual, alternative in facts:
        prompts = [
            ("discovery", "The capital of Spain is Madrid. The capital of " + country + " is"),
            ("efficacy-calibration", "Name of the national capital of " + country + ":"),
            ("efficacy-calibration", "Country and capital: Spain - Madrid; " + country + " -"),
            ("evaluation", "Q: What is the national capital of " + country + "?\nA:"),
            ("evaluation", "In a geography quiz, " + possessive + " national capital is"),
            ("evaluation", possessive[0].upper() + possessive[1:] + " national capital city is"),
            ("evaluation", "Country: " + country + "\nCapital city:"),
            ("evaluation", "Capital cities: Spain is Madrid; " + country + " is"),
            ("evaluation", "Madrid is Spain's capital. " + possessive[0].upper() + possessive[1:] + " capital is"),
        ]
        role_counts = {}
        for role, prompt in prompts:
            role_counts[role] = role_counts.get(role, 0) + 1
            factual_continuation = continuation(prompt, factual)
            alternative_continuation = continuation(prompt, alternative)
            cases.append({"case_id": f"full-top-{fact_id}-{role}-{role_counts[role]:02}",
                          "role": role, "fact_id": fact_id, "country": country,
                          "prompt": prompt, "token_ids": encode(prompt),
                          "factual": factual_continuation,
                          "counterfactual": alternative_continuation})

    locality = [
        ("city", "The Eiffel Tower stands in the city of", " Paris"),
        ("city", "Tower Bridge spans the River Thames in", " London"),
        ("city", "The Colosseum amphitheatre is in the city of", " Rome"),
        ("city", "The famous Opera House on Bennelong Point stands in the city of", " Sydney"),
        ("geography", "The city of Madrid belongs to the nation of", " Spain"),
        ("geography", "The national language spoken most widely in Brazil is", " Portuguese"),
        ("geography", "Mount Fuji is found in the nation of", " Japan"),
        ("geography", "The Alps include mountain ranges in the continent of", " Europe"),
        ("unrelated", "The number of sides of an octagon is", " eight"),
        ("unrelated", "A week normally has seven", " days"),
        ("unrelated", "At standard atmospheric pressure water freezes at zero degrees", " Celsius"),
        ("unrelated", "A young domestic dog is called a", " puppy"),
    ]
    groups = {}
    for group, prompt, expected in locality:
        groups[group] = groups.get(group, 0) + 1
        accepted = [continuation(prompt, expected)]
        variants = [expected.lower(), " " + expected.strip().capitalize()]
        if expected == " puppy":
            variants.extend([" pup", " Pup"])
        if expected == " eight":
            variants.append(" 8")
        for variant_text in variants:
            if variant_text == expected:
                continue
            try:
                variant = continuation(prompt, variant_text)
            except RuntimeError:
                variant = None
            if variant is not None and variant["token_id"] not in {x["token_id"] for x in accepted}:
                accepted.append(variant)
        cases.append({"case_id": f"full-top-locality-{group}-{groups[group]:02}",
                      "role": "locality", "stratum": group, "prompt": prompt,
                      "token_ids": encode(prompt), "expected": accepted[0], "accepted": accepted})

    if len(cases) != 48 or len({case["case_id"] for case in cases}) != 48:
        raise RuntimeError("wrong case count or repeated ID")
    collisions = []
    for i, a in enumerate(cases):
        for b in cases[:i]:
            x, y = a["token_ids"], b["token_ids"]
            if x[:len(y)] == y or y[:len(x)] == x:
                collisions.append([a["case_id"], b["case_id"]])
    document = {"version": "targeted-edit-full-top-cases-v1", "model_forward": False,
                "protocol_commit": "5a0f0ef151a3a17e263ee902730bb2bf0d00d7e4",
                "protocol_sha256": PROTOCOL_SHA, "tokenizer_sha256": TOKENIZER_SHA,
                "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                "fact_order": [fact[0] for fact in facts], "cases": cases,
                "internal_prefix_collisions": collisions}
    raw = (json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(raw)
    print("output:", args.output)
    print("sha256:", hashlib.sha256(raw).hexdigest())
    print("cases:", len(cases), "internal collisions:", collisions)


if __name__ == "__main__":
    main()
