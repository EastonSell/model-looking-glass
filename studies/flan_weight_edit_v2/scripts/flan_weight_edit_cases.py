"""Build the model-free prospective FLAN weight-change case bundle."""

import argparse
import hashlib
import json

ASSET_DIR = "work/t5-cache/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
PROTOCOL_SHA256 = "c06fd598e6b5d9cf8d9734c26f6711fbf53b0fbfda08f009ab691ee91dca5478"


def fixed_texts():
    """Return all fixed proposals, including previously exposed development rows."""
    rows = []
    instructions = (
        "Classify the sentiment of this review as positive or negative:",
        "Is this review positive or negative?",
    )
    for phase, subject, indexes in (
        ("discovery", "book", ((0, 1), (32, 33))),
        ("calibration", "movie", ((8, 9), (40, 41))),
    ):
        for instruction_index, (target_index, donor_index) in enumerate(indexes):
            prefix = f"{instructions[instruction_index]} The {subject} was "
            for side, adjective, expected, source_index in (
                ("target", "wonderful", "positive", target_index),
                ("donor", "awful", "negative", donor_index),
            ):
                rows.append({"id": f"{phase}-{instruction_index + 1}-{side}",
                             "role": f"{phase}-{side}", "text": prefix + adjective + ".",
                             "expected": expected, "source_screen_index": source_index})

    targets = (
        "Classify this review as positive or negative: The concert was wonderful.",
        "Give the sentiment label positive or negative: The room was wonderful.",
        "Respond with either positive or negative. Our trip was wonderful.",
        "What is the review's sentiment, positive or negative? Their dinner was wonderful.",
    )
    for number, target in enumerate(targets, 1):
        rows.append({"id": f"evaluation-{number}-target", "role": "evaluation-target",
                     "text": target, "expected": "positive"})
        rows.append({"id": f"evaluation-{number}-donor", "role": "evaluation-donor",
                     "text": target.replace("wonderful", "awful"), "expected": "negative"})

    localities = (
        ("locality-sentiment", "Classify this review as positive or negative: The concert was terrible.", "negative"),
        ("locality-sentiment", "Give the sentiment label positive or negative: The room was excellent.", "positive"),
        ("locality-sentiment", "Respond with either positive or negative. Our trip was disappointing.", "negative"),
        ("locality-sentiment", "What is the review's sentiment, positive or negative? Their dinner was amazing.", "positive"),
        ("locality-target-token", "Repeat exactly this word: wonderful", "wonderful"),
        ("locality-target-token", "Answer with a number: A wonderful triangle has how many sides?", "3"),
        ("locality-unrelated", "Answer with a number: How many sides does a square have?", "4"),
        ("locality-unrelated", "Answer with one word: What is the opposite of hot?", "cold"),
    )
    for number, (role, text, expected) in enumerate(localities, 1):
        rows.append({"id": f"locality-{number}", "role": role, "text": text, "expected": expected})
    return rows


def load_tokenizer(asset_dir):
    from transformers import AutoTokenizer

    manifest_bytes = (asset_dir / "asset-manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083":
        raise ValueError("asset manifest identity differs")
    manifest = json.loads(manifest_bytes)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        content = (asset_dir / name).read_bytes()
        if len(content) != manifest["files"][name]["bytes"] or hashlib.sha256(content).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"tokenizer asset identity differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), use_fast=True, local_files_only=True,
                                               trust_remote_code=False)
    if type(tokenizer).__name__ != "T5TokenizerFast" or tokenizer.eos_token_id != 1 or tokenizer.pad_token_id != 0:
        raise ValueError("tokenizer class or special tokens differ")
    return tokenizer


def build_bundle(tokenizer):
    rows = []
    for source in fixed_texts():
        encoded = tokenizer(source["text"], return_offsets_mapping=True, return_special_tokens_mask=True)
        ids = list(encoded["input_ids"])
        attention = list(encoded["attention_mask"])
        special = list(encoded["special_tokens_mask"])
        offsets = [list(pair) for pair in encoded["offset_mapping"]]
        if not ids or len(ids) > 64 or ids[-1] != 1 or 1 in ids[:-1] or attention != [1] * len(ids):
            raise ValueError(f"invalid source input: {source['id']}")
        expected = list(tokenizer(source["expected"], add_special_tokens=False)["input_ids"])
        if not expected or len(expected) > 3 or any(x in tokenizer.all_special_ids for x in expected):
            raise ValueError(f"invalid candidate answer: {source['id']}")
        row = {**source, "input_ids": ids, "attention_mask": attention,
               "special_tokens_mask": special, "offset_mapping": offsets,
               "expected_ids": expected + [1]}
        if source["role"].endswith(("-target", "-donor")):
            word = "wonderful" if source["role"].endswith("-target") else "awful"
            if source["text"].count(word) != 1:
                raise ValueError("ambiguous adjective span")
            start = source["text"].index(word)
            end = start + len(word)
            matches = [i for i, (a, b) in enumerate(offsets)
                       if a in (start - 1, start) and b == end and a >= 0
                       and source["text"][a:b] in (word, " " + word)
                       and not special[i] and attention[i] == 1]
            if len(matches) != 1 or ids[matches[0]] != (1627 if word == "wonderful" else 16242):
                raise ValueError(f"adjective token not aligned: {source['id']}")
            row["adjective_index"] = matches[0]
        rows.append(row)
    for i in (0, 2, 4, 6, 8, 10, 12, 14):
        left, right = rows[i:i + 2]
        if (len(left["input_ids"]) != len(right["input_ids"])
                or left["attention_mask"] != right["attention_mask"]
                or left["special_tokens_mask"] != right["special_tokens_mask"]
                or [k for k, (a, b) in enumerate(zip(left["input_ids"], right["input_ids"])) if a != b]
                   != [left["adjective_index"]]
                or left["adjective_index"] != right["adjective_index"]):
            raise ValueError("fixed pair has token differences outside the adjective")
    if any(conflicts(rows[i]["input_ids"], rows[j]["input_ids"])
           for i in range(len(rows)) for j in range(i + 1, len(rows))):
        raise ValueError("proposal contains an internal bidirectional prefix conflict")
    return {"schema": "flan-t5-input-row-weight-change-cases-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "asset_manifest_sha256": "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083",
            "cases": rows}


def conflicts(a, b):
    if not a or not b or a[-1] != 1 or b[-1] != 1 or 1 in a[:-1] or 1 in b[:-1]:
        raise ValueError("each input must have exactly one final EOS")
    left, right = a[:-1], b[:-1]
    return left == right[:len(left)] or right == left[:len(right)]


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path(ASSET_DIR))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(load_tokenizer(args.asset_dir))
    data = (json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"path": str(args.output), "cases": len(bundle["cases"]),
                      "sha256": hashlib.sha256(data).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
