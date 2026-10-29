"""Build the fixed BERT agreement suffix cases using tokenizer assets only.

This script neither loads model weights nor evaluates model outputs.  It
refuses to overwrite an existing manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
TOKENIZER_HASHES = {
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
}
SUBJECTS = (("scout", "scouts"), ("monk", "monks"),
            ("poet", "poets"), ("guide", "guides"))
FRAMES = (("At the station", "near", ("visitor", "visitors")),
          ("Beyond the gate", "beside", ("tourist", "tourists")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(snapshot):
    for name, expected in TOKENIZER_HASHES.items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"tokenizer asset mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.mask_token_id != 103 or tokenizer.sep_token_id != 102:
        raise ValueError("unexpected special token identity")
    for pair in (*SUBJECTS, *(frame[2] for frame in FRAMES)):
        if any(len(tokenizer(word, add_special_tokens=False)["input_ids"]) != 1 for word in pair):
            raise ValueError(f"noun is not a single token: {pair}")
    cases = []
    for frame, (prefix, relation, attractors) in enumerate(FRAMES):
        for attractor_number, attractor in enumerate(attractors):
            for lemma, subjects in enumerate(SUBJECTS):
                for number, subject in enumerate(subjects):
                    index = len(cases)
                    stem = f"{prefix}, the {subject} {relation} the {attractor}"
                    variants = {}
                    for name, suffix in (("terminal", ""), ("continued", " ready.")):
                        text = f"{stem} [MASK]{suffix}"
                        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
                        ids = encoded["input_ids"]
                        if (ids[6] != tokenizer(subject, add_special_tokens=False)["input_ids"][0]
                                or ids[10] != tokenizer.mask_token_id
                                or ids[-1] != tokenizer.sep_token_id
                                or len(ids) != (12 if name == "terminal" else 14)):
                            raise ValueError("unexpected subject, mask or suffix alignment")
                        variants[name] = {"text": text, "input_ids": ids,
                                          "attention_mask": encoded["attention_mask"],
                                          "subject_position": 6, "output_position": 10}
                    if (variants["terminal"]["input_ids"][:-1]
                            != variants["continued"]["input_ids"][:11]):
                        raise ValueError("paired prefix through mask differs")
                    cases.append({"index": index, "case_id": f"f{frame}-a{attractor_number}-l{lemma}-n{number}",
                                  "frame": frame, "attractor_number": attractor_number,
                                  "attractor": attractor, "lemma": lemma, "number": number,
                                  "subject": subject, "same_number_donor_lemma": lemma ^ 1,
                                  "variants": variants})
    return {"schema": "bert-agreement-suffix-cases-v1", "model": "distilbert/distilbert-base-uncased",
            "revision": REVISION, "tokenizer_asset_sha256": TOKENIZER_HASHES,
            "targets": {"is": 2003, "are": 2024, "was": 2001, "were": 2020},
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = (json.dumps(build(args.snapshot), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with args.output.open("xb") as out:
        out.write(data)
    print(f"wrote {len(data)} bytes; sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
