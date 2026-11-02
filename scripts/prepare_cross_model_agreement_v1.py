"""Build the tokenizer-only fixed public cross-model agreement manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

FRAMES = (
    ("Inside the warehouse, the {subject} near the {attractor}", ("chair", "chairs")),
    ("At the marketplace, the {subject} beside the {attractor}", ("table", "tables")),
)
SUBJECTS = (("basket", "baskets"), ("barrel", "barrels"),
            ("cup", "cups"), ("crate", "crates"))
DONOR_LEMMA = (1, 0, 3, 2)
TARGETS = {"gpt": {"is": 318, "are": 389, "was": 373, "were": 547},
           "bert": {"is": 2003, "are": 2024, "was": 2001, "were": 2020}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenized(tokenizer, text, subject, *, masked):
    actual = text + " [MASK]" if masked else text
    encoded = tokenizer(actual, add_special_tokens=masked, return_attention_mask=True,
                        return_offsets_mapping=True, truncation=False)
    marker = ", the " + subject
    start = text.index(marker) + len(", the ")
    end = start + len(subject)
    positions = [i for i, (lo, hi) in enumerate(encoded["offset_mapping"])
                 if lo < end and hi > start]
    if len(positions) != 1 or len(encoded["input_ids"]) > 32:
        raise ValueError(f"subject does not align to one token: {actual}")
    expected = tokenizer.encode(subject if masked else " " + subject,
                                add_special_tokens=False)
    pos = positions[0]
    if len(expected) != 1 or encoded["input_ids"][pos] != expected[0]:
        raise ValueError(f"subject token differs: {actual}")
    ids = encoded["input_ids"]
    if masked:
        if ids.count(tokenizer.mask_token_id) != 1 or ids[-2] != tokenizer.mask_token_id:
            raise ValueError("expected exactly one terminal mask before SEP")
        output_position = len(ids) - 2
    else:
        output_position = len(ids) - 1
        for word, token_id in TARGETS["gpt"].items():
            if tokenizer.encode(text + " " + word, add_special_tokens=False) != ids + [token_id]:
                raise ValueError(f"continuation is not a single appended token: {actual}")
    return {"text": actual, "input_ids": ids, "attention_mask": encoded["attention_mask"],
            "subject_position": pos, "output_position": output_position}


def prepare(gpt_dir, bert_dir):
    tokenizers = {"gpt": AutoTokenizer.from_pretrained(gpt_dir, local_files_only=True),
                  "bert": AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)}
    for name, tokenizer in tokenizers.items():
        for word, token_id in TARGETS[name].items():
            spelling = " " + word if name == "gpt" else word
            if tokenizer.encode(spelling, add_special_tokens=False) != [token_id]:
                raise ValueError(f"target tokenizer differs: {name} {word}")
    forms = [form for pair in SUBJECTS for form in pair]
    forms += [form for _, pair in FRAMES for form in pair]
    for name, tokenizer in tokenizers.items():
        ids = []
        for form in forms:
            word = " " + form if name == "gpt" else form
            encoding = tokenizer.encode(word, add_special_tokens=False)
            if len(encoding) != 1 or encoding[0] in ids:
                raise ValueError(f"non-unique or split token: {name} {form}")
            ids.append(encoding[0])
    cases = []
    for frame, (template, distractors) in enumerate(FRAMES):
        for attractor_number, attractor in enumerate(distractors):
            group = []
            for lemma, pair in enumerate(SUBJECTS):
                for number, subject in enumerate(pair):
                    text = template.format(subject=subject, attractor=attractor)
                    case = {"case_id": f"f{frame}-a{attractor_number}-l{lemma}-n{number}",
                            "index": len(cases), "frame": frame,
                            "attractor_number": attractor_number, "lemma": lemma,
                            "number": number, "subject": subject, "attractor": attractor,
                            "same_number_donor_lemma": DONOR_LEMMA[lemma],
                            "gpt": tokenized(tokenizers["gpt"], text, subject, masked=False),
                            "bert": tokenized(tokenizers["bert"], text, subject, masked=True)}
                    group.append(case)
                    cases.append(case)
            for name in ("gpt", "bert"):
                first = group[0][name]
                for case in group[1:]:
                    other = case[name]
                    pos = first["subject_position"]
                    if (other["subject_position"] != pos or len(other["input_ids"]) != len(first["input_ids"])
                            or other["input_ids"][:pos] != first["input_ids"][:pos]
                            or other["input_ids"][pos+1:] != first["input_ids"][pos+1:]):
                        raise ValueError(f"tokens outside subject differ in {name} group")
    if len(cases) != 32:
        raise ValueError("wrong case count")
    return {"schema": "cross-model-agreement-cases-v1", "targets": TARGETS,
            "gpt": {"model": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                    "tokenizer_sha256": digest(Path(gpt_dir) / "tokenizer.json")},
            "bert": {"model": "distilbert/distilbert-base-uncased", "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
                     "tokenizer_sha256": digest(Path(bert_dir) / "tokenizer.json")},
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", required=True, type=Path)
    parser.add_argument("--bert", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = (json.dumps(prepare(args.gpt, args.bert), sort_keys=True, indent=2) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(output)
    print(f"wrote {len(output)} bytes; SHA-256 {hashlib.sha256(output).hexdigest()}")


if __name__ == "__main__":
    main()
