"""Tokenize the publicly corrected v4.1 cases; never load or run a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
import tokenizers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
SNAPSHOT = args.snapshot.resolve(strict=True)
PROTOCOL = args.protocol.resolve(strict=True)
PINS = {"tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
        "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
        "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"}
if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != "a2bb3587215f7f472799046a64fd95395eb14279a68ede28612df9816bfbf9b8":
    raise ValueError("protocol bytes differ from the fixed public protocol")
for name, digest in PINS.items():
    if hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() != digest:
        raise ValueError("tokenizer asset differs: " + name)
if tokenizers.__version__ != "0.22.2":
    raise ValueError("tokenizers version differs from the fixed case preparation")
tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True, trust_remote_code=False, use_fast=True)
frames = [("Before the ceremony, the {} near the lanterns", ("singer", "dancer")),
          ("After the performance, the {} beside the museum", ("lawyer", "actor")),
          ("During the lesson, the {} behind the curtains", ("singer", "lawyer")),
          ("Throughout the morning, the {} by the orchard", ("dancer", "actor"))]
extensions = ((" is", 318), (" are", 389), (" was", 373), (" were", 547))
cases = []
for frame_index, (frame, nouns) in enumerate(frames):
    variants = [(lemma, number, lemma if number == "singular" else lemma + "s")
                for lemma in nouns for number in ("singular", "plural")]
    texts = [frame.format(subject) for _, _, subject in variants]
    rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if len({len(row) for row in rows}) != 1 or not 1 <= len(rows[0]) <= 12:
        raise ValueError("quartet lengths do not meet the fixed protocol")
    varied = [i for i in range(len(rows[0])) if len({row[i] for row in rows}) > 1]
    if len(varied) != 1 or len({row[varied[0]] for row in rows}) != 4:
        raise ValueError("quartet does not vary at one distinct subject-token position")
    for local, ((lemma, number, subject), text, row) in enumerate(zip(variants, texts, rows, strict=True)):
        continuations = []
        for suffix, token in extensions:
            full = tokenizer.encode(text + suffix, add_special_tokens=False)
            if full != row + [token]:
                raise ValueError("answer is not the fixed one-token continuation")
            continuations.append({"suffix": suffix, "token_id": token, "tokens": full})
        cases.append({"case_id": f"fresh-head-v4-1-{frame_index + 1:02d}-{local + 1:02d}",
                      "frame": frame_index, "lemma": lemma, "subject": subject, "number": number,
                      "subject_position": varied[0], "text": text, "token_ids": row, "continuations": continuations})
if len(cases) != 16 or len({tuple(case["token_ids"]) for case in cases}) != 16:
    raise ValueError("case count or distinct token rows differ")
manifest = {"schema": "distilgpt2-agreement-head-fresh-context-cases-v4-1", "protocol_commit": "9ee1e5ce84f4fa0cd31fdba12bc86160ee75dfdd",
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "tokenizer_asset_sha256": PINS, "tokenizers_version": tokenizers.__version__,
            "selection_sha256": "5936e7f37b425c597fb6ce55a9172445a1d45d41fdbed7c2f196b12f277dba73",
            "candidate": {"layer": 3, "head": 4}, "comparator": {"layer": 3, "head": 8},
            "model_forward_asserted": False, "cases": cases}
raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
path = args.output
with path.open("xb") as output:
    if output.write(raw) != len(raw):
        raise OSError("short case manifest write")
    output.flush()
    os.fsync(output.fileno())
print("cases", len(cases), "sha256", hashlib.sha256(raw).hexdigest())
