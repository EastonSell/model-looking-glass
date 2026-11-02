"""Tokenize fixed public agreement-propagation cases; never load a model."""
import argparse
import hashlib
import json
from pathlib import Path


PINS = {
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}
FRAMES = (
    ("Before the festival, the {} near the river", ("captain", "artist")),
    ("After the rehearsal, the {} beside the violins", ("merchant", "poet")),
    ("At the workshop, the {} behind the lanterns", ("captain", "merchant")),
    ("Throughout the evening, the {} near the garden", ("artist", "poet")),
)
EXTENSIONS = ((" is", 318), (" are", 389), (" was", 373), (" were", 547))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    for name, expected in PINS.items():
        if hashlib.sha256((a.snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError("tokenizer asset differs: " + name)
    import tokenizers
    from transformers import AutoTokenizer
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("unexpected tokenizer runtime")
    tokenizer = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    cases = []
    for frame_index, (template, lemmas) in enumerate(FRAMES):
        variants = [(lemma, number, lemma if number == "singular" else lemma + "s")
                    for lemma in lemmas for number in ("singular", "plural")]
        texts = [template.format(subject) for _, _, subject in variants]
        rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        if len({len(row) for row in rows}) != 1 or not 1 <= len(rows[0]) <= 12:
            raise ValueError("invalid quartet lengths")
        positions = [i for i in range(len(rows[0])) if len({row[i] for row in rows}) != 1]
        if len(positions) != 1 or len({row[positions[0]] for row in rows}) != 4:
            raise ValueError("quartet does not vary in exactly one distinct subject token")
        for local, ((lemma, number, subject), text, row) in enumerate(zip(variants, texts, rows, strict=True)):
            continuations = []
            for suffix, token in EXTENSIONS:
                full = tokenizer.encode(text + suffix, add_special_tokens=False)
                if full != row + [token]:
                    raise ValueError("continuation is not the fixed single token")
                continuations.append({"suffix": suffix, "token_id": token, "tokens": full})
            cases.append({"case_id": f"agreement-propagation-v1-{frame_index + 1:02d}-{local + 1:02d}",
                          "frame": frame_index, "local_index": local, "lemma": lemma,
                          "subject": subject, "number": number, "subject_position": positions[0],
                          "text": text, "token_ids": row, "continuations": continuations})
    if len(cases) != 16 or len({tuple(x["token_ids"]) for x in cases}) != 16:
        raise ValueError("case count or uniqueness differs")
    manifest = {"schema": "distilgpt2-agreement-head-propagation-cases-v1",
                "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
                "model_id": "distilbert/distilgpt2", "model_revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
                "tokenizer_asset_sha256": PINS, "tokenizers_version": tokenizers.__version__,
                "candidate": {"layer": 3, "head": 4}, "comparator": {"layer": 3, "head": 8},
                "model_forward_asserted": False, "cases": cases}
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with a.output.open("xb") as f:
        if f.write(raw) != len(raw):
            raise OSError("short manifest write")
    print(json.dumps({"cases": len(cases), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
