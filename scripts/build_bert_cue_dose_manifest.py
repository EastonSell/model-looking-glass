"""Construct a fixed public dose-response manifest using only the pinned tokenizer."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tokenizers import Tokenizer

CUE_SETS = {
    "prior": {"negative": ["never", "rarely"], "affirmative": ["always", "often"]},
    "new": {"negative": ["not", "hardly"], "affirmative": ["really", "certainly"]},
}
TRIPLES = (("calibration", "gardener", "watered", "plants"),
           ("evaluation", "chef", "prepared", "meals"),
           ("evaluation", "farmer", "harvested", "crops"),
           ("evaluation", "pilot", "inspected", "instruments"))
TEMPLATE = "The {role} has {cue} {verb} [MASK] of the {object}."
INDICES = {"cue": 4, "mask": 6, "object": 9}
PRIOR_MANIFEST_SHA = "b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f"


def build(tokenizer, prior):
    all_cues = [cue for group in CUE_SETS.values() for klass in ("negative", "affirmative")
                for cue in group[klass]]
    cue_ids = {}
    for cue in all_cues:
        encoded = tokenizer.encode(cue, add_special_tokens=False)
        if encoded.tokens != [cue] or len(encoded.ids) != 1:
            raise ValueError("cue is not one exact WordPiece: " + cue)
        cue_ids[cue] = encoded.ids[0]
    old = {case["id"]: case for case in prior["cases"]}
    cases = []
    for index, (split, role, verb, obj) in enumerate(TRIPLES):
        triple_id = "exposed-calibration" if split == "calibration" else f"fresh-{index:02d}"
        for cue_set, group in CUE_SETS.items():
            cues = group["negative"] + group["affirmative"]
            rows = {}
            for cue in cues:
                sentence = TEMPLATE.format(role=role, cue=cue, verb=verb, object=obj)
                encoded = tokenizer.encode(sentence)
                if (encoded.ids[INDICES["cue"]] != cue_ids[cue] or
                        encoded.tokens[INDICES["cue"]] != cue or
                        encoded.tokens[INDICES["mask"]] != "[MASK]" or
                        encoded.ids[INDICES["mask"]] != 103 or
                        encoded.tokens[INDICES["object"]] != obj or
                        encoded.tokens[INDICES["mask"] - 1] != verb or
                        encoded.attention_mask != [1] * len(encoded.ids)):
                    raise ValueError("tokenizer layout differs from fixed design")
                rows[cue] = {"text": sentence, "input_ids": encoded.ids,
                             "attention_mask": encoded.attention_mask}
            reference = rows[cues[0]]["input_ids"]
            for cue in cues:
                expected = reference.copy()
                expected[INDICES["cue"]] = cue_ids[cue]
                if rows[cue]["input_ids"] != expected:
                    raise ValueError("cue variants differ outside the selected token")
            previous_id = f"eval-02-main-{cue_set}" if split == "calibration" else None
            if previous_id is not None and old[previous_id]["rows"] != rows:
                raise ValueError("exposed calibration differs from the prior public rows")
            cases.append({"id": f"{triple_id}-{cue_set}", "split": split, "triple_id": triple_id,
                          "frame": "main", "cue_set": cue_set, "role": role, "verb": verb,
                          "object": obj, "indices": INDICES, "rows": rows,
                          "previous_case_id": previous_id})
    rows = [tuple(row["input_ids"]) for case in cases for row in case["rows"].values()]
    if len(cases) != 8 or len(rows) != 32 or len(set(rows)) != 32:
        raise ValueError("fixed case identities are not distinct")
    return {"schema": "bert-cue-dose-response-cases-v1",
            "selection": "Three fresh public evaluation triples plus one explicitly exposed calibration triple, fixed before the new forwards.",
            "prior_evidence": "Chosen after observing the factorial and earlier study outcomes and natural distance ranges.",
            "calibration_reference": {"manifest_sha256": PRIOR_MANIFEST_SHA,
                                      "result_sha256": "d3276451d0ee6793acd5f84063a65cf00ac6f511a0e0bad0048fa64e1da30caf",
                                      "source_commit": "2889532b4b657e3d6f1c00566f8ed5df7f2b2e33"},
            "cue_sets": CUE_SETS, "cue_ids": cue_ids, "scored_tokens": {"any": 2151, "some": 2070},
            "template": TEMPLATE, "indices": INDICES, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("prior_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    previous = args.prior_manifest.read_bytes()
    if hashlib.sha256(previous).hexdigest() != PRIOR_MANIFEST_SHA:
        raise ValueError("prior manifest differs from the exposed calibration reference")
    manifest = build(Tokenizer.from_file(str(args.tokenizer)), json.loads(previous))
    raw = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
    if "torch" in sys.modules:
        raise RuntimeError("tokenizer-only selection imported Torch")
    print(json.dumps({"cases": 8, "rows": 32, "calibration_rows": 8,
                      "sha256": hashlib.sha256(raw).hexdigest(), "torch_imported": False}, sort_keys=True))


if __name__ == "__main__":
    main()
