"""Construct fixed public cue-by-frame cases using the local tokenizer only."""
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
TRIPLES = (("teacher", "graded", "essays"), ("gardener", "watered", "plants"),
           ("mechanic", "repaired", "engines"))
FRAMES = {
    "main": {"template": "The {role} has {cue} {verb} [MASK] of the {object}.",
             "indices": {"cue": 4, "mask": 6, "object": 9}},
    "subject_relative": {
        "template": "The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.",
        "indices": {"cue": 5, "mask": 7, "object": 10}},
}


def build(tokenizer):
    all_cues = [cue for group in CUE_SETS.values() for klass in ("negative", "affirmative")
                for cue in group[klass]]
    cue_ids = {}
    for cue in all_cues:
        encoded = tokenizer.encode(cue, add_special_tokens=False)
        if encoded.tokens != [cue] or len(encoded.ids) != 1:
            raise ValueError("cue is not one exact WordPiece: " + cue)
        cue_ids[cue] = encoded.ids[0]
    cases = []
    for number, (role, verb, obj) in enumerate(TRIPLES, 1):
        for frame, info in FRAMES.items():
            for cue_set, group in CUE_SETS.items():
                cues = group["negative"] + group["affirmative"]
                indices, rows = info["indices"], {}
                for cue in cues:
                    sentence = info["template"].format(role=role, cue=cue, verb=verb, object=obj)
                    encoded = tokenizer.encode(sentence)
                    if (encoded.ids[indices["cue"]] != cue_ids[cue] or
                            encoded.tokens[indices["cue"]] != cue or
                            encoded.tokens[indices["mask"]] != "[MASK]" or
                            encoded.ids[indices["mask"]] != 103 or
                            encoded.tokens[indices["object"]] != obj or
                            encoded.tokens[indices["mask"] - 1] != verb or
                            encoded.attention_mask != [1] * len(encoded.ids)):
                        raise ValueError("tokenizer layout differs from fixed design")
                    rows[cue] = {"text": sentence, "input_ids": encoded.ids,
                                 "attention_mask": encoded.attention_mask}
                reference = rows[cues[0]]["input_ids"]
                for cue in cues:
                    expected = reference.copy()
                    expected[indices["cue"]] = cue_ids[cue]
                    if rows[cue]["input_ids"] != expected:
                        raise ValueError("cue variants differ outside the selected token")
                cases.append({"id": f"eval-{number:02d}-{frame}-{cue_set}", "split": "evaluation",
                              "triple_id": f"triple-{number:02d}", "frame": frame,
                              "cue_set": cue_set, "role": role, "verb": verb, "object": obj,
                              "indices": indices, "rows": rows})
    rows = [tuple(row["input_ids"]) for case in cases for row in case["rows"].values()]
    if len(cases) != 12 or len(rows) != 48 or len(set(rows)) != 48:
        raise ValueError("fixed case identities are not distinct")
    return {"schema": "bert-cue-factorial-cases-v1",
            "selection": "Twelve public cells, three paired triples and two cue sets crossed with two placements, fixed before this study's forwards.",
            "prior_evidence": "Selected after observing the prior cue-class and held-out-frame outcomes and their dose ranges.",
            "cue_sets": CUE_SETS, "cue_ids": cue_ids, "scored_tokens": {"any": 2151, "some": 2070},
            "frames": FRAMES, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = build(Tokenizer.from_file(str(args.tokenizer)))
    raw = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
    if "torch" in sys.modules:
        raise RuntimeError("tokenizer-only selection imported Torch")
    print(json.dumps({"cases": 12, "rows": 48, "sha256": hashlib.sha256(raw).hexdigest(),
                      "torch_imported": False}, sort_keys=True))


if __name__ == "__main__":
    main()
