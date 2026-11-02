"""Build the fixed held-out-frame manifest with a local tokenizer only."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tokenizers import Tokenizer

CUES = ("not", "hardly", "really", "certainly")
TRIPLES = (("inspector", "examined", "packages"),
           ("assistant", "sorted", "reports"),
           ("worker", "cleaned", "tools"))
FRAMES = {
    "subject_relative": {
        "template": "The {role} who has {cue} {verb} [MASK] of the {object} has signed the form.",
        "indices": {"cue": 5, "mask": 7, "object": 10},
    },
    "causal_adjunct": {
        "template": "The clerk is writing a note because the {role} has {cue} {verb} [MASK] of the {object}.",
        "indices": {"cue": 11, "mask": 13, "object": 16},
    },
}


def build(tokenizer):
    cue_ids = {}
    for cue in CUES:
        token = tokenizer.encode(cue, add_special_tokens=False)
        if token.tokens != [cue] or len(token.ids) != 1:
            raise ValueError("cue is not one exact WordPiece: " + cue)
        cue_ids[cue] = token.ids[0]
    cases = []
    for frame, info in FRAMES.items():
        indices = info["indices"]
        for number, (role, verb, obj) in enumerate(TRIPLES, 1):
            rows = {}
            for cue in CUES:
                text = info["template"].format(role=role, verb=verb, object=obj, cue=cue)
                encoded = tokenizer.encode(text)
                if (encoded.ids[indices["cue"]] != cue_ids[cue] or
                        encoded.tokens[indices["cue"]] != cue or
                        encoded.tokens[indices["mask"]] != "[MASK]" or
                        encoded.ids[indices["mask"]] != 103 or
                        encoded.tokens[indices["object"]] != obj or
                        encoded.tokens[indices["mask"] - 1] != verb or
                        encoded.attention_mask != [1] * len(encoded.ids)):
                    raise ValueError("tokenizer layout differs from fixed design")
                rows[cue] = {"text": text, "input_ids": encoded.ids,
                             "attention_mask": encoded.attention_mask}
            reference = rows[CUES[0]]["input_ids"]
            for cue in CUES:
                expected = reference.copy()
                expected[indices["cue"]] = cue_ids[cue]
                if rows[cue]["input_ids"] != expected:
                    raise ValueError("cue variants differ outside the selected token")
            cases.append({"id": f"eval-{frame}-{number:02d}", "split": "evaluation", "frame": frame,
                          "role": role, "verb": verb, "object": obj, "indices": indices, "rows": rows})
    if len(cases) != 6 or len({tuple(row["input_ids"]) for case in cases for row in case["rows"].values()}) != 24:
        raise ValueError("fixed rows are not distinct")
    return {"schema": "bert-cue-heldout-frames-cases-v1",
            "selection": "Six public evaluation contexts fixed before any follow-up model forward; no new development rows.",
            "prior_evidence": "Selected after observing prior cue-class outcomes and preliminary earlier-study vocabulary analysis.",
            "rejected_pre_forward_frames": [
                {"candidate": "object-relative or coordinated suffix after a matrix cue and mask",
                 "reason": "independent review found the prior cue-to-mask matrix placement was retained"}],
            "cue_ids": cue_ids, "negative_class": ["not", "hardly"],
            "affirmative_class": ["really", "certainly"],
            "scored_tokens": {"any": 2151, "some": 2070}, "frames": FRAMES, "cases": cases}


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
    print(json.dumps({"cases": 6, "rows": 24, "sha256": hashlib.sha256(raw).hexdigest(),
                      "torch_imported": False}, sort_keys=True))


if __name__ == "__main__":
    main()
