"""Build the fixed public signed cue-context manifest with the pinned tokenizer."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

CUES = ("never", "rarely", "always", "often")
PAIRS = (
    ("discovery-01", "discovery", (("doctor", "treated", "patients"), ("farmer", "planted", "seeds"))),
    ("evaluation-01", "evaluation", (("clerk", "counted", "coins"), ("student", "answered", "questions"))),
    ("evaluation-02", "evaluation", (("coach", "selected", "players"), ("chef", "sliced", "tomatoes"))),
    ("evaluation-03", "evaluation", (("judge", "reviewed", "cases"), ("painter", "painted", "walls"))),
)
TEMPLATE = "The {role} has {cue} {verb} [MASK] of the {object}."
INDICES = {"cue": 4, "mask": 6, "object": 9}


def build(tokenizer):
    cue_ids = {cue: tokenizer.token_to_id(cue) for cue in CUES}
    if any(value is None for value in cue_ids.values()) or len(set(cue_ids.values())) != 4:
        raise ValueError("cue vocabulary differs")
    pairs, seen = [], set()
    for pair_id, split, triples in PAIRS:
        contexts = []
        for index, (role, verb, obj) in enumerate(triples):
            rows = {}
            for cue in CUES:
                text = TEMPLATE.format(role=role, cue=cue, verb=verb, object=obj)
                encoded = tokenizer.encode(text)
                expected_tokens = ["[CLS]", "the", role, "has", cue, verb,
                                   "[MASK]", "of", "the", obj, ".", "[SEP]"]
                if encoded.tokens != expected_tokens or encoded.attention_mask != [1] * 12:
                    raise ValueError("a fixed word is not one token or the template changed")
                if tuple(encoded.ids) in seen:
                    raise ValueError("duplicate variant")
                seen.add(tuple(encoded.ids))
                rows[cue] = {"text": text, "input_ids": encoded.ids,
                             "attention_mask": encoded.attention_mask}
            reference = rows[CUES[0]]["input_ids"]
            if any([a for j, a in enumerate(row["input_ids"]) if j != INDICES["cue"]] !=
                   [a for j, a in enumerate(reference) if j != INDICES["cue"]] for row in rows.values()):
                raise ValueError("within-context variants differ outside the cue")
            contexts.append({"id": pair_id + ("-a" if index == 0 else "-b"),
                             "role": role, "verb": verb, "object": obj,
                             "indices": INDICES, "rows": rows})
        pairs.append({"id": pair_id, "split": split, "contexts": contexts})
    return {"schema": "bert-cue-context-signed-cases-v1",
            "model": "distilbert/distilbert-base-uncased",
            "model_revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
            "template": TEMPLATE, "indices": INDICES,
            "cue_groups": {"negative_low_frequency": list(CUES[:2]),
                           "affirmative_high_frequency": list(CUES[2:])},
            "cue_ids": cue_ids, "scored_tokens": {cue: tokenizer.token_to_id(cue) for cue in ("any", "some")},
            "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(Tokenizer.from_file(str(args.tokenizer)))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
