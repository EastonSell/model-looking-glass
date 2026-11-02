"""Tokenizer-only assembly of the independently reviewed fixed France rows.

No model is loaded. This does not establish execution chronology, semantic
validity, pretraining novelty, or a complete public/private exposure check.
"""
import argparse
import hashlib
import json
import pathlib
import sys

PROTOCOL_SHA = "c2bc0b6e7932eaa3578d5ad30d74e9665574b55ea307e86c957b3a3cd6910c25"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
MODEL = "distilbert/distilgpt2"
REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"

DEVELOPMENT = (
    ("fit", "full-top-france-evaluation-04", "Country: France\nCapital city:"),
    ("calibration-1", "full-top-france-efficacy-calibration-02", "Country and capital: Spain - Madrid; France -"),
    ("calibration-2", "full-top-france-evaluation-05", "Capital cities: Spain is Madrid; France is"),
)
TARGETS = (
    ("T01", "Country: France\nNational capital:"),
    ("T02", "Nation: France\nCapital:"),
    ("T03", "France | capital city:"),
    ("T04", "Complete the geography entry. Country: France; capital:"),
    ("T05", "Country / capital: France /"),
    ("T06", "Capital of France:"),
    ("T07", "Q: What city is the national capital of France?\nA:"),
    ("T08", "Capitals: Portugal is Lisbon; France is"),
)
LOCALITY = (
    ("L01", "Country: Japan\nCapital city:", (" Tokyo",)),
    ("L02", "Country: the United Kingdom\nCapital city:", (" London",)),
    ("L03", "Country: Canada\nCapital city:", (" Ottawa",)),
    ("L04", "Country: Germany\nCapital city:", (" Berlin",)),
    ("L05", "Country: Italy\nCapital city:", (" Rome",)),
    ("L06", "Country: Spain\nCapital city:", (" Madrid",)),
    ("L07", "City: Paris\nCountry:", (" France",)),
    ("L08", "City: London\nSovereign state:", (" UK",)),
    ("L09", "The largest planet in the solar system is", (" Jupiter",)),
    ("L10", "A triangle has three", (" sides", " angles", " corners")),
    ("L11", "In the calendar, Monday is followed by", (" Tuesday",)),
    ("L12", "The opposite of winter is", (" summer",)),
    ("L13", "A baby dog is a", (" puppy", " pup")),
    ("L14", "The plural of dog is", (" dogs",)),
    ("L15", "The first month of the year is", (" January",)),
    ("L16", "In the base-ten system, 2 plus 2 is", (" four", " 4")),
)
BASIS = (
    ("locality-5", "The largest ocean is the"),
    ("locality-6", "Two plus two equals"),
    ("locality-7", "The opposite of hot is"),
    ("locality-8", "A week has seven"),
)


def file_sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def assemble(tokenizer_path, protocol_path):
    import tokenizers
    import transformers
    if file_sha(tokenizer_path) != TOKENIZER_SHA or file_sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("tokenizer or reviewed protocol bytes differ")
    native = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    wrapper = transformers.PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    if native.get_vocab_size(with_added_tokens=True) != 50257:
        raise ValueError("tokenizer vocabulary size differs")

    def encode(text):
        native_out = native.encode(text, add_special_tokens=False)
        wrapped = wrapper(text, add_special_tokens=False, return_attention_mask=True)
        ids = list(native_out.ids)
        mask = list(native_out.attention_mask)
        if (not ids or any(type(item) is not int or not 0 <= item < 50257 for item in ids)
                or mask != [1] * len(ids) or ids != wrapped["input_ids"]
                or mask != wrapped["attention_mask"]):
            raise ValueError("tokenizer outputs disagree or row is invalid")
        return ids, mask

    def row(name, kind, text, answers=(), prior_case=None):
        ids, mask = encode(text)
        continuations = []
        for answer in answers:
            continuation_ids, _ = encode(answer)
            complete_ids, _ = encode(text + answer)
            if len(continuation_ids) != 1 or complete_ids != ids + continuation_ids:
                raise ValueError(f"{name}: declared answer is not a stable appended token: {answer!r}")
            continuations.append({"text": answer, "token_id": continuation_ids[0],
                                  "appended_token_ids": complete_ids})
        if len({entry["token_id"] for entry in continuations}) != len(continuations):
            raise ValueError(f"{name}: accepted token IDs are not distinct")
        return {"case_id": name, "kind": kind, "text": text, "token_ids": ids,
                "attention_mask": mask, "continuations": continuations,
                "prior_case_id": prior_case}

    rows = [row(role, "development", text, (" Paris", " London"), old)
            for role, old, text in DEVELOPMENT]
    rows.extend(row(name, "target", text, (" Paris", " London")) for name, text in TARGETS)
    rows.extend(row(name, "locality", text, answers,
                    "prior-full-top-screen" if name in ("L01", "L02", "L05") else None)
                for name, text, answers in LOCALITY)
    basis = [row(name, "projection-basis", text, (), name) for name, text in BASIS]
    if (len(rows) != 27 or len(basis) != 4 or
            any([answer["token_id"] for answer in item["continuations"]] != [6342, 3576]
                for item in rows[:11]) or
            len({tuple(item["token_ids"]) for item in rows + basis}) != 31):
        raise ValueError("fixed identities, target tokens or distinct rows differ")
    return {"schema": "france-full-top-fixed-cases-v2", "model": MODEL, "revision": REVISION,
            "protocol_sha256": PROTOCOL_SHA, "tokenizer_sha256": TOKENIZER_SHA,
            "assembler_sha256": file_sha(__file__),
            "runtime": {"python": sys.version, "tokenizers": tokenizers.__version__,
                        "transformers": transformers.__version__},
            "rows": rows, "projection_basis": basis,
            "scope": "tokenizer-only fixed rows; exposure, semantics and execution require separate review"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = assemble(args.tokenizer, args.protocol)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False) + "\n").encode("ascii")
    with open(args.output, "xb") as handle:
        handle.write(raw)
        handle.flush()
    print("sha256=" + hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
