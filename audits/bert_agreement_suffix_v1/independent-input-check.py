"""Independent model-free check of one BERT suffix proposal and named inputs.

The vocabulary reconstruction handles these fixed ASCII sentences only; it
does not load a tokenizer library or a checkpoint.  Public row references
may repeat, and unknown/private/pretraining exposure is outside the scope.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

if not __debug__:
    raise RuntimeError("this checker requires Python without -O or -OO")

CASE_HASH = "1da4f8e4773009fedc750480c21597fc60ecaa0c990dcff0af15a5c7101c3dca"
VOCAB_HASH = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
PENDING_HASH = "4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7"
OLD_REPORT_HASH = "b9cc99838fa98c99d1640ff09482ba4e6604d4390e4448d5aa19b7570690d695"
SUBJECTS = (("scout", "scouts"), ("monk", "monks"), ("poet", "poets"), ("guide", "guides"))
FRAMES = (("At the station", "near", ("visitor", "visitors")),
          ("Beyond the gate", "beside", ("tourist", "tourists")))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--git", required=True, type=Path, help="trusted absolute Git executable")
parser.add_argument("--repo", required=True, type=Path, help="checkout containing the cited Git objects")
parser.add_argument("--cases", required=True, type=Path, help="the pinned proposed manifest")
parser.add_argument("--vocab", required=True, type=Path, help="the pinned vocab.txt")
parser.add_argument("--pending", required=True, type=Path, help="the pinned pending context-transfer manifest")
parser.add_argument("--output", required=True, type=Path, help="new report path; never overwritten")
args = parser.parse_args()
if not args.git.is_absolute() or not args.git.is_file():
    raise ValueError("--git must be an existing absolute executable")

def sha(b):
    return hashlib.sha256(b).hexdigest()


def strict_json(b):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("invalid JSON constant " + value)
    return json.loads(b, object_pairs_hook=pairs, parse_constant=invalid)


def git_bytes(commit, path):
    return subprocess.check_output([str(args.git), "-C", str(args.repo), "--no-replace-objects", "show", commit + ":" + path])


raw = args.cases.read_bytes()
assert sha(raw) == CASE_HASH
data = strict_json(raw)
assert data["schema"] == "bert-agreement-suffix-cases-v1"
assert data["model"] == "distilbert/distilbert-base-uncased"
assert data["revision"] == "12040accade4e8a0f71eabdb258fecc2e7e948be"
assert data["targets"] == {"is": 2003, "are": 2024, "was": 2001, "were": 2020}
vocab_raw = args.vocab.read_bytes()
assert sha(vocab_raw) == VOCAB_HASH == data["tokenizer_asset_sha256"]["vocab.txt"]
vocab = vocab_raw.decode("utf-8").splitlines()
assert len(vocab) == 30522 and len(set(vocab)) == len(vocab)
ids_by_token = {word: i for i, word in enumerate(vocab)}
assert [vocab[i] for i in (101, 102, 103, 1012, 2003, 2024, 2001, 2020, 3201)] == [
    "[CLS]", "[SEP]", "[MASK]", ".", "is", "are", "was", "were", "ready"]

nouns = {word for pair in SUBJECTS for word in pair} | {word for _, _, pair in FRAMES for word in pair}
assert all(word in ids_by_token and not word.startswith("##") for word in nouns)
noun_ids = {ids_by_token[word] for word in nouns}
assert len(noun_ids) == len(nouns) == 12
frame_tokens = [[ids_by_token[token.lower()] for token in frame.split()] for frame, _, _ in FRAMES]
frame_strings = [frame.lower() for frame, _, _ in FRAMES]

rows = data["cases"]
assert len(rows) == 32
proposal = []
lookup = {}
for i, case in enumerate(rows):
    frame, attr, lemma, number = i // 16, (i // 8) % 2, (i // 2) % 4, i % 2
    for name, value in (("index", i), ("frame", frame), ("attractor_number", attr), ("lemma", lemma), ("number", number), ("same_number_donor_lemma", lemma ^ 1)):
        assert type(case[name]) is int and case[name] == value, (i, name)
    prefix, relation, attractors = FRAMES[frame]
    subject = SUBJECTS[lemma][number]
    assert case["case_id"] == f"f{frame}-a{attr}-l{lemma}-n{number}"
    assert case["subject"] == subject and case["attractor"] == attractors[attr]
    assert set(case["variants"]) == {"terminal", "continued"}
    for variant, suffix in (("terminal", ""), ("continued", " ready.")):
        row = case["variants"][variant]
        expected_text = f"{prefix}, the {subject} {relation} the {attractors[attr]} [MASK]{suffix}"
        assert row["text"] == expected_text
        pieces = re.findall(r"\[MASK\]|[A-Za-z]+|[,.]", expected_text)
        assert not re.sub(r"\[MASK\]|[A-Za-z]+|[,.]|\s+", "", expected_text)
        expected_ids = [101] + [ids_by_token[t if t == "[MASK]" else t.lower()] for t in pieces] + [102]
        assert row["input_ids"] == expected_ids and all(type(t) is int for t in row["input_ids"])
        assert row["attention_mask"] == [1] * len(expected_ids) and all(type(t) is int for t in row["attention_mask"])
        assert type(row["subject_position"]) is int and row["subject_position"] == 6
        assert type(row["output_position"]) is int and row["output_position"] == 10
        assert expected_ids[6] == ids_by_token[subject] and expected_ids[10] == 103
        proposal.append((i, variant, tuple(expected_ids)))
    terminal = case["variants"]["terminal"]["input_ids"]
    continued = case["variants"]["continued"]["input_ids"]
    assert terminal[:-1] + [3201, 1012, 102] == continued
    lookup[(frame, attr, lemma, number)] = case
assert len(lookup) == 32 and len(proposal) == 64
for case in rows:
    k = (case["frame"], case["attractor_number"], case["lemma"], case["number"])
    opposite = lookup[(k[0], k[1], k[2], 1 - k[3])]
    same = lookup[(k[0], k[1], k[2] ^ 1, k[3])]
    for variant in ("terminal", "continued"):
        target_ids = case["variants"][variant]["input_ids"]
        for donor in (opposite, same):
            donor_ids = donor["variants"][variant]["input_ids"]
            assert target_ids[6] != donor_ids[6]
            assert target_ids[:6] == donor_ids[:6] and target_ids[7:] == donor_ids[7:]


def prefix_collision(a, b):
    def prefix(x, y):
        return x[:len(y)] == y or y[:len(x)] == x
    return prefix(a, b) or prefix(a[:-1], b[:-1])


internal = [(a[0], a[1], b[0], b[1]) for a, b in itertools.combinations(proposal, 2) if prefix_collision(a[2], b[2])]
assert len(internal) == 32 and all(a == b and {x, y} == {"terminal", "continued"} for a, x, b, y in internal)


def token_arrays(obj):
    if isinstance(obj, dict):
        for val in obj.values():
            yield from token_arrays(val)
    elif isinstance(obj, list):
        if len(obj) >= 2 and all(type(x) is int for x in obj) and obj[0] == 101 and obj[-1] == 102:
            yield tuple(obj)
        else:
            for val in obj:
                yield from token_arrays(val)


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for val in obj.values():
            yield from strings(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from strings(val)


source_specs = []
old_report = git_bytes("763402e8311e2402c49eb865c1812c5a8b5380a2", "audits/cross_model_agreement_v1/bert-public-exposure-review.json")
assert sha(old_report) == OLD_REPORT_HASH
for s in strict_json(old_report)["sources"]:
    source_specs.append((s["path"], s["commit"], s["sha256"], s["rows"], None))
source_specs += [
    ("experiments/cross_model_agreement_cases_v1.json", "8e2cb4f815a12ffb8149f4afe6d3d073df9c170f", "b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503", 32, None),
    ("experiments/bert_color_preference_public_cases_v2.json", "f79dc502ff85b467b0682f4e8d168fbeb776cadf", "f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10", 37, None),
    ("pending/bert_cue_context_transfer_cases_v1.json", None, PENDING_HASH, 24, args.pending),
]
assert len(source_specs) == 24
word_re = re.compile(r"\b(?:" + "|".join(sorted(nouns, key=len, reverse=True)) + r")\b", re.I)
reports = []
all_prefix, all_token, all_phrase, all_text = [], [], [], []
for path, commit, expected_hash, expected_count, local in source_specs:
    source_raw = local.read_bytes() if local is not None else git_bytes(commit, path)
    assert sha(source_raw) == expected_hash, path
    obj = strict_json(source_raw)
    inputs = list(token_arrays(obj))
    if path.endswith("bert_negation_public_cases_v1.json"):
        cue_pos = obj["indices"]["cue"]
        expanded = []
        for row in inputs:
            assert row[cue_pos] == obj["cues"]["just"]
            for cue in ("not", "never", "just", "also"):
                row2 = list(row)
                row2[cue_pos] = obj["cues"][cue]
                expanded.append(tuple(row2))
        inputs = expanded
    assert len(inputs) == expected_count, (path, len(inputs), expected_count)
    pref = [(i, variant, j) for i, variant, row in proposal for j, old in enumerate(inputs) if prefix_collision(row, old)]
    token_hits = [(j, sorted(noun_ids.intersection(old))) for j, old in enumerate(inputs) if noun_ids.intersection(old)]
    phrase_hits = [(j, frame_strings[f]) for j, old in enumerate(inputs) for f, frame in enumerate(frame_tokens)
                   if any(list(old[k:k+len(frame)]) == frame for k in range(len(old) - len(frame) + 1))]
    text_hits = sorted({match.group(0).lower() for value in strings(obj) for match in word_re.finditer(value)})
    reports.append({"path": path, "commit": commit, "sha256": sha(source_raw), "row_references": len(inputs),
                    "prefix_collisions": len(pref), "noun_token_rows": len(token_hits),
                    "frame_token_rows": len(phrase_hits), "noun_text_hits": text_hits})
    all_prefix += [(path, *hit) for hit in pref]
    all_token += [(path, *hit) for hit in token_hits]
    all_phrase += [(path, *hit) for hit in phrase_hits]
    all_text += [(path, hit) for hit in text_hits]

report = {"schema": "bert-suffix-input-proposal-independent-v1", "case_sha256": CASE_HASH,
          "vocab_sha256": VOCAB_HASH, "cases": 32, "variants": 64,
          "expected_within_pair_prefix_relations": len(internal), "sources": reports,
          "source_row_references": sum(s["row_references"] for s in reports),
          "external_prefix_collisions": all_prefix, "noun_token_overlaps": all_token,
          "surface_frame_token_overlaps": all_phrase, "noun_string_hits": all_text,
          "scope": "Independent ASCII/vocab and named-source readback; no model or tokenizer library."
                   " Public inputs and reused references are nonblind; no unknown/private/pretraining or novel-syntax claim."}
with args.output.open("x", encoding="utf8", newline="\n") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
print("case SHA", CASE_HASH, "32 pairs; expected within-pair relations", len(internal))
print("24 sources / row refs", report["source_row_references"])
print("prior-row prefix / noun-token / frame-token / noun-string hits", *(len(x) for x in (all_prefix, all_token, all_phrase, all_text)))
if any((all_prefix, all_token, all_phrase, all_text)):
    print("overlap details", all_prefix, all_token, all_phrase, all_text)
print("report", args.output, "SHA256", sha(args.output.read_bytes()))
