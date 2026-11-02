"""Reproduce the outcome-informed tokenizer-only pilot without model weights."""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import tokenizers


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--candidates", type=Path, required=True)
PARSER.add_argument("--asset-dir", type=Path, required=True)
PARSER.add_argument("--repo", type=Path, required=True)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--compare", type=Path, help="Optional original .json or .json.gz report")
ARGS = PARSER.parse_args()
CANDIDATES = ARGS.candidates.resolve()
OUTPUT = ARGS.output.resolve()
REPO = ARGS.repo.resolve()
ASSETS = ARGS.asset_dir.resolve()
SOURCE = "461a6d44f180e4cf924a0ad47e3f624a9d5191ad"
EXPECTED_LIST_SHA = "2b32da31aab398d19ef80e013c377d2ed3d1a06912a431ee918bfeccb64107d1"
EXPECTED_ASSET_MANIFEST_SHA = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
EXPECTED_SOURCE_SHA = {
    "results/flan-t5-sentiment-intervention-v1-20261029/custody/pool.json": "de4c13ea29ecb0f58eb36a6a6c171abf1488e5036fd0c2fe567b13d34692b93e",
    "docs/FLAN_T5_SENTIMENT_SCREEN_V1.md": "78898528c8dafb4016210ffa0914d9169eb58a02b0f54d0206cffd2622442bff",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(path):
    # Pin exact source bytes even if local Git state has been altered. Ignore
    # ambient Git overrides and replacement objects, and prohibit lazy fetch.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
                "GIT_TERMINAL_PROMPT": "0"})
    raw = subprocess.check_output(["git", "--no-replace-objects", "-C", str(REPO),
                                   "cat-file", "blob", f"{SOURCE}:{path}"], env=env)
    if sha(raw) != EXPECTED_SOURCE_SHA[path]:
        raise ValueError(f"published source hash differs: {path}")
    return raw


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def main():
    raw_candidates = CANDIDATES.read_bytes()
    require(sha(raw_candidates) == EXPECTED_LIST_SHA, "candidate list hash differs")
    candidates = json.loads(raw_candidates)
    require(candidates["schema"] == "flan-multitoken-tokenizer-feasibility-candidates-v1", "candidate schema differs")
    words = candidates["positive"] + candidates["negative"]
    require(len(candidates["positive"]) == len(candidates["negative"]) == 80, "candidate count differs")
    require(len(set(words)) == len(words) and all(re.fullmatch(r"[a-z]{1,24}", w) for w in words), "duplicate or malformed candidate")
    require(len(candidates["templates"]) == 4, "template count differs")
    require(all(t.count("{word}") == 1 for t in candidates["templates"]), "malformed template")

    asset_manifest_bytes = (ASSETS / "asset-manifest.json").read_bytes()
    require(sha(asset_manifest_bytes) == EXPECTED_ASSET_MANIFEST_SHA, "asset manifest hash differs")
    asset_manifest = json.loads(asset_manifest_bytes)
    tokenizer_bytes = (ASSETS / "tokenizer.json").read_bytes()
    require(sha(tokenizer_bytes) == asset_manifest["files"]["tokenizer.json"]["sha256"], "tokenizer hash differs")
    tokenizer = tokenizers.Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
    # No torch, transformers model, model.safetensors or GPU API is loaded here.

    released_pool_path = "results/flan-t5-sentiment-intervention-v1-20261029/custody/pool.json"
    released_pool_bytes = git_blob(released_pool_path)
    released_pool = json.loads(released_pool_bytes)
    previously_revealed = {w for unit in released_pool["units"] for w in unit["adjectives"].values()}
    development_path = "docs/FLAN_T5_SENTIMENT_SCREEN_V1.md"
    development_bytes = git_blob(development_path)
    previous_pairs = re.findall(r"^\d+\. `([a-z]+)` / `([a-z]+)`$", development_bytes.decode("utf-8"), re.M)
    require(len(previous_pairs) == 4, "published development pair list differs")
    previous_development = {word for pair in previous_pairs for word in pair}

    rows = []
    buckets = collections.defaultdict(lambda: {"positive": [], "negative": []})
    for polarity in ("positive", "negative"):
        for order, word in enumerate(candidates[polarity]):
            overlap = []
            if word in previously_revealed:
                overlap.append("released-sentiment-selection-pool")
            if word in previous_development:
                overlap.append("published-sentiment-development-model-screen")
            row = {"polarity_proposal": polarity, "order": order, "word": word,
                   "public_overlap_reasons": overlap, "semantic_review": "not-performed",
                   "templates": [], "structural_rejections": []}
            signature = []
            for template_index, template in enumerate(candidates["templates"]):
                start = template.index("{word}")
                text = template.replace("{word}", word)
                end = start + len(word)
                encoded = tokenizer.encode(text, add_special_tokens=True)
                ids = list(encoded.ids)
                offsets = [list(pair) for pair in encoded.offsets]
                attention = list(encoded.attention_mask)
                special = list(encoded.special_tokens_mask)
                whole = [i for i, (a, b) in enumerate(offsets) if a < end and b > start - 1]
                lexical = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
                errors = []
                if text[start:end] != word or start < 1 or text[start - 1] != " " or text[end:] != ".":
                    errors.append("text-boundary")
                if len(ids) > 64 or not ids or ids[-1] != 1 or ids.count(1) != 1:
                    errors.append("length-or-eos")
                if not whole or not lexical:
                    errors.append("missing-word-span")
                else:
                    if whole != list(range(whole[0], whole[-1] + 1)) or lexical[-1] != whole[-1]:
                        errors.append("noncontiguous-span")
                    if any(attention[i] != 1 or special[i] != 0 or ids[i] == 2 for i in whole):
                        errors.append("unattended-special-or-unknown")
                    if any(not (start - 1 <= offsets[i][0] < offsets[i][1] <= end) for i in whole):
                        errors.append("offset-crosses-boundary")
                    covered = {k for i in whole for k in range(offsets[i][0], offsets[i][1])}
                    if covered != set(range(start - 1, end)):
                        errors.append("incomplete-character-coverage")
                record = {"template_index": template_index, "text": text, "input_ids": ids,
                          "attention_mask": attention, "special_tokens_mask": special,
                          "offset_mapping": offsets, "adjective_char_span": [start, end],
                          "word_plus_space_token_indices": whole, "lexical_token_indices": lexical,
                          "last_lexical_index": lexical[-1] if lexical else None,
                          "standalone_leading_space": bool(whole and offsets[whole[0]] == [start - 1, start]),
                          "structural_rejections": errors}
                row["templates"].append(record)
                row["structural_rejections"].extend(f"template-{template_index}:{error}" for error in errors)
                if not errors:
                    left, right = whole[0], whole[-1] + 1
                    signature.append({"lexical_count": len(lexical), "word_plus_space_count": len(whole),
                                      "last_lexical_index": lexical[-1],
                                      "prefix_ids": ids[:left], "suffix_ids": ids[right:],
                                      "prefix_attention": attention[:left], "suffix_attention": attention[right:],
                                      "prefix_special": special[:left], "suffix_special": special[right:]})
            if not row["structural_rejections"]:
                bucket_id = sha(canonical(signature))
                row["signature_sha256"] = bucket_id
                row["signature"] = signature
                if not overlap:
                    buckets[bucket_id][polarity].append(word)
            else:
                row["signature_sha256"] = None
            row["fresh_for_scoped_public_sources"] = not overlap
            rows.append(row)

    summaries = []
    for bucket_id, values in buckets.items():
        representative = next(row for row in rows if row["signature_sha256"] == bucket_id)
        signature = representative["signature"]
        summaries.append({"signature_sha256": bucket_id,
                          "lexical_counts": [s["lexical_count"] for s in signature],
                          "word_plus_space_counts": [s["word_plus_space_count"] for s in signature],
                          "last_lexical_indices": [s["last_lexical_index"] for s in signature],
                          "positive_count": len(values["positive"]), "negative_count": len(values["negative"]),
                          "potential_disjoint_four_word_units": min(len(values["positive"]) // 2, len(values["negative"]) // 2),
                          "illustrative_positive": values["positive"][:2],
                          "illustrative_negative": values["negative"][:2]})
    summaries.sort(key=lambda x: (x["lexical_counts"], x["word_plus_space_counts"], x["signature_sha256"]))
    lexical_counts = {polarity: dict(sorted(collections.Counter(
        str(row["templates"][0]["lexical_token_indices"].__len__()) for row in rows
        if row["polarity_proposal"] == polarity and not row["structural_rejections"]).items()))
                      for polarity in ("positive", "negative")}
    report = {
        "schema": "flan-multitoken-tokenizer-feasibility-report-v1",
        "scope": "outcome-informed development tokenization only; no pretrained model loaded or forwarded",
        "source_sha256": sha(Path(__file__).read_bytes()), "candidate_sha256": sha(raw_candidates),
        "asset_manifest_sha256": sha(asset_manifest_bytes), "tokenizer_sha256": sha(tokenizer_bytes),
        "released_pool_git_commit": SOURCE, "released_pool_sha256": sha(released_pool_bytes),
        "development_screen_path": development_path, "development_screen_sha256": sha(development_bytes),
        "runtime": {"python": sys.version, "tokenizers": tokenizers.__version__},
        "previously_revealed_word_count": len(previously_revealed),
        "previous_development_words": sorted(previous_development),
        "candidates": len(rows), "tokenized_rows": sum(len(row["templates"]) for row in rows),
        "structural_failure_count": sum(bool(row["structural_rejections"]) for row in rows),
        "overlap_count": sum(not row["fresh_for_scoped_public_sources"] for row in rows),
        "lexical_count_by_proposed_polarity": lexical_counts,
        "signature_buckets_after_scoped_exclusions": summaries, "all_candidates": rows,
    }
    if ARGS.compare is not None:
        compare = ARGS.compare.resolve()
        payload = compare.read_bytes()
        expected = json.loads(gzip.decompress(payload) if compare.suffix == ".gz" else payload)
        recorded = dict(expected)
        replayed = dict(report)
        for key in ("source_sha256", "runtime"):
            recorded.pop(key)
            replayed.pop(key)
        if replayed != recorded:
            raise ValueError("pilot scientific fields differ from the original report")
        print("comparison", "all scientific fields match; source and runtime metadata excluded")
    output_bytes = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    with OUTPUT.open("xb") as stream:
        stream.write(output_bytes)
    print("source_sha256", report["source_sha256"])
    print("report_sha256", sha(OUTPUT.read_bytes()))
    print("candidates", report["candidates"], "tokenized_rows", report["tokenized_rows"],
          "structural_failures", report["structural_failure_count"], "old_source_overlaps", report["overlap_count"])
    print("lexical_counts", json.dumps(lexical_counts, sort_keys=True))
    for summary in summaries:
        print("bucket", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
