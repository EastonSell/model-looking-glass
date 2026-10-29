"""Extract top-token changes from the fixed closed journal without a model."""
import argparse
import array
import base64
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

MANIFEST_SHA = "843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df"
RESULT_SHA = "0a788dfb0655ac8da55d94b1154fa240860850bf9464928b391d8cfa3327f62a"
JOURNAL_SHA = "a6c586dd9a8f125f60aa4003a8d8cbeac9cfae0f7862332412b5dde4a6417909"
VOCAB_SHA = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
SOURCE = "91f9960e419cf20df9df156a6ebc1980d1b6a864"
CUES = ("never", "rarely", "always", "often")
SITES = ("cue2", "object2", "mask2", "cue5")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_id(encoded):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 30522 * 4:
        raise ValueError("full float32 mask logits required")
    values = array.array("f")
    if values.itemsize != 4:
        raise ValueError("unexpected float width")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite mask logit")
    return max(range(len(values)), key=values.__getitem__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "result", "journal", "vocab", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    paths = (args.manifest, args.result, args.journal, args.vocab)
    expected = (MANIFEST_SHA, RESULT_SHA, JOURNAL_SHA, VOCAB_SHA)
    if tuple(sha(path) for path in paths) != expected:
        raise ValueError("fixed input hash differs")
    manifest, result = json.loads(args.manifest.read_bytes()), json.loads(args.result.read_bytes())
    if result.get("source_commit") != SOURCE or result.get("events_sha256") != JOURNAL_SHA:
        raise ValueError("result source/journal identity differs")
    vocabulary = args.vocab.read_text(encoding="utf-8").splitlines()
    if len(vocabulary) != 30522:
        raise ValueError("vocabulary size differs")
    splits = {pair["id"]: pair["split"] for pair in manifest["pairs"]}
    clean, seen, rows = {}, set(), []
    event_count = 0
    with gzip.open(args.journal, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            event_count += 1
            if event["kind"] == "clean":
                key = event["pair_id"], event["context"], event["cue"]
                if key in clean:
                    raise ValueError("duplicate clean record")
                clean[key] = top_id(event["mask_logits_f32le"])
            elif event["kind"] == "transfer":
                pair, site = event["pair_id"], event["site"]
                target, cue = event["target_context"], event["target_cue"]
                donor, donor_cue = event["donor_context"], event["donor_cue"]
                key = pair, site, target, cue, donor, donor_cue
                if key in seen or pair not in splits or site not in SITES or cue not in CUES or donor_cue not in CUES:
                    raise ValueError("duplicate or unknown transfer")
                seen.add(key)
                clean_id = clean[pair, target, cue]
                patched_id = top_id(event["mask_logits_f32le"])
                if donor == target:
                    category = "within_context_other"
                elif donor_cue == cue:
                    category = "same_cue_alternate_context"
                elif (cue in CUES[:2]) == (donor_cue in CUES[:2]):
                    category = "same_class_alternate_context"
                else:
                    category = "opposite_class_alternate_context"
                rows.append({"pair": pair, "split": splits[pair], "site": site,
                             "target_context": target, "target_cue": cue, "donor_context": donor,
                             "donor_cue": donor_cue, "category": category, "clean_id": clean_id,
                             "clean_token": vocabulary[clean_id], "patched_id": patched_id,
                             "patched_token": vocabulary[patched_id], "changed": clean_id != patched_id})
    if event_count != 1096 or len(clean) != 32 or len(rows) != 896 or len(seen) != 896:
        raise ValueError("saved record count differs")
    if tuple(sha(path) for path in paths) != expected:
        raise ValueError("fixed inputs changed during extraction")
    args.output.mkdir(parents=False, exist_ok=False)
    columns = ["pair", "split", "site", "target_context", "target_cue", "donor_context", "donor_cue",
               "category", "clean_id", "clean_token", "patched_id", "patched_token", "changed"]
    with (args.output / "top-tokens.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for split in ("discovery", "evaluation"):
        for site in SITES:
            group = [row for row in rows if row["split"] == split and row["site"] == site]
            summaries.append({"split": split, "site": site, "transfers": len(group),
                              "top_token_changes": sum(row["changed"] for row in group)})
    report = {"status": "passed", "events": event_count, "transfers": len(rows),
              "source_commit": SOURCE, "journal_sha256": JOURNAL_SHA, "summaries": summaries,
              "scope": "Top-token identities from saved full logits; not dose, landing or execution attestation."}
    (args.output / "top-summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
