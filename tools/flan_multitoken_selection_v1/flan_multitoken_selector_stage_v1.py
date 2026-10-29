"""Bounded model-free staging for the FLAN multi-subtoken selector.

Static assets, public Git blobs, and supplied custodian receipts are checked
before selection.  Receipts are cooperative observations, not proof of unseen
history or chronology.  No model framework or checkpoint weights are loaded.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import types

CORE_SHA256 = "999ef7ecc1a5ca423d41ad197d4bf406d554dfef21b2ada0c7088f4a27692019"
SOURCE_DIR = "tools/flan_multitoken_selection_v1"
CORE_PATH = Path(__file__).resolve().with_name("flan_multitoken_selector_v1.py")
with CORE_PATH.open("rb") as core_stream:
    CORE_BYTES = core_stream.read(16 * 1024 * 1024 + 1)
if len(CORE_BYTES) > 16 * 1024 * 1024 or hashlib.sha256(CORE_BYTES).hexdigest() != CORE_SHA256:
    raise RuntimeError("selector core bytes differ from the reviewed stage binding")
core = types.ModuleType("flan_multitoken_selector_bound_v1")
core.__file__ = str(CORE_PATH)
exec(compile(CORE_BYTES, str(CORE_PATH), "exec"), core.__dict__)


PRIOR_SELECTION = {
    "path": "results/flan-t5-sentiment-intervention-v1-20261029/custody/selection.json",
    "revision": "6b7433fba2c909c8f383565570bfc0844b77f2ad",
    "sha256": "bbea2226b0ddddf1d6733b400ec0b74204832810c99cba41f9465bbab7cd5bb2",
}
HELDOUT_CARD = {"path": "results/flan-t5-heldout-response-cpu-v1-20261029/custody/pool-commitment.json",
                "revision": "6fe4cc4e16ece717d672d8fc73fd3860e76ed459",
                "sha256": "05e2ab84f5dea02e4aac3d771ecf4e19fa5636d0594d743db81bc56cf5be38f5"}
HELDOUT_SALT = {"path": "results/flan-t5-heldout-response-cpu-v1-20261029/custody/pool-salt.bin",
                "revision": "6fe4cc4e16ece717d672d8fc73fd3860e76ed459",
                "sha256": "abe9b5736ed25cdea9e31f42f64b509cc8ed9c1f303749406dea4ca7e30a4cf2"}
SENTIMENT_CARD = {"path": "results/flan-t5-sentiment-intervention-v1-20261029/custody/pool-card.json",
                  "revision": "6b7433fba2c909c8f383565570bfc0844b77f2ad",
                  "sha256": "f6e174110bcca104c9ec0cf3ba0abce986dbf7f7e5aadcbcda089418b6419563"}
SENTIMENT_SALT = {"path": "results/flan-t5-sentiment-intervention-v1-20261029/custody/pool-salt.hex",
                  "revision": "6b7433fba2c909c8f383565570bfc0844b77f2ad",
                  "sha256": "33359592bbec2508eba903f2b1236bfb04a6808dea0b11e3da9aeaefca3a5cd1"}


HELDOUT_POOL_PATH = "results/flan-t5-heldout-response-cpu-v1-20261029/custody/pool.json"
HELDOUT_SCREEN_PATH = "results/flan-t5-heldout-response-cpu-v1-20261029/custody/screen.json"


def check_bytes(path, expected, *, compressed=False):
    raw = core.read_bounded(path, compressed=compressed)
    core.require(core.sha256_bytes(raw) == expected, f"asset hash mismatch: {path}")
    return raw


def write_new(path, value):
    raw = core.canonical(value) + b"\n"
    core.require(len(raw) <= core.MAX_JSON_BYTES, "output exceeds stage limit")
    with Path(path).open("xb") as stream:
        core.require(stream.write(raw) == len(raw), "short output write")
        stream.flush()
        os.fsync(stream.fileno())
    core.require(core.read_bounded(path) == raw, "saved output readback differs")
    print(f"{path}: sha256={core.sha256_bytes(raw)} bytes={len(raw)}")
    return raw


def _descriptor_ok(item):
    if not isinstance(item, dict) or set(item) != {"path", "revision", "sha256"}:
        return False
    path, revision, digest = item["path"], item["revision"], item["sha256"]
    return (isinstance(path, str) and 0 < len(path.encode("utf-8")) <= 256 and
            "\\" not in path and ":" not in path and not PurePosixPath(path).is_absolute() and
            all(part not in ("", ".", "..") for part in path.split("/")) and
            isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None and
            isinstance(digest, str) and core.HEX256.fullmatch(digest) is not None)


def read_git_blob(repo, descriptor):
    """Bind a public descriptor to a local commit and exact blob bytes."""
    core.require(_descriptor_ok(descriptor), "invalid public source descriptor")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
                "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"})
    command = ["git", "--no-replace-objects", "-C", str(repo)]
    try:
        kind = subprocess.check_output(command + ["cat-file", "-t", descriptor["revision"]],
                                       env=env, stderr=subprocess.DEVNULL, timeout=30).strip()
        core.require(kind == b"commit", "public source revision is not a commit")
        spec = descriptor["revision"] + ":" + descriptor["path"]
        size = subprocess.check_output(command + ["cat-file", "-s", spec], env=env,
                                       stderr=subprocess.DEVNULL, timeout=30).strip()
        core.require(size.isdigit() and int(size) <= core.MAX_JSON_BYTES, "public source size")
        raw = subprocess.check_output(command + ["cat-file", "blob", spec], env=env,
                                      stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise core.SelectionError("public Git source unavailable") from exc
    core.require(len(raw) == int(size) and core.sha256_bytes(raw) == descriptor["sha256"],
                 "public source bytes differ")
    return raw


def require_isolation():
    core.require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and
                 sys.flags.dont_write_bytecode == 1,
                 "launch with a direct Python interpreter and -I -S -B")
    try:
        direct_base = os.path.samefile(sys.executable, sys._base_executable)
    except (AttributeError, OSError):
        direct_base = False
    core.require(direct_base, "launch the direct base Python interpreter, not a virtual-environment shim")


def verify_running_sources(repo, revision):
    """Read back both executing source files from one specified Git commit."""
    core.require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                 "source commit format")
    stage_path = Path(__file__).resolve()
    stage_bytes = core.read_bounded(stage_path)
    source_hashes = {"flan_multitoken_selector_v1.py": CORE_SHA256,
                     "flan_multitoken_selector_stage_v1.py": core.sha256_bytes(stage_bytes)}
    for name, digest in source_hashes.items():
        read_git_blob(repo, {"path": SOURCE_DIR + "/" + name,
                             "revision": revision, "sha256": digest})
    return {"commit": revision, "files_sha256": source_hashes,
            "scope": "Cooperative saved-byte check, not execution attestation."}


def token_body(ids):
    core.require(isinstance(ids, list) and 1 < len(ids) and
                 all(type(x) is int and 0 <= x < 32128 for x in ids) and
                 ids[-1] == 1 and ids.count(1) == 1, "invalid source token row")
    return tuple(ids[:-1])


def extract_rows(raw, is_jsonl):
    """Collect recorded input_ids using the prior study's recursive convention."""
    payloads = ([core.parse_json(line) for line in raw.splitlines() if line]
                if is_jsonl else [core.parse_json(raw)])
    rows = []
    def walk(value):
        if isinstance(value, dict):
            if "input_ids" in value:
                token_body(value["input_ids"])
                rows.append(value["input_ids"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for payload in payloads:
        walk(payload)
    return rows


def verify_prior_pool_coverage(heldout_pool_raw, heldout_screen_raw, heldout_card_raw, heldout_salt,
                               sentiment_pool_raw, sentiment_screen_raw, sentiment_card_raw, sentiment_salt_hex):
    """Check published pool commitments and that paired screens cover their cases."""
    heldout_pool, heldout_screen = core.parse_json(heldout_pool_raw), core.parse_json(heldout_screen_raw)
    heldout_card = core.parse_json(heldout_card_raw)
    core.require(len(heldout_salt) == 32, "heldout pool salt size")
    heldout_canonical = json.dumps(heldout_pool, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    heldout_commit = core.sha256_bytes(b"mlg-flant5-heldout-pool-v1\0" + heldout_salt + heldout_canonical)
    core.require(heldout_card.get("commitment") == heldout_screen.get("pool_commitment") == heldout_commit,
                 "heldout pool commitment")
    pairs, candidates, templates = heldout_pool.get("pairs"), heldout_screen.get("candidates"), heldout_pool.get("templates")
    core.require(isinstance(pairs, list) and len(pairs) == 24 and isinstance(candidates, list) and
                 len(candidates) == 24 and isinstance(templates, list) and len(templates) == 2,
                 "heldout pool and screen sizes")
    checked_heldout = []
    for pair, candidate in zip(pairs, candidates):
        core.require(pair.get("candidate_index") == candidate.get("candidate_index") and
                     len(pair.get("source_words", [])) == len(pair.get("target_words", [])) == 2 and
                     len(candidate.get("cases", [])) == 2, "heldout candidate mapping")
        for template, case in zip(templates, candidate["cases"]):
            core.require(case.get("template_id") == template.get("template_id") and
                         len(case.get("source", [])) == len(case.get("targets", [])) == 2,
                         "heldout template mapping")
            prefix, suffix = template["prefix"], template["suffix"]
            core.require(case["neutral"]["text"] == prefix + suffix, "heldout neutral text")
            records = [case["neutral"]]
            for i in range(2):
                core.require(case["source"][i]["text"] == prefix + pair["source_words"][i] + suffix and
                             case["targets"][i]["text"] == pair["target_words"][i], "heldout case text")
                records.extend((case["source"][i], case["targets"][i]))
            for record in records:
                token_body(record["input_ids"])
                checked_heldout.append(tuple(record["input_ids"]))
    extracted_heldout = [tuple(x) for x in extract_rows(heldout_screen_raw, False)]
    core.require(len(checked_heldout) == len(extracted_heldout) == 240 and
                 Counter(checked_heldout) == Counter(extracted_heldout), "heldout token row coverage")

    sentiment_pool, sentiment_screen = core.parse_json(sentiment_pool_raw), core.parse_json(sentiment_screen_raw)
    sentiment_card = core.parse_json(sentiment_card_raw)
    core.require(len(sentiment_salt_hex) == 65 and sentiment_salt_hex.endswith(b"\n"), "sentiment salt encoding")
    try:
        sentiment_salt = bytes.fromhex(sentiment_salt_hex[:-1].decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise core.SelectionError("sentiment salt encoding") from exc
    core.require(len(sentiment_salt) == 32, "sentiment salt size")
    sentiment_commit = core.sha256_bytes(b"FLAN-SENTIMENT-HOLDOUT-V1\0" + sentiment_salt + b"\0" + sentiment_pool_raw)
    core.require(sentiment_card.get("commitment") == sentiment_screen.get("pool_commitment") == sentiment_commit,
                 "sentiment pool commitment")
    units, screens = sentiment_pool.get("units"), sentiment_screen.get("unit_screens")
    core.require(isinstance(units, list) and len(units) == 12 and isinstance(screens, list) and len(screens) == 12,
                 "sentiment pool and screen sizes")
    checked_sentiment = []
    for unit, screen in zip(units, screens):
        core.require(unit.get("id") == screen.get("unit_id") and len(screen.get("instructions", [])) == 2,
                     "sentiment unit mapping")
        for instruction, group in enumerate(screen["instructions"]):
            core.require(group.get("instruction_index") == instruction and len(group.get("cases", [])) == 4,
                         "sentiment instruction mapping")
            core.require({case.get("side") for case in group["cases"]} == set(core.ROLES), "sentiment roles")
            for case in group["cases"]:
                word = unit["adjectives"][case["side"]]
                expected = f'{core.INSTRUCTIONS[instruction]} The {unit["subject"]} was {word}.'
                core.require(case.get("text") == expected, "sentiment case text")
                token_body(case["input_ids"])
                checked_sentiment.append(tuple(case["input_ids"]))
    extracted_sentiment = [tuple(x) for x in extract_rows(sentiment_screen_raw, False)]
    core.require(len(checked_sentiment) == len(extracted_sentiment) == 96 and
                 Counter(checked_sentiment) == Counter(extracted_sentiment), "sentiment token row coverage")
    return {"heldout_pool_commitment": heldout_commit, "heldout_screen_rows": 240,
            "sentiment_pool_commitment": sentiment_commit, "sentiment_screen_rows": 96,
            "scope": "Published bytes and ordered text inventories; not tokenizer or chronology attestation."}


def scan_rows(case_rows, source_rows):
    """Compare EOS-stripped complete rows in both prefix directions."""
    result = []
    for row in case_rows:
        ids = row["case"]["input_ids"]
        matches = []
        comparisons = 0
        reason = ""
        try:
            left = token_body(ids)
            for source_index, rows in enumerate(source_rows):
                for occurrence, prior in enumerate(rows):
                    right = token_body(prior)
                    count = min(len(left), len(right))
                    comparisons += 1
                    if left[:count] == right[:count]:
                        matches.append([source_index, occurrence])
        except core.SelectionError as exc:
            reason = str(exc)
        result.append({"word": row["word"], "subject": row["subject"],
                       "instruction": row["instruction"], "input_ids_sha256": core.rows_digest(ids),
                       "public_prefix_status": "unknown" if reason else ("collision" if matches else "clear"),
                       "comparisons": comparisons, "matches": matches, "reason": reason})
    return result


def _static(args):
    protocol = check_bytes(args.protocol, core.PROTOCOL_SHA256)
    candidate = check_bytes(args.candidates, core.CANDIDATES_SHA256)
    pilot = check_bytes(args.pilot, core.PILOT_SHA256, compressed=True)
    tokenizer = check_bytes(args.tokenizer, core.TOKENIZER_SHA256)
    manifest = check_bytes(args.asset_manifest, core.ASSET_MANIFEST_SHA256)
    descriptors = check_bytes(args.descriptors, core.PRIOR_DESCRIPTORS_SHA256)
    pool = check_bytes(args.released_pool, core.RELEASED_POOL_SHA256)
    candidates, report = core.load_pinned_inventory(candidate, pilot)
    descriptor_list = core.parse_json(descriptors)
    core.require(isinstance(descriptor_list, list) and len(descriptor_list) == 10 and
                 all(_descriptor_ok(x) for x in descriptor_list), "prior source descriptors")
    core.require(len({(d["revision"], d["path"]) for d in descriptor_list}) == 10,
                 "duplicate prior source descriptor")
    return candidates, report, descriptor_list, tokenizer, pool


def retokenize(handoff, tokenizer_bytes, tokenizer_site):
    """Tokenize only the exact handoff words, using the pinned Rust backend."""
    core.require(isinstance(tokenizer_site, str), "tokenizer site path required")
    site = Path(tokenizer_site).resolve(strict=True)
    core.require(site.is_dir(), "tokenizer site path is not a directory")
    sys.path.insert(0, str(site))
    import tokenizers
    package_path = Path(tokenizers.__file__).resolve(strict=True)
    native_path = Path(tokenizers.tokenizers.__file__).resolve(strict=True)
    core.require(package_path.is_relative_to(site) and native_path.is_relative_to(site) and
                 tokenizers.__version__ == "0.22.2" and platform.python_version() == "3.11.15",
                 "tokenizers origin or runtime differs")
    runtime = {"python": platform.python_version(), "tokenizers": tokenizers.__version__,
               "package_sha256": core.sha256_bytes(core.read_bounded(package_path)),
               "native_sha256": core.sha256_bytes(core.read_bounded(native_path))}
    tokenizer = tokenizers.Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    rows = []
    for item in handoff["items"]:
        word = item["word"]
        for subject in handoff["subjects"]:
            for instruction in (0, 1):
                text = core.expected_text(word, subject, instruction)
                encoded = tokenizer.encode(text, add_special_tokens=True)
                case = {"text": text, "input_ids": list(encoded.ids),
                        "attention_mask": list(encoded.attention_mask),
                        "special_tokens_mask": list(encoded.special_tokens_mask),
                        "offset_mapping": [list(pair) for pair in encoded.offsets]}
                rows.append({"word": word, "subject": subject,
                             "instruction": instruction, "case": case})
    return rows, runtime


def public_scan(rows, descriptors, additional, repo, sentiment_pool_raw):
    sources = list(descriptors) + [PRIOR_SELECTION] + list(additional)
    core.require(len(sources) <= 64 and all(_descriptor_ok(d) for d in sources), "public source inventory")
    core.require(len({(d["revision"], d["path"]) for d in sources}) == len(sources),
                 "duplicate public source identity")
    raw_sources = [read_git_blob(repo, d) for d in sources]
    extracted = [extract_rows(raw, d["path"].endswith(".jsonl"))
                 for d, raw in zip(sources, raw_sources)]
    paths = [d["path"] for d in sources]
    core.require(paths.count(HELDOUT_POOL_PATH) == paths.count(HELDOUT_SCREEN_PATH) ==
                 paths.count(PRIOR_SELECTION["path"]) == 1, "paired pool sources")
    commitment_sources = [HELDOUT_CARD, HELDOUT_SALT, SENTIMENT_CARD, SENTIMENT_SALT]
    commitment_raw = [read_git_blob(repo, descriptor) for descriptor in commitment_sources]
    coverage = verify_prior_pool_coverage(raw_sources[paths.index(HELDOUT_POOL_PATH)],
                raw_sources[paths.index(HELDOUT_SCREEN_PATH)], commitment_raw[0], commitment_raw[1],
                sentiment_pool_raw, raw_sources[paths.index(PRIOR_SELECTION["path"])],
                commitment_raw[2], commitment_raw[3])
    return {"schema": "flan-multisubtoken-public-prefix-scan-v1",
            "prefix_method": "one-terminal-eos-stripped-bidirectional-prefix",
            "sources": sources, "source_row_counts": [len(r) for r in extracted],
            "zero_token_row_sources": [i for i, extracted_rows in enumerate(extracted) if not extracted_rows],
            "commitment_sources": commitment_sources, "paired_pool_coverage": coverage,
            "rows": scan_rows(rows, extracted),
            "scope": "Named Git sources only; zero-row sources add no token-prefix comparisons. "
                     "No claim about unknown or private exposures."}


def bind_scope(rows, scan, custody_raw):
    """Bind independent public and custodian statuses to every row hash."""
    custody = core.parse_json(custody_raw)
    core.require(isinstance(custody, dict) and custody.get("schema") == "flan-multisubtoken-custody-v1" and
                 isinstance(custody.get("scope"), str) and 0 < len(custody["scope"]) <= 4096 and
                 isinstance(custody.get("rows"), list), "custodian receipt")
    def keyed(values):
        result = {}
        for item in values:
            key = (item["word"], item["subject"], item["instruction"])
            core.require(key not in result, "duplicate scope row")
            result[key] = item
        return result
    pub = keyed(scan["rows"])
    cust = keyed(custody["rows"])
    expected = {(row["word"], row["subject"], row["instruction"]) for row in rows}
    core.require(len(expected) == len(rows) and set(pub) == expected and set(cust) == expected,
                 "scope row coverage")
    scan_sha = core.sha256_bytes(core.canonical(scan) + b"\n")
    cust_sha = core.sha256_bytes(custody_raw)
    scope = {}
    for row in rows:
        key = (row["word"], row["subject"], row["instruction"])
        input_sha = core.rows_digest(row["case"]["input_ids"])
        c, p = cust[key], pub[key]
        core.require(c.get("input_ids_sha256") == p.get("input_ids_sha256") == input_sha,
                     "scope receipt row hash")
        core.require(c.get("status") in ("clear", "collision", "unknown") and
                     isinstance(c.get("reason"), str) and len(c["reason"]) <= 2048,
                     "custodian row status")
        scope[key] = {"input_ids_sha256": input_sha, "public_scope_sha256": scan_sha,
                      "custodian_receipt_sha256": cust_sha,
                      "public_prefix_status": p["public_prefix_status"],
                      "custodian_status": c["status"],
                      "reason": c["reason"] or p["reason"] or
                                ("public-prefix-collision" if p["matches"] else "")}
    return scope, scan_sha, cust_sha, custody["scope"]


def provenance(scan_sha, custody_sha, custody_scope, runtime, sources):
    return {"protocol_sha256": core.PROTOCOL_SHA256, "candidates_sha256": core.CANDIDATES_SHA256,
            "pilot_sha256": core.PILOT_SHA256, "tokenizer_sha256": core.TOKENIZER_SHA256,
            "asset_manifest_sha256": core.ASSET_MANIFEST_SHA256,
            "prior_descriptors_sha256": core.PRIOR_DESCRIPTORS_SHA256,
            "released_pool_sha256": core.RELEASED_POOL_SHA256,
            "prefix_method": "one-terminal-eos-stripped-bidirectional-prefix",
            "public_scope_sha256": scan_sha, "custodian_receipt_sha256": custody_sha,
            "public_scope_description": "Pinned ten sources, released selection, and declared additional Git sources.",
            "custodian_scope_description": custody_scope,
            "tokenizer_runtime": core.canonical(runtime).decode("utf-8"), "selector_sources": sources}


def main():
    require_isolation()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("handoff", "discovery-screen", "discovery",
                                          "painting-handoff", "painting-screen", "painting"))
    for name in ("protocol", "candidates", "pilot", "tokenizer", "asset-manifest", "descriptors", "released-pool", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("reviews", "custody", "repo", "source-commit", "tokenizer-site",
                 "additional-sources", "scan-output", "screen", "scan", "discovery", "discovery-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    core.require(args.repo and args.source_commit, "reviewed source commit and repository required")
    source_identity = verify_running_sources(args.repo, args.source_commit)
    candidates, pilot, descriptors, tokenizer_bytes, sentiment_pool_raw = _static(args)
    prefiltered = core.prefilter(candidates, pilot)
    handoff = core.discovery_handoff(prefiltered)
    core.require(len(handoff["items"]) == 68, "pinned pilot prefilter count")
    if args.phase == "handoff":
        write_new(args.output, handoff)
        return
    if args.phase in ("painting-handoff", "painting-screen", "painting"):
        core.require(args.discovery and args.discovery_sha256, "sealed discovery and digest required")
        discovery_raw = core.read_bounded(args.discovery)
        discovery = core.load_sealed_discovery(discovery_raw, args.discovery_sha256)
        handoff = core.holdout_handoff(discovery)
        if args.phase == "painting-handoff":
            write_new(args.output, handoff)
            return
    core.require(args.reviews and args.repo and args.additional_sources,
                 "review, repo and additional sources required")
    reviews = core.parse_json(core.read_bounded(args.reviews))
    if args.phase in ("painting-screen", "painting"):
        core.validate_painting_reviews(discovery, handoff, reviews)
    else:
        core.validate_reviews(handoff, reviews)
    additional = core.parse_json(core.read_bounded(args.additional_sources))
    core.require(isinstance(additional, list), "additional source descriptor list")
    rows, runtime = retokenize(handoff, tokenizer_bytes, args.tokenizer_site)
    scan = public_scan(rows, descriptors, additional, args.repo, sentiment_pool_raw)
    screen = {"schema": "flan-multisubtoken-retokenized-v1",
              "tokenizer_sha256": core.TOKENIZER_SHA256, "runtime": runtime,
              "selector_sources": source_identity, "rows": rows}
    if args.phase in ("discovery-screen", "painting-screen"):
        core.require(args.scan_output, "scan output required")
        write_new(args.scan_output, scan)
        write_new(args.output, screen)
        return
    core.require(args.custody and args.screen and args.scan, "custody, screen and scan required")
    screen_raw = core.read_bounded(args.screen)
    scan_raw = core.read_bounded(args.scan)
    core.require(screen_raw == core.canonical(screen) + b"\n" and
                 scan_raw == core.canonical(scan) + b"\n", "saved screen or scan does not replay")
    custody_raw = core.read_bounded(args.custody)
    scope, scan_sha, custody_sha, custody_scope = bind_scope(rows, scan, custody_raw)
    prov = provenance(scan_sha, custody_sha, custody_scope, runtime, source_identity)
    cases = {(r["word"], r["subject"], r["instruction"]): r["case"] for r in rows}
    if args.phase == "discovery":
        result = core.select_slots(candidates, prefiltered, reviews, cases, scope, prov)
    else:
        result = core.assess_holdout(discovery_raw, args.discovery_sha256, reviews, cases, scope, prov)
    core.require(core.sha256_bytes(scan_raw) == scan_sha, "scan serialization")
    write_new(args.output, result)


if __name__ == "__main__":
    main()
