"""Model-free stage tests using fabricated token rows and a tiny local Git repo."""

import json
import os
import subprocess
import sys

import pytest

import flan_multitoken_selector_stage_v1 as stage
import flan_multitoken_selector_v1 as core
from test_flan_multitoken_selector_v1 import synthetic_case, synthetic_reviews


def test_prefix_scan_checks_both_directions_and_preserves_invalid_rows():
    rows = [
        {"word": "aaaa", "subject": "story", "instruction": 0, "case": {"input_ids": [4, 5, 1]}},
        {"word": "bbbb", "subject": "story", "instruction": 0, "case": {"input_ids": [9, 8, 1]}},
        {"word": "cccc", "subject": "story", "instruction": 0, "case": {"input_ids": [7, 8]}},
    ]
    result = stage.scan_rows(rows, [[[4, 5, 6, 1], [4, 1], [2, 3, 1]]])
    assert result[0]["matches"] == [[0, 0], [0, 1]]
    assert result[0]["comparisons"] == 3 and result[0]["public_prefix_status"] == "collision"
    assert result[1]["public_prefix_status"] == "clear" and result[1]["comparisons"] == 3
    assert result[2]["public_prefix_status"] == "unknown" and result[2]["reason"]


def test_custodian_receipts_bind_every_row_and_preserve_nonclear_status():
    row = {"word": "aaaa", "subject": "story", "instruction": 0, "case": {"input_ids": [4, 5, 1]}}
    scan = {"rows": stage.scan_rows([row], [[[4, 5, 6, 1]]])}
    custody = {"schema": "flan-multisubtoken-custody-v1", "scope": "fabricated",
               "rows": [{"word": "aaaa", "subject": "story", "instruction": 0,
                         "input_ids_sha256": core.rows_digest([4, 5, 1]),
                         "status": "clear", "reason": ""}]}
    scope, public_sha, cust_sha, desc = stage.bind_scope([row], scan, core.canonical(custody))
    item = scope[("aaaa", "story", 0)]
    assert item["public_prefix_status"] == "collision" and item["reason"] == "public-prefix-collision"
    assert item["public_scope_sha256"] == public_sha and item["custodian_receipt_sha256"] == cust_sha
    assert desc == "fabricated"
    custody["rows"][0]["input_ids_sha256"] = "0" * 64
    with pytest.raises(stage.core.SelectionError, match="row hash"):
        stage.bind_scope([row], scan, core.canonical(custody))


def test_git_source_binding_ignores_ambient_git_overrides_and_rejects_tampering(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)
    git("init", "-q")
    blob = b'{"input_ids":[4,5,1]}\n'
    (repo / "fixture.json").write_bytes(blob)
    git("add", "fixture.json")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture")
    revision = git("rev-parse", "HEAD").decode().strip()
    descriptor = {"path": "fixture.json", "revision": revision, "sha256": core.sha256_bytes(blob)}
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing"))
    assert stage.read_git_blob(repo, descriptor) == blob
    bad = dict(descriptor, sha256="0" * 64)
    with pytest.raises(stage.core.SelectionError, match="source bytes"):
        stage.read_git_blob(repo, bad)
    bad_path = dict(descriptor, path="../fixture.json")
    with pytest.raises(stage.core.SelectionError, match="descriptor"):
        stage.read_git_blob(repo, bad_path)


def test_public_extractor_and_exclusive_output(tmp_path):
    source = b'{"input_ids":[4,5,1],"nested":{"input_ids":[6,1]}}\n'
    assert stage.extract_rows(source, False) == [[4, 5, 1], [6, 1]]
    with pytest.raises(stage.core.SelectionError, match="source token row"):
        stage.extract_rows(b'{"input_ids":[4,5]}', False)
    out = tmp_path / "output.json"
    first = stage.write_new(out, {"b": 2, "a": 1})
    assert first == b'{"a":1,"b":2}\n' and out.read_bytes() == first
    with pytest.raises(FileExistsError):
        stage.write_new(out, {"other": 3})


def test_direct_base_interpreter_is_required():
    script = os.path.abspath(stage.__file__)
    base = sys._base_executable
    direct = subprocess.run([base, "-I", "-S", "-B", script, "--help"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert direct.returncode == 0
    if not os.path.samefile(sys.executable, base):
        shim = subprocess.run([sys.executable, "-I", "-S", "-B", script, "--help"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert shim.returncode != 0 and b"not a virtual-environment shim" in shim.stderr


def test_paired_pool_inventory_and_commitments_are_checked():
    held_salt, senti_salt = b"h" * 32, b"s" * 32
    templates = [{"prefix": "A ", "suffix": " Z", "template_id": "a"},
                 {"prefix": "B ", "suffix": " Q", "template_id": "b"}]
    held_pool = {"templates": templates, "pairs": []}
    held_screen = {"candidates": []}
    token = 100
    def record(text):
        nonlocal token
        token += 1
        return {"text": text, "input_ids": [token, 1]}
    for i in range(24):
        pair = {"candidate_index": i, "source_words": [f"a{i}", f"b{i}"],
                "target_words": [f"c{i}", f"d{i}"]}
        held_pool["pairs"].append(pair)
        cases = []
        for template in templates:
            prefix, suffix = template["prefix"], template["suffix"]
            cases.append({"template_id": template["template_id"],
                          "neutral": record(prefix + suffix),
                          "source": [record(prefix + w + suffix) for w in pair["source_words"]],
                          "targets": [record(w) for w in pair["target_words"]]})
        held_screen["candidates"].append({"candidate_index": i, "cases": cases})
    held_raw = core.canonical(held_pool)
    held_commit = core.sha256_bytes(b"mlg-flant5-heldout-pool-v1\0" + held_salt + held_raw)
    held_screen["pool_commitment"] = held_commit
    held_card = core.canonical({"commitment": held_commit})

    senti_pool, senti_screen = {"units": []}, {"unit_screens": []}
    for i in range(12):
        unit = {"id": i, "subject": f"subject{i}",
                "adjectives": {role: f"word{role}{i}" for role in core.ROLES}}
        senti_pool["units"].append(unit)
        groups = []
        for instruction, prefix in enumerate(core.INSTRUCTIONS):
            cases = []
            for role in core.ROLES:
                item = record(f'{prefix} The {unit["subject"]} was {unit["adjectives"][role]}.')
                item["side"] = role
                cases.append(item)
            groups.append({"instruction_index": instruction, "cases": cases})
        senti_screen["unit_screens"].append({"unit_id": i, "instructions": groups})
    senti_raw = core.canonical(senti_pool)
    senti_commit = core.sha256_bytes(b"FLAN-SENTIMENT-HOLDOUT-V1\0" + senti_salt + b"\0" + senti_raw)
    senti_screen["pool_commitment"] = senti_commit
    senti_card = core.canonical({"commitment": senti_commit})
    args = [held_raw, core.canonical(held_screen), held_card, held_salt, senti_raw,
            core.canonical(senti_screen), senti_card, senti_salt.hex().encode() + b"\n"]
    result = stage.verify_prior_pool_coverage(*args)
    assert result["heldout_screen_rows"] == 240 and result["sentiment_screen_rows"] == 96
    held_screen["candidates"][0]["cases"][0]["source"][0]["text"] = "wrong"
    args[1] = core.canonical(held_screen)
    with pytest.raises(stage.core.SelectionError, match="heldout case text"):
        stage.verify_prior_pool_coverage(*args)
    args[1] = b"{}"
    with pytest.raises(stage.core.SelectionError, match="heldout pool commitment"):
        stage.verify_prior_pool_coverage(*args)


def test_full_fabricated_stage_flow_seals_discovery_before_painting(tmp_path, monkeypatch):
    def word(prefix, index):
        return prefix + chr(97 + index // 26) + chr(97 + index % 26) + "x"
    candidates = {"positive": [word("p", i) for i in range(42)],
                  "negative": [word("n", i) for i in range(26)]}
    lookup = {word: polarity for polarity in candidates for word in candidates[polarity]}
    case_map, pilot_rows = {}, []
    for polarity in ("positive", "negative"):
        one_count = 35 if polarity == "positive" else 12
        for index, item in enumerate(candidates[polarity]):
            count = 1 if index < one_count else 2
            token_id = 1000 + (300 if polarity == "negative" else 0) + 2 * index
            signatures, templates = [], []
            for subject in core.SUBJECTS:
                for instruction in (0, 1):
                    case = synthetic_case(item, subject, instruction, count, token_id)
                    parsed = core.inspect_case(case, item, subject, instruction)
                    signatures.append(parsed["signature"])
                    templates.append(dict(case, adjective_char_span=parsed["adjective_char_span"]))
                    case_map[(item, subject, instruction)] = case
            for instruction in (0, 1):
                case_map[(item, "painting", instruction)] = synthetic_case(item, "painting", instruction, count, token_id)
            pilot_rows.append({"word": item, "polarity_proposal": polarity, "order": index,
                               "structural_rejections": [], "public_overlap_reasons": [],
                               "fresh_for_scoped_public_sources": True, "signature": signatures,
                               "signature_sha256": core.sha256_bytes(core.canonical(signatures)),
                               "templates": templates})
    pilot = {"all_candidates": pilot_rows}
    monkeypatch.setattr(stage, "require_isolation", lambda: None)
    monkeypatch.setattr(stage, "verify_running_sources", lambda repo, revision:
                        {"commit": "f" * 40, "files_sha256": {"fixture": "a" * 64}})
    monkeypatch.setattr(stage, "_static", lambda args: (candidates, pilot, [], b"fabricated", b"fake-pool"))
    def retokenize(handoff, _, site):
        rows = [{"word": item["word"], "subject": subject, "instruction": instruction,
                 "case": case_map[(item["word"], subject, instruction)]}
                for item in handoff["items"] for subject in handoff["subjects"] for instruction in (0, 1)]
        return rows, {"python": "fabricated", "tokenizers": "fabricated"}
    monkeypatch.setattr(stage, "retokenize", retokenize)
    monkeypatch.setattr(stage, "public_scan", lambda rows, descriptors, additional, repo, sentiment_pool_raw:
                        {"schema": "fabricated-scan", "rows": stage.scan_rows(rows, [[[999, 1]]])})
    common = sum((["--" + field, "unused"] for field in
                  ("protocol", "candidates", "pilot", "tokenizer", "asset-manifest", "descriptors", "released-pool",
                   "repo", "source-commit")), [])
    extra = tmp_path / "extra.json"
    extra.write_text("[]", encoding="utf-8")
    def run(phase, output, extra_args=()):
        monkeypatch.setattr(sys, "argv", ["stage", phase, *common, "--output", str(output), *extra_args])
        stage.main()
    handoff_file = tmp_path / "handoff.json"
    run("handoff", handoff_file)
    handoff = core.parse_json(handoff_file.read_bytes())
    assert len(handoff["items"]) == 68
    review_file = tmp_path / "reviews.json"
    review_file.write_bytes(core.canonical(synthetic_reviews(handoff, lookup)))
    screen_file, scan_file = tmp_path / "screen.json", tmp_path / "scan.json"
    shared = ["--reviews", str(review_file), "--repo", "unused", "--additional-sources", str(extra)]
    run("discovery-screen", screen_file, [*shared, "--scan-output", str(scan_file)])
    screen = core.parse_json(screen_file.read_bytes())
    assert len(screen["rows"]) == 272
    def custody_for(screen, path):
        rows = [{"word": r["word"], "subject": r["subject"], "instruction": r["instruction"],
                 "input_ids_sha256": core.rows_digest(r["case"]["input_ids"]),
                 "status": "clear", "reason": ""} for r in screen["rows"]]
        path.write_bytes(core.canonical({"schema": "flan-multisubtoken-custody-v1", "scope": "fabricated", "rows": rows}))
    custody_file = tmp_path / "custody.json"
    custody_for(screen, custody_file)
    discovery_file = tmp_path / "discovery.json"
    run("discovery", discovery_file, [*shared, "--screen", str(screen_file), "--scan", str(scan_file), "--custody", str(custody_file)])
    discovery_raw = discovery_file.read_bytes()
    discovery = core.parse_json(discovery_raw)
    assert [s["status"] for s in discovery["slots"]] == ["selected"] * 4
    discovery_sha = core.sha256_bytes(discovery_raw)
    painting_handoff = tmp_path / "painting-handoff.json"
    seal = ["--discovery", str(discovery_file), "--discovery-sha256", discovery_sha]
    run("painting-handoff", painting_handoff, seal)
    paint_handoff = core.parse_json(painting_handoff.read_bytes())
    assert len(paint_handoff["items"]) == 8
    paint_review = tmp_path / "painting-reviews.json"
    paint_review.write_bytes(core.canonical(synthetic_reviews(paint_handoff, lookup)))
    paint_shared = ["--reviews", str(paint_review), "--repo", "unused", "--additional-sources", str(extra), *seal]
    paint_screen, paint_scan = tmp_path / "painting-screen.json", tmp_path / "painting-scan.json"
    run("painting-screen", paint_screen, [*paint_shared, "--scan-output", str(paint_scan)])
    paint_screen_data = core.parse_json(paint_screen.read_bytes())
    assert len(paint_screen_data["rows"]) == 16
    paint_custody = tmp_path / "painting-custody.json"
    custody_for(paint_screen_data, paint_custody)
    paint_selection = tmp_path / "painting-selection.json"
    run("painting", paint_selection, [*paint_shared, "--screen", str(paint_screen), "--scan", str(paint_scan), "--custody", str(paint_custody)])
    result = core.parse_json(paint_selection.read_bytes())
    assert result["discovery_sha256"] == discovery_sha
    assert [slot["status"] for slot in result["slots"]] == ["eligible", "eligible"]
    assert discovery_file.read_bytes() == discovery_raw
