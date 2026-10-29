"""No-model tests for the two-stage feasibility owner's authority boundaries."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_answer_format_owner_test", ROOT / "scripts/answer_format_feasibility_owner.py")
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@pytest.fixture
def payload():
    cases = json.loads((ROOT / owner.base.CASES).read_bytes())
    records = [{"record_id": "planned-" + row["case_id"], "category": "planned",
                "case_role": row["role"], "public_text": row["prompt"], "token_ids": row["token_ids"],
                "candidate_token_ids": [row["labels"][label]["token_id"] for label in owner.LABELS],
                "reference_commit": owner.CASES_COMMIT} for row in cases["cases"]]
    records.append({"record_id": "foreign", "category": "planned", "token_ids": [50256, 4, 3]})
    return cases, {"version": "fabricated", "records": records}


def test_public_payload_and_source_inventory(payload):
    cases, public = payload
    result = owner.check_public_payload(public, digest(public), cases, [[50256, 4, 3]])
    assert result["own_reserved_rows"] == 64
    assert result["other_row_comparisons"] == 64
    assert len(owner.base.EXTRA) == len(set(owner.base.EXTRA))
    assert all((ROOT / name).is_file() for name in owner.base.EXTRA)
    with pytest.raises(owner.base.StudyError, match="missing"):
        owner.check_public_payload(public, digest(public), cases, [[50256, 3, 4]])


@pytest.mark.parametrize("mutation", ["missing", "text", "role", "labels", "commit", "duplicate", "prefix"])
def test_rehashed_reservation_mutation_or_foreign_collision_rejects(payload, mutation):
    cases, public = payload
    if mutation == "missing":
        public["records"].pop(0)
    elif mutation == "text":
        public["records"][0]["public_text"] += "x"
    elif mutation == "role":
        public["records"][0]["case_role"] = "heldout"
    elif mutation == "labels":
        public["records"][0]["candidate_token_ids"][0] += 1
    elif mutation == "commit":
        public["records"][0]["reference_commit"] = "0" * 40
    elif mutation == "duplicate":
        row = dict(public["records"][0])
        row["record_id"] = "foreign-duplicate"
        public["records"].append(row)
    else:
        public["records"].append({"record_id": "foreign-prefix", "category": "planned",
                                  "token_ids": cases["cases"][0]["token_ids"][:-1]})
    with pytest.raises(owner.base.StudyError):
        owner.check_public_payload(public, digest(public), cases)


def test_published_source_and_original_commitment_chain():
    raw, reference, prior = owner._fixed(ROOT)
    assert owner.base.digest(raw) == owner.base.CASES_SHA
    assert prior["model_state_sha256"] == reference["model_state_sha256"]
    assert prior["buffer_state_sha256"] == "f12cbc9f0cdf2fcb9567418d7d1129da0dd6db4f6c7aaa7f9edb32b3fe7668e8"


@pytest.mark.parametrize("mutation", ["none", "unbound", "wrong_commit", "missing_reference", "wrong_hash", "heldout", "duplicate"])
def test_only_evidence_bound_development_observations_are_exempt(payload, mutation):
    cases, public = payload
    binding = {"commit": "a" * 40, "references": ["cases.json", "phase.json", "matrix.f32"],
               "hashes": ["b" * 64, "c" * 64, "d" * 64]}
    index = 32 if mutation == "heldout" else 0
    record = public["records"][index]
    record["record_id"] = record["record_id"].replace("planned-", "observed-", 1)
    record["category"] = "observed-saved"
    record["reference_commit"] = binding["commit"]
    record["references"] = list(binding["references"])
    record["provenance"] = " ".join(binding["hashes"])
    if mutation == "wrong_commit":
        record["reference_commit"] = "0" * 40
    elif mutation == "missing_reference":
        record["references"].pop()
    elif mutation == "wrong_hash":
        record["provenance"] = " ".join(binding["hashes"][:-1])
    elif mutation == "duplicate":
        duplicate = dict(record)
        duplicate["record_id"] = duplicate["record_id"].replace("observed-", "planned-", 1)
        public["records"].append(duplicate)
    if mutation == "none":
        assert owner.check_public_payload(public, digest(public), cases, observed_binding=binding)["own_reserved_rows"] == 64
    else:
        with pytest.raises(owner.base.StudyError):
            owner.check_public_payload(public, digest(public), cases,
                                       observed_binding=None if mutation == "unbound" else binding)


@pytest.mark.parametrize("changed", ["buffer", "runtime"])
def test_state_mismatch_calls_no_phase_writer(changed, tmp_path):
    runtime = {"cpu": True}
    guard = SimpleNamespace(runtime_settings=runtime.copy(), state_sha="a" * 64,
                            weight_sha="b" * 64, buffer_sha="c" * 64)
    manifest = {"baseline_runtime_settings": runtime.copy(), "original_model_state_sha256": guard.state_sha,
                "original_weight_sha256": guard.weight_sha, "baseline_buffer_state_sha256": guard.buffer_sha}
    calls = []
    runner = SimpleNamespace(run=lambda *a, **kw: calls.append((a, kw)))
    if changed == "buffer":
        guard.buffer_sha = "f" * 64
    else:
        guard.runtime_settings = {"cpu": False}
    with pytest.raises(owner.base.StudyError, match="buffers or runtime"):
        owner._run_bound_phase(guard, runtime, manifest, runner, b"x", {}, "development", tmp_path)
    assert calls == []


def test_custody_requires_the_bound_natural_frame_and_scoped_absence(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"custodian": "natural", "reported_by": "fabricated", "observed_utc": now,
             "status": "public_frame_clear", "case_sha256": owner.base.CASES_SHA,
             "frame_sha256": owner.SUPPORT_SHA, "report_sha256": owner.FRAME_SHA,
             "commitment_sha256": owner.NAT_COMMITMENT_SHA, "original_split_unchanged": True,
             "no_additional_relevant_splits_reported": True, "successor_status": "no_successor_split_reported"},
            {"custodian": "injection", "reported_by": "fabricated", "observed_utc": now,
             "status": "no_protected_rows_reported"}]
    path = tmp_path / "custody.json"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    assert owner.check_custody(path)["observations"] == rows
    rows[0]["successor_status"] = "unknown"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    with pytest.raises(owner.base.StudyError, match="successor"):
        owner.check_custody(path)


@pytest.mark.parametrize("mutation", ["none", "stale", "wrong_version", "wrong_hash", "unknown_inputs", "later_tree"])
def test_latest_inventory_requires_fresh_scoped_public_tree(monkeypatch, tmp_path, mutation):
    source = tmp_path / "looking_glass/pretrained_exposures_v28.py"
    source.parent.mkdir()
    source.write_text("synthetic", encoding="utf-8")
    now = datetime.now(timezone.utc)
    report = {"observed_utc": now.isoformat(), "reviewed_commit": "a" * 40,
              "latest_main_commit": "b" * 40, "latest_inventory_version": "distilgpt2-development-exposures-v28",
              "latest_inventory_sha256": "c" * 64, "no_unreserved_relevant_inputs_reported": True,
              "currentness_reported_by": "fabricated", "source_path": "looking_glass/pretrained_exposures_v28.py"}
    if mutation == "stale":
        report["observed_utc"] = (now - timedelta(minutes=16)).isoformat()
    elif mutation == "wrong_version":
        report["latest_inventory_version"] = "distilgpt2-development-exposures-v27"
    elif mutation == "wrong_hash":
        report["latest_inventory_sha256"] = "0" * 64
    elif mutation == "unknown_inputs":
        report["no_unreserved_relevant_inputs_reported"] = False
    tree = b"looking_glass/pretrained_exposures_v28.py\n"
    if mutation == "later_tree":
        tree += b"looking_glass/pretrained_exposures_v29.py\n"
    launch = SimpleNamespace(_git=lambda *args: tree, _checked_path=lambda p: Path(p).resolve(strict=True))
    monkeypatch.setattr(owner.base, "commit_object", lambda *a: None)
    monkeypatch.setattr(owner.base, "committed_file", lambda *a: None)
    if mutation == "none":
        owner._check_latest_inventory(launch, tmp_path, report, "distilgpt2-development-exposures-v28", "c" * 64)
    else:
        with pytest.raises(owner.base.StudyError):
            owner._check_latest_inventory(launch, tmp_path, report, "distilgpt2-development-exposures-v28", "c" * 64)


@pytest.mark.parametrize("expired", ["none", "inventory", "custody"])
def test_launch_rechecks_saved_observations_and_expiry_calls_no_child(monkeypatch, tmp_path, expired):
    checkout, attempt = tmp_path / "checkout", tmp_path / "attempt"
    source = checkout / "looking_glass/pretrained_exposures_v28.py"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic", encoding="utf-8")
    attempt.mkdir()
    now = datetime.now(timezone.utc)
    old = (now - timedelta(minutes=15)).isoformat()
    current = now.isoformat()
    report = {"observed_utc": old if expired == "inventory" else current, "reviewed_commit": "a" * 40,
              "latest_main_commit": "b" * 40, "latest_inventory_version": "distilgpt2-development-exposures-v28",
              "latest_inventory_sha256": "c" * 64, "no_unreserved_relevant_inputs_reported": True,
              "currentness_reported_by": "fabricated", "source_path": "looking_glass/pretrained_exposures_v28.py"}
    timestamp = old if expired == "custody" else current
    custody = {"observations": [{"custodian": "natural", "reported_by": "fabricated", "observed_utc": timestamp,
               "status": "public_frame_clear", "case_sha256": owner.base.CASES_SHA,
               "frame_sha256": owner.SUPPORT_SHA, "report_sha256": owner.FRAME_SHA,
               "commitment_sha256": owner.NAT_COMMITMENT_SHA, "original_split_unchanged": True,
               "no_additional_relevant_splits_reported": True, "successor_status": "no_successor_split_reported"},
              {"custodian": "injection", "reported_by": "fabricated", "observed_utc": timestamp,
               "status": "no_protected_rows_reported"}]}
    report_raw, custody_raw = owner.base.canonical(report), owner.base.canonical(custody)
    (attempt / "public-exposure-attestation.json").write_bytes(report_raw)
    (attempt / "custody-observation.json").write_bytes(custody_raw)
    launch = SimpleNamespace(_git=lambda *args: b"looking_glass/pretrained_exposures_v28.py\n",
                             _checked_path=lambda p: Path(p).resolve(strict=True))
    monkeypatch.setattr(owner.base, "commit_object", lambda *a: None)
    monkeypatch.setattr(owner.base, "committed_file", lambda *a: None)
    calls = []
    monkeypatch.setattr(owner.base, "supervise_child", lambda *a, **kw: calls.append(True) or "launched")
    args = (launch, checkout, attempt, owner.base.digest(custody_raw), owner.base.digest(report_raw),
            "distilgpt2-development-exposures-v28", "c" * 64, None, [], tmp_path)
    if expired == "none":
        assert owner._supervise_checked(*args, env={}, job=None) == "launched"
        assert calls == [True]
    else:
        with pytest.raises(owner.base.StudyError):
            owner._supervise_checked(*args, env={}, job=None)
        assert calls == []


def test_abbreviated_child_phase_rejected():
    with pytest.raises(SystemExit):
        owner.main(["child", "--checkout", str(ROOT), "--venv", str(ROOT), "--dependency", str(ROOT),
                    "--work", str(ROOT), "--pha=heldout"])


@pytest.mark.parametrize("mutation", ["none", "review", "unclean", "audit", "matrix", "result"])
def test_development_handoff_fences_inconsistent_review_before_heldout(monkeypatch, tmp_path, mutation):
    checkout = tmp_path / "checkout"
    directory = checkout / owner.DEVELOPMENT_DIR
    directory.mkdir(parents=True)
    parent = directory.parent
    now = datetime.now(timezone.utc).isoformat()
    manifest_sha, matrix_sha, result_sha = "a" * 64, "b" * 64, "c" * 64
    source, commit = "d" * 40, "e" * 40
    manifest = {"source_commit": source, "original_model_state_sha256": "f" * 64,
                "original_weight_sha256": "1" * 64, "baseline_buffer_state_sha256": "2" * 64,
                "baseline_runtime_settings": {"cpu": True}, "checkpoint": {"asset_sha256": {}},
                "source_sha256": {}, "created_utc": now}
    child = {"phase": "development", "manifest_sha256": manifest_sha, "model_disposed": True,
             "no_model_loaded": False, "owner_pid": 42, "owner_run_id": "3" * 32,
             "model_state_sha256": "f" * 64, "weight_sha256": "1" * 64,
             "buffer_state_sha256": "2" * 64, "runtime_settings": {"cpu": True},
             "post_work_checkpoint_assets": {}, "imported_sources": {}, "finished_utc": now}
    child_raw = owner.base.canonical(child)
    extra_artifacts = {"custody-observation.json": {"sha256": "6" * 64},
                       "manifest-review-attestation.json": {"sha256": "7" * 64},
                       "public-exposure.json": {"sha256": "8" * 64},
                       "public-exposure-attestation.json": {"sha256": "9" * 64}}
    started_raw = owner.base.canonical({"started_utc": now, "phase": "development", "source_commit": source,
        "manifest_sha256": manifest_sha, "custody_observation_sha256": "6" * 64,
        "manifest_review_sha256": "7" * 64, "public_exposure_file_sha256": "8" * 64,
        "public_exposure_attestation_sha256": "9" * 64})
    (directory / "child-result.json").write_bytes(child_raw)
    (directory / "started.json").write_bytes(started_raw)
    artifacts = {**extra_artifacts, "child-result.json": {"sha256": owner.base.digest(child_raw)},
                 "started.json": {"sha256": owner.base.digest(started_raw)},
                 "development/matrix.f32": {"sha256": matrix_sha},
                 "development/results.json": {"sha256": result_sha}}
    record = {"version": "answer-format-phase-v1", "status": "complete", "phase": "development",
              "source_commit": source, "manifest_sha256": manifest_sha, "owner_reaped": True,
              "job_drained": True, "child": child, "supervisor": {"clean": True, "exit_code": 0,
              "timed_out": False, "pid": 42, "run_id": "3" * 32}, "artifacts": artifacts,
              "child_result_sha256": owner.base.digest(child_raw),
              "started_sha256": owner.base.digest(started_raw), "finished_utc": now}
    if mutation == "unclean":
        record["supervisor"]["clean"] = False
    record_raw = owner.base.canonical(record)
    record_path = directory / "phase-record.json"
    record_path.write_bytes(record_raw)
    audit_doc = {"schema": "independent-answer-format-owner-comparison-v1", "status": "passed",
                 "heldout": None, "development": {"matrix_sha256": matrix_sha,
                 "result_sha256": result_sha, "journal_rows": 32}}
    if mutation == "audit":
        audit_doc["status"] = "failed"
    audit_path = parent / "independent-audit.json"
    audit_raw = owner.base.canonical(audit_doc)
    audit_path.write_bytes(audit_raw)
    review = {"status": "passed", "phase_sha256": owner.base.digest(record_raw),
              "manifest_sha256": manifest_sha, "case_sha256": owner.base.CASES_SHA,
              "reviewer": "fabricated", "reviewed_commit": commit, "observed_utc": now,
              "matrix_sha256": matrix_sha, "result_sha256": result_sha,
              "audit_path": audit_path.relative_to(checkout).as_posix(),
              "audit_sha256": owner.base.digest(audit_raw)}
    if mutation == "review":
        review["status"] = "failed"
    elif mutation == "matrix":
        review["matrix_sha256"] = "4" * 64
    elif mutation == "result":
        review["result_sha256"] = "5" * 64
    for name in ("check_phase_ancestry", "committed_file", "check_committed_inventory", "check_exact_phase_files"):
        monkeypatch.setattr(owner.base, name, lambda *a, **k: None)
    monkeypatch.setattr(owner.base, "windows_path_absent", lambda *a: True)
    monkeypatch.setattr(owner, "_imports", lambda *a: None)
    monkeypatch.setattr(owner, "_inventory", lambda *a: artifacts)
    launch = SimpleNamespace(_checked_path=lambda p: Path(p).resolve(strict=True))
    if mutation == "none":
        assert owner._check_development(launch, checkout, commit, manifest, manifest_sha, commit,
                                        record_path, owner.base.digest(record_raw), review) == (directory, matrix_sha, result_sha)
    else:
        with pytest.raises(owner.base.StudyError):
            owner._check_development(launch, checkout, commit, manifest, manifest_sha, commit,
                                     record_path, owner.base.digest(record_raw), review)
