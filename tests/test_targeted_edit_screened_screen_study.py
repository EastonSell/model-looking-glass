"""Fabricated screen-owner boundaries; no checkpoint or public forward."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/targeted_edit_screened_screen_study.py"
spec = importlib.util.spec_from_file_location("screened_screen_owner_test", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def case_rows():
    return [{"case_id": f"case-{i}", "prompt": f"prompt {i}", "token_ids": [1, i + 1]}
            for i in range(44)]


def payload(rows):
    records = [{"record_id": "planned-" + row["case_id"], "category": "planned",
                "public_text": row["prompt"], "token_ids": row["token_ids"]} for row in rows]
    return {"version": "synthetic-v1", "records": records}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def test_screen_requires_all_exact_reservations_before_forward(monkeypatch):
    rows = case_rows()
    monkeypatch.setattr(study, "_case_bytes", lambda _checkout: json.dumps({"cases": rows}).encode())
    good = payload(rows)
    assert study._exposure(None, None, good, digest(good), ())["own_reserved_rows"] == 44
    missing = {"version": "synthetic-v1", "records": []}
    with pytest.raises(study.base.StudyError, match="44 exact"):
        study._exposure(None, None, missing, digest(missing), ())
    duplicated = payload(rows)
    duplicated["records"].append({"record_id": "other", "token_ids": rows[0]["token_ids"]})
    with pytest.raises(study.base.StudyError, match="prefix relation"):
        study._exposure(None, None, duplicated, digest(duplicated), ())


def test_runtime_and_nonpersistent_buffer_mismatch_reject_before_screen():
    runtime = {"threads": 1, "matmul": "highest"}
    manifest = {"discovery": {"runtime_settings": runtime, "buffer_state_sha256": "buffer"},
                "original_model_state_sha256": "state", "original_weight_sha256": "weight"}
    guard = SimpleNamespace(runtime_settings=runtime, state_sha="state", weight_sha="weight", buffer_sha="buffer")
    study._match_discovery_state(guard, runtime, manifest)
    calls = []
    runner = SimpleNamespace(screen=lambda *_args: calls.append("screen"))
    guard.buffer_sha = "changed"
    with pytest.raises(study.base.StudyError, match="state or runtime"):
        study._run_bound_screen(guard, runtime, manifest, runner, b"", {}, "unused")
    assert calls == []
    guard.buffer_sha = "buffer"
    guard.runtime_settings = {"threads": 1, "matmul": "medium"}
    with pytest.raises(study.base.StudyError, match="state or runtime"):
        study._run_bound_screen(guard, guard.runtime_settings, manifest, runner, b"", {}, "unused")
    assert calls == []


def write_output(attempt, source):
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
                 "development-review-attestation.json", "public-exposure.json", "public-exposure-attestation.json"):
        (attempt / name).write_bytes(b"{}")
    directory = attempt / "screen"
    directory.mkdir()
    (directory / "screen.f32").write_bytes(bytes(32 * 50257 * 4))
    (directory / "observations.jsonl").write_bytes(b"{}\n")
    (directory / "state-original.json").write_bytes(b"{}\n")
    artifacts = {}
    for name in ("screen.f32", "observations.jsonl", "state-original.json"):
        raw = (directory / name).read_bytes()
        artifacts[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    record = {"status": "private-screen", "case_sha256": study.base.CASES_SHA,
              "identity": {"source_commit": source, "manifest_sha256": "b" * 64,
                           "development_record_sha256": study.DISCOVERY_PHASE_SHA,
                           "candidate_sha256": study.CANDIDATE_SHA},
              "selection": {"rows": [{}] * 32, "selected_count": 0, "formal_target_eligible": False},
              "artifacts": artifacts}
    raw = study.base.canonical(record)
    (directory / "screen-selection.json").write_bytes(raw)
    return record, {"selection_sha256": hashlib.sha256(raw).hexdigest(), "manifest_sha256": "b" * 64,
                    "selected_count": 0, "formal_target_eligible": False}


def test_parent_checks_complete_screen_and_source_identity_even_if_rehashed(tmp_path):
    source = "a" * 40
    record, observed = write_output(tmp_path, source)
    inventory = study._phase_inventory(tmp_path, "screen", observed, source)
    assert len(inventory) == 10
    record["identity"]["source_commit"] = "f" * 40
    raw = study.base.canonical(record)
    (tmp_path / "screen/screen-selection.json").write_bytes(raw)
    observed["selection_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(study.base.StudyError, match="selection differs"):
        study._phase_inventory(tmp_path, "screen", observed, source)


def test_parent_rejects_missing_output_and_wrong_matrix_size(tmp_path):
    source = "a" * 40
    _record, observed = write_output(tmp_path, source)
    (tmp_path / "screen/state-original.json").unlink()
    with pytest.raises(study.base.StudyError, match="output set"):
        study._phase_inventory(tmp_path, "screen", observed, source)
    (tmp_path / "screen/state-original.json").write_bytes(b"{}\n")
    (tmp_path / "screen/screen.f32").write_bytes(b"bad")
    with pytest.raises(study.base.StudyError, match="matrix size"):
        study._phase_inventory(tmp_path, "screen", observed, source)
