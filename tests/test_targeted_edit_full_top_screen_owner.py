"""Fabricated owner-boundary tests; no checkpoint or study prompt forward."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_full_top_screen_owner_test", ROOT / "scripts/targeted_edit_full_top_screen_owner.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@pytest.fixture
def public():
    cases = json.loads((ROOT / study.base.CASES).read_bytes())
    records = [{"record_id": "planned-" + row["case_id"], "category": "planned",
                "public_text": row["prompt"], "token_ids": row["token_ids"]} for row in cases["cases"]]
    records.append({"record_id": "unrelated", "category": "planned", "public_text": "fabricated",
                    "token_ids": [50256, 4, 3]})
    return cases, {"version": "fabricated", "records": records}


def test_exact_reservation_and_frozen_minimum(public):
    cases, inventory = public
    result = study.check_public_payload(inventory, digest(inventory), cases, [[50256, 4, 3]])
    assert result["own_reserved_rows"] == 48
    assert result["other_row_comparisons"] == 48
    assert len(study.base.EXTRA) == len(set(study.base.EXTRA))
    with pytest.raises(study.base.StudyError, match="missing public token row"):
        study.check_public_payload(inventory, digest(inventory), cases, [[50256, 3, 4]])


def test_published_support_commitment_and_prior_baseline_bindings():
    cases, reference, prior = study._fixed(ROOT)
    assert hashlib.sha256(cases).hexdigest() == study.base.CASES_SHA
    assert prior["model_state_sha256"] == reference["model_state_sha256"]
    assert prior["weight_sha256"] == reference["weight_sha256"]
    assert prior["buffer_state_sha256"] == "f12cbc9f0cdf2fcb9567418d7d1129da0dd6db4f6c7aaa7f9edb32b3fe7668e8"


@pytest.mark.parametrize("mutation", ["missing", "changed", "duplicate", "prefix"])
def test_rehashed_bad_reservation_or_foreign_collision_rejects(public, mutation):
    cases, inventory = public
    if mutation == "missing":
        inventory["records"].pop(0)
    elif mutation == "changed":
        inventory["records"][0]["public_text"] += "x"
    elif mutation == "duplicate":
        record = dict(inventory["records"][0])
        record["record_id"] = "independent-duplicate"
        inventory["records"].append(record)
    else:
        inventory["records"].append({"record_id": "independent-prefix", "category": "planned",
                                     "public_text": "fabricated", "token_ids": cases["cases"][0]["token_ids"][:-1]})
    with pytest.raises(study.base.StudyError):
        study.check_public_payload(inventory, digest(inventory), cases)


def test_custody_keeps_public_frame_distinct_from_no_protected_rows(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"custodian": "natural", "reported_by": "fabricated", "observed_utc": now,
             "status": "public_frame_clear", "case_sha256": study.base.CASES_SHA,
             "frame_sha256": study.SUPPORT_SHA, "report_sha256": study.FRAME_SHA,
             "commitment_sha256": study.NAT_COMMITMENT_SHA, "original_split_unchanged": True,
             "no_additional_relevant_splits_reported": True, "successor_status": "no_successor_split_reported"},
            {"custodian": "injection", "reported_by": "fabricated", "observed_utc": now,
             "status": "no_protected_rows_reported"}]
    path = tmp_path / "custody.json"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    assert study.check_custody(path)["observations"] == rows
    rows[0]["status"] = "no_protected_rows_reported"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    with pytest.raises(study.base.StudyError, match="natural"):
        study.check_custody(path)


def test_unknown_successor_and_wrong_commitment_reject(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"custodian": "natural", "reported_by": "fabricated", "observed_utc": now,
             "status": "public_frame_clear", "case_sha256": study.base.CASES_SHA,
             "frame_sha256": study.SUPPORT_SHA, "report_sha256": study.FRAME_SHA,
             "commitment_sha256": study.NAT_COMMITMENT_SHA, "original_split_unchanged": True,
             "no_additional_relevant_splits_reported": True, "successor_status": "unknown"},
            {"custodian": "injection", "reported_by": "fabricated", "observed_utc": now,
             "status": "no_protected_rows_reported"}]
    path = tmp_path / "custody.json"
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    with pytest.raises(study.base.StudyError, match="successor"):
        study.check_custody(path)
    rows[0]["successor_status"] = "no_successor_split_reported"
    rows[0]["commitment_sha256"] = "0" * 64
    path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
    with pytest.raises(study.base.StudyError, match="natural"):
        study.check_custody(path)


def test_direct_child_cannot_be_reclassified_by_phase_abbreviation():
    with pytest.raises(SystemExit):
        study.main(["child", "--checkout", str(ROOT), "--venv", str(ROOT), "--dependency", str(ROOT),
                    "--work", str(ROOT), "--pha=screen"])


@pytest.mark.parametrize("changed", ["buffer", "runtime"])
def test_changed_baseline_state_fences_before_screen(changed, tmp_path):
    guard = SimpleNamespace(runtime_settings={"synthetic": True}, state_sha="a" * 64,
                            weight_sha="b" * 64, buffer_sha="c" * 64)
    runtime = {"synthetic": True}
    manifest = {"baseline_runtime_settings": runtime.copy(), "original_model_state_sha256": guard.state_sha,
                "original_weight_sha256": guard.weight_sha, "baseline_buffer_state_sha256": guard.buffer_sha}
    calls = []

    class Runner:
        @staticmethod
        def screen(*args):
            calls.append(args)
            return "unexpected"

    if changed == "buffer":
        guard.buffer_sha = "f" * 64
    else:
        guard.runtime_settings = {"synthetic": False}
    with pytest.raises(study.base.StudyError, match="buffers or runtime"):
        study._run_bound_screen(guard, runtime, manifest, Runner(), b"x", {}, tmp_path)
    assert calls == []
