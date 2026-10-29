"""Synthetic checks of the screened study's prospective owner boundary."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_targeted_edit_projected_development import guarded_model
from looking_glass import targeted_edit_screened_development as development


path = Path(__file__).resolve().parents[1] / "scripts/targeted_edit_screened_study.py"
spec = importlib.util.spec_from_file_location("screened_study_test", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def case_rows():
    return [{"case_id": f"case-{i}", "prompt": f"prompt {i}", "token_ids": [1, i + 1]}
            for i in range(44)]


def inventory(records):
    value = {"version": "synthetic-v1", "records": records}
    digest = study.base.digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode())
    return value, digest


def reserved(row):
    return {"record_id": "planned-" + row["case_id"], "public_text": row["prompt"],
            "category": "planned", "token_ids": row["token_ids"]}


def test_public_check_allows_only_exact_complete_own_reservations():
    rows = case_rows()
    unrelated = {"record_id": "old", "token_ids": [7, 8]}
    payload, digest = inventory([unrelated])
    assert study.check_public_payload(payload, digest, {"cases": rows})["own_reserved_rows"] == 0
    payload, digest = inventory([unrelated, *map(reserved, rows)])
    result = study.check_public_payload(payload, digest, {"cases": rows}, [[7, 8]], include_rows=True)
    assert result["own_reserved_rows"] == 44 and len(result["known_complete_rows"]) == 45
    payload, digest = inventory([unrelated, reserved(rows[0])])
    with pytest.raises(study.base.StudyError, match="partial"):
        study.check_public_payload(payload, digest, {"cases": rows})


def test_other_identity_cannot_hide_duplicate_or_prefix():
    rows = case_rows()
    for duplicate in (rows[0]["token_ids"], rows[0]["token_ids"] + [3], rows[0]["token_ids"][:1]):
        payload, digest = inventory([*map(reserved, rows), {"record_id": "other", "token_ids": duplicate}])
        with pytest.raises(study.base.StudyError, match="prefix relation"):
            study.check_public_payload(payload, digest, {"cases": rows})
    payload, digest = inventory([])
    with pytest.raises(study.base.StudyError, match="missing"):
        study.check_public_payload(payload, digest, {"cases": rows}, [[7, 8]])


def test_changed_own_reservation_or_digest_fails_closed():
    rows = case_rows()
    all_records = list(map(reserved, rows))
    all_records[0]["public_text"] = "different"
    payload, digest = inventory(all_records)
    with pytest.raises(study.base.StudyError, match="reservation identity"):
        study.check_public_payload(payload, digest, {"cases": rows})
    with pytest.raises(study.base.StudyError, match="digest"):
        study.check_public_payload(payload, "0" * 64, {"cases": rows})


def test_fresh_custody_status_distinguishes_private_pass_and_no_rows(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    observations = [{"custodian": "natural", "reported_by": "custodian", "observed_utc": now,
                     "status": "private_comparison_clear", "case_sha256": study.base.CASES_SHA,
                     "receipt_sha256": study.NATURAL_RECEIPT_SHA},
                    {"custodian": "injection", "reported_by": "custodian", "observed_utc": now,
                     "status": "no_protected_rows_reported"}]
    path = tmp_path / "custody.json"
    path.write_text(json.dumps({"observations": observations}), encoding="utf-8")
    assert study.custody_observation(path)["observations"] == observations
    observations[0]["status"] = "no_protected_rows_reported"
    path.write_text(json.dumps({"observations": observations}), encoding="utf-8")
    with pytest.raises(study.base.StudyError, match="case file"):
        study.custody_observation(path)
    observations[0]["status"] = "private_comparison_clear"
    observations[0]["case_sha256"] = "0" * 64
    path.write_text(json.dumps({"observations": observations}), encoding="utf-8")
    with pytest.raises(study.base.StudyError, match="case file"):
        study.custody_observation(path)


def test_parent_rejects_missing_development_artifact_before_completion(tmp_path):
    _model, guard = guarded_model()
    candidate = development.prepare_development(guard, (3,), ((1,), (2,)), 3, 1, tmp_path / "candidate")
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
                 "public-exposure.json", "public-exposure-attestation.json"):
        (tmp_path / name).write_bytes(b"{}")
    candidate_raw = (tmp_path / "candidate" / "pre-evaluation-candidate.json").read_bytes()
    observed = {"candidate_sha256": hashlib.sha256(candidate_raw).hexdigest()}
    saved = study.inventory_attempt(tmp_path, "development")
    study.validate_phase_output(tmp_path, "development", observed, saved)
    assert len(candidate["artifacts"]) == 97 and len(saved) == 103
    (tmp_path / "candidate" / "calibration-projected-algebraic.f64").unlink()
    damaged = study.inventory_attempt(tmp_path, "development")
    with pytest.raises(study.base.StudyError, match="output set"):
        study.validate_phase_output(tmp_path, "development", observed, damaged)


@pytest.mark.parametrize("flag", ["--phase=evaluation", "--phase", "--phase=screen"])
def test_unsupported_pool_execution_never_enters_base_owner(monkeypatch, flag):
    monkeypatch.setattr(study.base, "entry", lambda _args: pytest.fail("unexpected base entry"))
    args = ["run", flag]
    if flag == "--phase":
        args.append("evaluation")
    with pytest.raises(study.base.StudyError, match="only discovery"):
        study.entry(args)


@pytest.mark.parametrize("args", [
    ["run", "--pha=evaluation"],
    ["run", "--pha", "evaluation"],
    ["run", "--phase=development", "--phas=evaluation"],
])
def test_abbreviated_phase_options_never_reach_reused_parser(monkeypatch, args):
    monkeypatch.setattr(study.base, "entry", lambda _args: pytest.fail("unexpected base entry"))
    with pytest.raises(study.base.StudyError, match="abbreviated phase"):
        study.entry(args)
