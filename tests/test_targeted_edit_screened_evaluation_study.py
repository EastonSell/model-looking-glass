import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("screened_evaluation_study", ROOT / "scripts/targeted_edit_screened_evaluation_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
CASES = (ROOT / "experiments/targeted_edit_screened_public_cases_v1.json").read_bytes()


def test_source_inventory_extras_are_unique_and_include_both_owners():
    assert len(study.base.EXTRA) == len(set(study.base.EXTRA))
    assert "scripts/targeted_edit_screened_screen_study.py" in study.base.EXTRA
    assert study.base.SCRIPT in study.base.EXTRA


def payload(observed=False):
    rows = json.loads(CASES)["cases"]
    records = []
    for row in rows:
        screened = observed and row["role"] == "target"
        record = {"record_id": ("observed-" if screened else "planned-") + row["case_id"],
                  "category": "observed-saved" if screened else "planned",
                  "token_ids": row["token_ids"], "public_text": row["prompt"]}
        if screened:
            record.update({"reference_commit": study.SCREEN_EVIDENCE_COMMIT,
                           "references": list(study.SCREEN_EVIDENCE_PATHS), "provenance": study.SCREEN_PROVENANCE})
        records.append(record)
    return {"version": "fake", "records": records}


def checked(monkeypatch, value, minimum=()):
    monkeypatch.setattr(study.screen_owner, "_case_bytes", lambda _checkout: CASES)
    calls = []
    monkeypatch.setattr(study.base, "check_inventory_attestation", lambda *args: calls.append(args))
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    result = study.current_inventory(None, None, None, raw, hashlib.sha256(raw).hexdigest(), {}, minimum)
    assert len(calls) == 1
    return result


def test_exact_planned_and_authorized_observed_rows_are_accepted(monkeypatch):
    for observed in (False, True):
        result = checked(monkeypatch, payload(observed))
        assert result["own_reserved_or_observed_rows"] == 44
        assert result["known_relations"] == 0


def test_own_change_or_foreign_duplicate_cannot_be_exempted(monkeypatch):
    value = payload(True)
    value["records"][0]["public_text"] += " changed"
    with pytest.raises(study.base.StudyError, match="own reservation"):
        checked(monkeypatch, value)
    value = payload(True)
    value["records"].append({**value["records"][0], "record_id": "foreign-duplicate"})
    with pytest.raises(study.base.StudyError, match="prefix relation"):
        checked(monkeypatch, value)
    value = payload(True)
    value["records"][0]["reference_commit"] = "0" * 40
    with pytest.raises(study.base.StudyError, match="own reservation"):
        checked(monkeypatch, value)
    value = payload(True)
    value["records"][32]["record_id"] = "observed-" + value["records"][32]["record_id"].removeprefix("planned-")
    value["records"][32]["category"] = "observed-saved"
    with pytest.raises(study.base.StudyError, match="own reservation"):
        checked(monkeypatch, value)


def test_run_requires_every_row_frozen_in_the_evaluation_manifest(monkeypatch):
    previous = {"exposure": {"known_complete_rows": []}}
    manifest = {"exposure": {"known_complete_rows": [[50000, 50001]]}}
    assert study.minimum_exposure(previous, None) == []
    assert study.minimum_exposure(previous, manifest) == [[50000, 50001]]
    value = payload()
    value["records"].append({"record_id": "later", "category": "planned", "token_ids": [50000, 50001],
                             "public_text": "separate"})
    checked(monkeypatch, value, study.minimum_exposure(previous, manifest))
    value["records"].pop()
    with pytest.raises(study.base.StudyError, match="previous public rows"):
        checked(monkeypatch, value, study.minimum_exposure(previous, manifest))


def test_phase_completion_requires_all_full_matrices_and_report(tmp_path):
    names = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance", "matched")
    for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
                 "development-review-attestation.json", "screen-review-attestation.json", "public-exposure.json",
                 "public-exposure-attestation.json"):
        (tmp_path / name).write_bytes(b"{}")
    directory = tmp_path / "evaluation"
    directory.mkdir()
    for name in names:
        with (directory / (name + ".f32")).open("wb") as stream:
            stream.truncate(14 * 50257 * 4)
        (directory / ("state-" + name + ".json")).write_bytes(b"{}")
    (directory / "observations.jsonl").write_bytes(b"")
    report = {"selected_count": 2, "discovery_match_available": True,
              "provisional_equal_norm_decision": "ineligible", "provisional_matched_discovery_decision": "ineligible"}
    raw = json.dumps(report).encode()
    (directory / "provisional-report.json").write_bytes(raw)
    observed = {"selected_count": 2, "report_sha256": hashlib.sha256(raw).hexdigest(),
                "provisional_equal_norm_decision": "ineligible", "provisional_matched_discovery_decision": "ineligible"}
    assert len(study.phase_inventory(tmp_path, "evaluation", observed)) == 23
    (directory / "matched.f32").unlink()
    with pytest.raises(study.base.StudyError, match="output set"):
        study.phase_inventory(tmp_path, "evaluation", observed)
