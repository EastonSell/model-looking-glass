"""Synthetic orchestration checks; no checkpoint or pretrained-model forward."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_targeted_edit_v2_gate import fake_handoff
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_edits as edits
from looking_glass import targeted_edit_v2_evidence as evidence
from looking_glass import targeted_edit_v2_evaluation_owner as evaluation
from looking_glass import targeted_edit_v2_evaluation_evidence as audit
from looking_glass import targeted_edit_v2_gate as gate
from looking_glass import targeted_edit_v2_owner as owner
from looking_glass.supervisor import SupervisionResult


def _inputs(fake_handoff, monkeypatch):
    manifest = fake_handoff["manifest"]
    manifest_sha = v1._sha(v1._canonical(manifest))
    fake_handoff["call"]["manifest_sha256"] = manifest_sha
    fake_handoff["recorded"]["manifest_sha256"] = manifest_sha
    fake_handoff["recorded"]["metadata"]["manifest_sha256"] = manifest_sha
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    values = fake_handoff["args"]
    guard = SimpleNamespace(state_sha=values["model_state_sha256"],
        weight_sha=v1._tensor_sha(values["original"]), buffer_sha=values["buffer_state_sha256"],
        original=values["original"], runtime_settings=values["runtime_settings"], verify=lambda: None)
    identity = {"version": v2.VERSION, "phase": "evaluation", "run_id": "1" * 32,
        "manifest_sha256": manifest_sha, "source_commit": manifest["source_commit"],
        "selection_commit": snapshot.selection_commit, "selection_terminal_sha256": snapshot.terminal_sha256,
        "evaluation_witness_commit": "2" * 40, "evaluation_witness_sha256": "3" * 64}
    loaded = SimpleNamespace(metadata={"model_id": "synthetic-only"})
    compatibility = fake_handoff["recorded"]["metadata"]["compatibility"]
    prepared = (loaded, guard, values["v1_gradient"], values["v1_random"], values["discovery"],
                values["reference_bytes"], compatibility, values["environment"])
    deps = (torch, np, None, v1, v2, edits, evidence, None, None, None, gate)
    return identity, manifest, snapshot, guard, prepared, deps


def test_evaluation_work_orders_reconstruction_before_any_rows(fake_handoff, monkeypatch, tmp_path):
    identity, manifest, snapshot, guard, prepared, deps = _inputs(fake_handoff, monkeypatch)
    calls = []
    def prepare(*_args):
        calls.append("reconstruct_v1")
        return prepared
    def controls(actual_snapshot, **kwargs):
        calls.append("verify_controls")
        assert actual_snapshot is snapshot
        assert kwargs["original"] is guard.original and kwargs["buffer_state_sha256"] == guard.buffer_sha
        assert kwargs["runtime_settings"] == guard.runtime_settings
        return dict(fake_handoff["updates"])
    def rows(output, actual_identity, actual_snapshot, actual_guard, updates, applied, *_deps):
        calls.append("rows")
        assert output == tmp_path and actual_identity == identity and actual_snapshot is snapshot
        assert actual_guard is guard and updates["v1_1_percent"] is prepared[2]
        assert set(updates) == {"selected", "random", "opposite", "v1_1_percent"}
        assert set(applied) == set(v2.EVALUATION_CONDITIONS)
    def analyze(*_args):
        calls.append("analysis")
        return {"analysis_denormal_probe_before": evidence.DENORMAL_PROBE,
                "analysis_denormal_probe_after": evidence.DENORMAL_PROBE}
    monkeypatch.setattr(owner, "_prepare_v1_reconstruction", prepare)
    monkeypatch.setattr(gate, "reconstruct_selected_controls", controls)
    monkeypatch.setattr(evaluation, "_evaluation_conditions", rows)
    monkeypatch.setattr(evaluation, "_evaluation_analysis", analyze)
    result = evaluation._evaluation_work(None, None, tmp_path, identity, manifest, snapshot, deps)
    assert calls == ["reconstruct_v1", "verify_controls", "rows", "analysis"]
    assert result["controls"] == fake_handoff["recorded"]["metadata"]["controls"]
    assert result["model"]["evaluation_mode"] is True
    assert not (tmp_path / "terminal.json").exists()


@pytest.mark.parametrize("failure", ["source", "manifest", "selection", "same_run", "interrupted_discovery",
                                      "reconstruction", "control_set"])
def test_evaluation_work_fails_before_held_out_rows(fake_handoff, monkeypatch, tmp_path, failure):
    identity, manifest, snapshot, _guard, prepared, deps = _inputs(fake_handoff, monkeypatch)
    if failure == "source":
        identity["source_commit"] = "0" * 40
    elif failure == "manifest":
        identity["manifest_sha256"] = "0" * 64
    elif failure == "selection":
        identity["selection_commit"] = "0" * 40
    elif failure == "same_run":
        identity["run_id"] = fake_handoff["recorded"]["metadata"]["run_id"]
    calls = []
    def prepare(*_args):
        calls.append("prepare")
        if failure == "interrupted_discovery":
            raise KeyboardInterrupt()
        return prepared
    def controls(*_args, **_kwargs):
        calls.append("controls")
        if failure == "reconstruction":
            raise v1.StudyError("synthetic incompatible reconstruction")
        if failure == "control_set":
            return {"selected": fake_handoff["updates"]["selected"]}
        return dict(fake_handoff["updates"])
    monkeypatch.setattr(owner, "_prepare_v1_reconstruction", prepare)
    monkeypatch.setattr(gate, "reconstruct_selected_controls", controls)
    monkeypatch.setattr(evaluation, "_evaluation_conditions", lambda *_args: calls.append("rows"))
    expected_error = KeyboardInterrupt if failure == "interrupted_discovery" else (owner.OwnerError, v1.StudyError)
    with pytest.raises(expected_error):
        evaluation._evaluation_work(None, None, tmp_path, identity, manifest, snapshot, deps)
    assert "rows" not in calls and not (tmp_path / "terminal.json").exists()
    if failure in {"source", "manifest", "selection", "same_run"}:
        assert calls == []
    if failure == "interrupted_discovery":
        assert calls == ["prepare"] and not (tmp_path / "observations.jsonl").exists()


def test_forged_control_after_reconstruction_is_rejected_before_rows(fake_handoff, monkeypatch, tmp_path):
    identity, manifest, snapshot, guard, prepared, deps = _inputs(fake_handoff, monkeypatch)
    guard.torch = torch
    fake_controls = dict(fake_handoff["updates"])
    fake_controls["opposite"] = fake_controls["opposite"] + 1
    rows = []
    monkeypatch.setattr(owner, "_prepare_v1_reconstruction", lambda *_args: prepared)
    monkeypatch.setattr(gate, "reconstruct_selected_controls", lambda *_args, **_kwargs: fake_controls)
    monkeypatch.setattr(v1, "_logit_row", lambda *_args: rows.append(True))
    with pytest.raises(owner.OwnerError, match="intended update"):
        evaluation._evaluation_work(None, None, tmp_path, identity, manifest, snapshot, deps)
    assert rows == [] and not (tmp_path / "observations.jsonl").exists()


def test_synthetic_work_reconstructs_writes_and_passes_independent_reader(fake_handoff, monkeypatch, tmp_path):
    # Use the real selection/control reconstruction and saved-output reader;
    # the model/discovery are synthetic and no checkpoint is loaded.
    values = fake_handoff["args"]
    reference = v1._decode_document(values["reference_bytes"])
    reference["model"]["evaluation_mode"] = True
    reference_bytes = v1._canonical(reference)
    monkeypatch.setattr(edits, "V1_PRECOMMIT_SHA256", v1._sha(reference_bytes))
    values["reference_bytes"] = reference_bytes
    values["model_metadata"] = reference["model"]
    compatibility = edits.verify_v1_compatibility(**{key: values[key] for key in
        ("reference_bytes", "environment", "model_metadata", "model_state_sha256", "original",
         "v1_gradient", "v1_random", "discovery")})
    fake_handoff["manifest"]["v1_compatibility"]["precommit_sha256"] = v1._sha(reference_bytes)
    fake_handoff["manifest"]["v1_compatibility"]["model"] = reference["model"]
    fake_handoff["recorded"]["metadata"]["model"] = reference["model"]
    fake_handoff["recorded"]["metadata"]["compatibility"] = compatibility
    identity, manifest, snapshot, guard, prepared, deps = _inputs(fake_handoff, monkeypatch)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 1)
    monkeypatch.setattr(v1, "PARIS", 2)
    cases = tuple((*case[:4], (" synthetic", 3) if case[1] == "locality" else None)
                  for case in v2.EVALUATION_CASES)
    monkeypatch.setattr(v2, "EVALUATION_CASES", cases)
    development = v1._decode_document(snapshot.metadata_bytes)
    applied = {"original": guard.weight_sha,
               **{name: development["controls"][name]["realized_weight_sha256"]
                  for name in ("selected", "random", "opposite")},
               "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["realized_weight_sha256"]}
    assert len(set(applied.values())) == len(applied)
    guard.torch, guard.model, guard.poisoned, guard.last_cleanup = torch, guard, False, "unverified"
    calls = []
    def verify():
        assert not guard.poisoned
    def condition(_delta, expected, callback):
        verify()
        guard.current = next(name for name, digest in applied.items() if digest == expected)
        calls.append(guard.current)
        guard.poisoned = True
        callback()
        guard.last_cleanup = "verified"
    def commit():
        assert guard.poisoned and guard.last_cleanup == "verified"
        guard.poisoned = False
    guard.verify, guard.condition, guard.commit_condition = verify, condition, commit
    monkeypatch.setattr(owner, "_runtime_settings", lambda _torch: guard.runtime_settings)
    monkeypatch.setattr(owner, "_prepare_v1_reconstruction", lambda *_args: prepared)
    case_index = {tuple(case[3]): index for index, case in enumerate(cases)}
    def fake_row(model, tokens):
        result = np.zeros(16, dtype="<f4")
        if case_index[tuple(tokens)] < 4:
            result[1] = 1 if model.current == "selected" else -1
        return result.tobytes()
    monkeypatch.setattr(v1, "_logit_row", fake_row)
    staging, destination = tmp_path / "staging", tmp_path / "published"
    staging.mkdir()
    (staging / "started.json").write_bytes(v1._canonical({**identity, "started_utc": "2026-10-29T03:32:00+00:00"}))
    (staging / "owner.json").write_bytes(v1._canonical({**identity, "supervisor_run_id": "4" * 32, "pid": 234}))
    candidate = evaluation._evaluation_work(None, None, staging, identity, manifest, snapshot, deps)
    assert calls == list(v2.EVALUATION_CONDITIONS) and not (staging / "terminal.json").exists()
    result = evaluation._publish_evaluation(staging, destination, identity, snapshot, candidate,
        {name: True for name in audit.CLEANUP_KEYS}, SupervisionResult("4" * 32, 234, 0, True, False, "synthetic"),
        "2026-10-29T03:31:00+00:00", "2026-10-29T03:33:00+00:00", v1, v2, evidence, gate, audit, torch)
    checked = audit.audit_evaluation(destination, snapshot=snapshot, manifest_sha256=identity["manifest_sha256"],
        terminal_sha256=result["terminal_sha256"], evaluation_witness_commit=identity["evaluation_witness_commit"],
        evaluation_witness_sha256=identity["evaluation_witness_sha256"], witness_observed_utc="2026-10-29T03:31:00+00:00")
    assert checked["status"] == "matching_saved_evaluation" and checked["numerical_decision"] == "success"
