"""Synthetic saved-output audits; no pretrained model or GPU is used."""

import copy

import numpy as np
import pytest

from test_targeted_edit_v2_gate import fake_handoff  # shared synthetic, reviewed handoff fixture
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_evidence as evidence
from looking_glass import targeted_edit_v2_evaluation_evidence as audit
from looking_glass import targeted_edit_v2_gate as gate


def write_json(path, value):
    path.write_bytes(v1._canonical(value))


def refresh(directory):
    terminal = v1._decode_document((directory / "terminal.json").read_bytes())
    terminal["artifacts"] = {name: v1._sha((directory / name).read_bytes()) for name in terminal["artifacts"]}
    write_json(directory / "terminal.json", terminal)
    return v1._sha((directory / "terminal.json").read_bytes())


@pytest.fixture
def saved_evaluation(fake_handoff, monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 1)
    monkeypatch.setattr(v1, "PARIS", 2)
    cases = tuple((*case[:4], (" label", 3) if case[1] == "locality" else None)
                  for case in v2.EVALUATION_CASES)
    monkeypatch.setattr(v2, "EVALUATION_CASES", cases)
    fake_handoff["manifest"]["v1_compatibility"]["environment"]["processor"] = None
    fake_handoff["recorded"]["metadata"]["environment"]["processor"] = None
    manifest_sha = v1._sha(v1._canonical(fake_handoff["manifest"]))
    fake_handoff["call"]["manifest_sha256"] = manifest_sha
    fake_handoff["recorded"]["manifest_sha256"] = manifest_sha
    fake_handoff["recorded"]["metadata"]["manifest_sha256"] = manifest_sha
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    manifest = v1._decode_document(snapshot.manifest_bytes)
    selection = v1._decode_document(snapshot.selection_bytes)
    dev = v1._decode_document(snapshot.metadata_bytes)
    directory = tmp_path / "evaluation"
    directory.mkdir()
    common = {"version": v2.VERSION, "phase": audit.PHASE, "run_id": "1" * 32,
              "manifest_sha256": manifest_sha, "source_commit": manifest["source_commit"],
              "selection_commit": snapshot.selection_commit, "selection_terminal_sha256": snapshot.terminal_sha256,
              "evaluation_witness_commit": "2" * 40, "evaluation_witness_sha256": "3" * 64}
    write_json(directory / "started.json", {**common, "started_utc": "2026-10-29T03:32:00+00:00"})
    owner = {"run_id": "4" * 32, "pid": 234}
    supervisor = {**owner, "exit_code": 0, "clean": True, "timed_out": False, "reaped": True}
    write_json(directory / "owner.json", {**common, "supervisor_run_id": owner["run_id"], "pid": owner["pid"]})
    metadata = {**common, **{key: copy.deepcopy(dev[key]) for key in
                            ("model", "environment", "compatibility", "controls", "buffer_state_sha256",
                             "runtime_settings", "analysis_denormal_probe_before", "analysis_denormal_probe_after")},
                "supervision": {"owner": owner, "supervisor": supervisor}}
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "selection.json", {**common, "selection": selection})
    baseline = np.zeros((len(cases), v1.VOCAB_SIZE), dtype="<f4")
    baseline[:4, v1.LONDON] = -1
    arrays = {name: baseline.copy() for name in audit.CONDITIONS}
    arrays["selected"][:4, v1.LONDON] = 1
    report = v2.evaluate(arrays, selection)
    assert report["decision"] == "success"  # tests below challenge a favorable numeric result
    write_json(directory / "report.json", {**common, "result": report})
    states, observations = bytearray(), bytearray()
    compatibility = dev["compatibility"]
    state, weight, buffer = compatibility["model_state_sha256"], compatibility["weight_sha256"], dev["buffer_state_sha256"]
    applied = {"original": weight,
               **{name: dev["controls"][name]["realized_weight_sha256"] for name in ("selected", "random", "opposite")},
               "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["realized_weight_sha256"]}
    row_bytes = v1.VOCAB_SIZE * 4
    for condition in audit.CONDITIONS:
        states.extend(v1._canonical({**common, "condition": condition, "model_state_before_sha256": state,
            "weight_before_sha256": weight, "applied_weight_sha256": applied[condition],
            "model_state_after_sha256": state, "weight_after_sha256": weight,
            "buffer_state_before_sha256": buffer, "buffer_state_after_sha256": buffer,
            "runtime_settings_before": dev["runtime_settings"], "runtime_settings_after": dev["runtime_settings"],
            "denormal_probe_before": evidence.DENORMAL_PROBE, "denormal_probe_after": evidence.DENORMAL_PROBE,
            "flags_restored": True, "grads_cleared": True, "hooks_restored": True}))
        raw = arrays[condition].tobytes()
        (directory / f"{condition}.f32").write_bytes(raw)
        for index, case in enumerate(cases):
            offset = index * row_bytes
            observations.extend(v1._canonical({**common, "condition": condition, "case_id": case[0],
                "offset": offset, "bytes": row_bytes, "sha256": v1._sha(raw[offset:offset + row_bytes]),
                "denormal_probe_before": evidence.DENORMAL_PROBE, "denormal_probe_after": evidence.DENORMAL_PROBE}))
    (directory / "observations.jsonl").write_bytes(observations)
    (directory / "condition_states.jsonl").write_bytes(states)
    artifacts = {*audit.DOCUMENTS, *(f"{name}.f32" for name in audit.CONDITIONS)}
    terminal = {**common, "status": "complete", "finished_utc": "2026-10-29T03:33:00+00:00",
                "cleanup": {name: True for name in audit.CLEANUP_KEYS},
                "artifacts": {name: v1._sha((directory / name).read_bytes()) for name in artifacts}}
    write_json(directory / "terminal.json", terminal)
    args = {"snapshot": snapshot, "manifest_sha256": manifest_sha,
            "terminal_sha256": v1._sha((directory / "terminal.json").read_bytes()),
            "evaluation_witness_commit": common["evaluation_witness_commit"],
            "evaluation_witness_sha256": common["evaluation_witness_sha256"],
            "witness_observed_utc": "2026-10-29T03:31:00+00:00"}
    return directory, args


def test_valid_evaluation_recomputes_metrics_and_marks_scope(saved_evaluation):
    directory, args = saved_evaluation
    result = audit.audit_evaluation(directory, **args)
    assert result["numerical_decision"] == "success"
    assert result["status"] == "matching_saved_evaluation"
    assert "not proved" in result["scope"]


def test_auditor_rechecks_its_thread_denormal_mode_after_scoring(saved_evaluation, monkeypatch):
    directory, args = saved_evaluation
    real_probe = v1._denormal_probe
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise v1.StudyError("auditor denormal mode changed")
        return real_probe()
    monkeypatch.setattr(v1, "_denormal_probe", probe)
    with pytest.raises(v1.StudyError, match="auditor denormal"):
        audit.audit_evaluation(directory, **args)
    assert calls == 2


@pytest.mark.parametrize("field", audit.CLEANUP_KEYS)
def test_favorable_metrics_cannot_override_failed_cleanup(saved_evaluation, field):
    directory, args = saved_evaluation
    terminal = v1._decode_document((directory / "terminal.json").read_bytes())
    terminal["cleanup"][field] = False
    write_json(directory / "terminal.json", terminal)
    args["terminal_sha256"] = v1._sha((directory / "terminal.json").read_bytes())
    with pytest.raises(v1.StudyError, match="cleanup"):
        audit.audit_evaluation(directory, **args)


@pytest.mark.parametrize("mutation", ["owner", "reaped", "model", "runtime", "environment_missing", "buffer", "control", "selection",
                                      "witness", "late_witness", "state", "state_runtime", "state_probe",
                                      "journal", "report", "row_rehashed", "extra", "missing"])
def test_rehashed_inconsistencies_and_partial_attempts_never_pass(saved_evaluation, mutation):
    directory, args = saved_evaluation
    if mutation in ("owner", "reaped", "model", "runtime", "environment_missing", "buffer", "control"):
        path = directory / "metadata.json"
        record = v1._decode_document(path.read_bytes())
        if mutation == "owner": record["supervision"]["owner"]["pid"] += 1
        if mutation == "reaped": record["supervision"]["supervisor"]["reaped"] = False
        if mutation == "model": record["model"] = {"id": "different"}
        if mutation == "runtime": record["runtime_settings"]["mkldnn_deterministic"] = True
        if mutation == "environment_missing": del record["environment"]["processor"]
        if mutation == "buffer": record["buffer_state_sha256"] = "8" * 64
        if mutation == "control": record["controls"]["selected"]["realized_weight_sha256"] = "8" * 64
        write_json(path, record)
    elif mutation == "selection":
        path = directory / "selection.json"
        record = v1._decode_document(path.read_bytes())
        record["selection"]["point_prediction"] = 99.0
        write_json(path, record)
    elif mutation == "witness":
        args["evaluation_witness_sha256"] = "8" * 64
    elif mutation == "late_witness":
        args["witness_observed_utc"] = "2026-10-29T03:32:01+00:00"
    elif mutation in ("state", "state_runtime", "state_probe"):
        path = directory / "condition_states.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        record = v1._decode_document(lines[1])
        if mutation == "state": record["applied_weight_sha256"] = "8" * 64
        if mutation == "state_runtime": record["runtime_settings_after"]["intraop_threads"] = 2
        if mutation == "state_probe": record["denormal_probe_after"]["subnormal_times_one_bits"] = 0
        lines[1] = v1._canonical(record)
        path.write_bytes(b"".join(lines))
    elif mutation == "journal":
        path = directory / "observations.jsonl"
        path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    elif mutation == "report":
        path = directory / "report.json"
        record = v1._decode_document(path.read_bytes())
        record["result"]["prediction"]["mean_absolute_error"] = 42.0
        write_json(path, record)
    elif mutation == "row_rehashed":
        path = directory / "selected.f32"
        raw = bytearray(path.read_bytes())
        raw[4:8] = np.float32(1.5).tobytes()
        path.write_bytes(raw)
        journal = directory / "observations.jsonl"
        lines = journal.read_bytes().splitlines(keepends=True)
        index = len(v2.EVALUATION_CASES)
        record = v1._decode_document(lines[index])
        record["sha256"] = v1._sha(raw[:v1.VOCAB_SIZE * 4])
        lines[index] = v1._canonical(record)
        journal.write_bytes(b"".join(lines))
    elif mutation == "extra":
        (directory / "unexpected").write_bytes(b"extra")
    else:
        (directory / "terminal.json").unlink()
    if mutation not in ("witness", "late_witness", "extra", "missing"):
        args["terminal_sha256"] = refresh(directory)
    with pytest.raises(v1.StudyError):
        audit.audit_evaluation(directory, **args)
