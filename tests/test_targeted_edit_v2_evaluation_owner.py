"""Synthetic evaluation-writer and independent-reader checks; no checkpoint."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_targeted_edit_v2_evaluation_evidence import saved_evaluation
from test_targeted_edit_v2_gate import fake_handoff
from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_evidence as evidence
from looking_glass import targeted_edit_v2_evaluation_evidence as audit
from looking_glass import targeted_edit_v2_evaluation_owner as writer
from looking_glass import targeted_edit_v2_gate as gate
from looking_glass import targeted_edit_v2_owner as owner


def _common(directory):
    terminal = v1._decode_document((directory / "terminal.json").read_bytes())
    return {key: terminal[key] for key in ("version", "phase", "run_id", "manifest_sha256", "source_commit",
            "selection_commit", "selection_terminal_sha256", "evaluation_witness_commit",
            "evaluation_witness_sha256")}


def test_writer_matches_independent_reader_and_does_not_publish_terminal(saved_evaluation, fake_handoff, monkeypatch, tmp_path):
    saved, args = saved_evaluation
    identity = _common(saved)
    snapshot = args["snapshot"]
    development = v1._decode_document(snapshot.metadata_bytes)
    manifest = v1._decode_document(snapshot.manifest_bytes)
    applied = {"original": development["compatibility"]["weight_sha256"],
               **{name: development["controls"][name]["realized_weight_sha256"]
                  for name in ("selected", "random", "opposite")},
               "v1_1_percent": manifest["v1_compatibility"]["candidates"]["gradient"]["realized_weight_sha256"]}
    class SyntheticGuard:
        def __init__(self):
            self.weight_sha = applied["original"]
            self.state_sha = development["compatibility"]["model_state_sha256"]
            self.buffer_sha = development["buffer_state_sha256"]
            self.runtime_settings = development["runtime_settings"]
            self.torch = torch
            self.original = fake_handoff["args"]["original"]
            self.model = self
            self.poisoned = False
            self.last_cleanup = "unverified"
            self.calls = []
        def verify(self):
            assert not self.poisoned
        def condition(self, delta, expected, callback):
            self.verify()
            name = "original" if delta is None else next(name for name, value in updates.items() if value is delta)
            assert expected == applied[name]
            self.poisoned = True
            self.current = name
            self.calls.append(name)
            callback()
            self.last_cleanup = "verified"
        def commit_condition(self):
            assert self.poisoned and self.last_cleanup == "verified"
            self.poisoned = False
    guard = SyntheticGuard()
    monkeypatch.setattr(owner, "_runtime_settings", lambda _torch: guard.runtime_settings)
    row_size = v1.VOCAB_SIZE * 4
    indices = {tuple(case[3]): index for index, case in enumerate(v2.EVALUATION_CASES)}
    def row(model, tokens):
        offset = indices[tuple(tokens)] * row_size
        return (saved / f"{model.current}.f32").read_bytes()[offset:offset + row_size]
    monkeypatch.setattr(v1, "_logit_row", row)
    output = tmp_path / "written"
    output.mkdir()
    updates = {**fake_handoff["updates"], "v1_1_percent": fake_handoff["args"]["v1_gradient"]}
    writer._evaluation_conditions(output, identity, snapshot, guard, updates, applied, v1, v2, evidence, gate)
    probes = writer._evaluation_analysis(output, identity, snapshot, guard, v1, v2, evidence, gate, np)
    assert guard.calls == list(v2.EVALUATION_CONDITIONS)
    assert probes["analysis_denormal_probe_before"] == evidence.DENORMAL_PROBE
    assert probes["analysis_denormal_probe_after"] == evidence.DENORMAL_PROBE
    assert not (output / "terminal.json").exists()
    for name in ("started.json", "owner.json", "metadata.json", "terminal.json"):
        shutil.copyfile(saved / name, output / name)
    assert all((output / path.name).read_bytes() == path.read_bytes() for path in saved.iterdir())
    result = audit.audit_evaluation(output, **args)
    assert result["status"] == "matching_saved_evaluation"


def _real_guard_inputs(monkeypatch):
    assert torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 4)
    monkeypatch.setattr(v2, "EVALUATION_CASES", (("one", "target", "synthetic", (1,), None),))
    model = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(1.0)
    guard = owner._ModelGuard(model, model, model.weight, torch, v1)
    original = guard.original.clone()
    updates = {name: torch.full_like(original, (index + 1) / 16)
               for index, name in enumerate(v2.EVALUATION_CONDITIONS[1:])}
    applied = {"original": guard.weight_sha,
               **{name: v1._tensor_sha(original + delta) for name, delta in updates.items()}}
    identity = {"version": v2.VERSION, "phase": "evaluation", "run_id": "a" * 32,
                "manifest_sha256": "b" * 64, "source_commit": "c" * 40,
                "selection_commit": "d" * 40, "selection_terminal_sha256": "e" * 64,
                "evaluation_witness_commit": "f" * 40, "evaluation_witness_sha256": "1" * 64}
    manifest = {"source_commit": identity["source_commit"], "v1_compatibility": {"candidates":
                {"gradient": {"realized_weight_sha256": applied["v1_1_percent"],
                              "intended_delta_sha256": v1._tensor_sha(updates["v1_1_percent"])}}}}
    manifest_bytes = v1._canonical(manifest)
    identity["manifest_sha256"] = v1._sha(manifest_bytes)
    development = {"source_commit": identity["source_commit"], "phase": "development", "run_id": "9" * 32,
        "manifest_sha256": identity["manifest_sha256"], "runtime_settings": guard.runtime_settings,
        "buffer_state_sha256": guard.buffer_sha, "compatibility": {"model_state_sha256": guard.state_sha,
        "weight_sha256": guard.weight_sha}, "controls": {name: {"realized_weight_sha256": applied[name],
        "intended_delta_sha256": v1._tensor_sha(updates[name])}
        for name in ("selected", "random", "opposite")}}
    snapshot = gate.SelectionSnapshot(manifest_bytes, v1._canonical({"development_qualified": False}),
        v1._canonical(development), identity["selection_commit"], identity["selection_terminal_sha256"])
    return model, guard, original, updates, applied, identity, snapshot


@pytest.mark.parametrize("failure", ["before_state", "after_state", "lost_ack", "edited_row"])
def test_partial_condition_stops_without_a_success_terminal(monkeypatch, tmp_path, failure):
    model, guard, original, updates, applied, identity, snapshot = _real_guard_inputs(monkeypatch)
    rows = []
    monkeypatch.setattr(v1, "_logit_row", lambda _model, ids: rows.append(ids) or np.zeros(4, dtype="<f4").tobytes())
    append = owner._append
    if failure in ("before_state", "after_state"):
        def fail_state(handle, data):
            if handle.name.endswith("condition_states.jsonl"):
                if failure == "after_state":
                    append(handle, data)
                raise OSError("interrupted state write")
            append(handle, data)
        monkeypatch.setattr(owner, "_append", fail_state)
        error = OSError
    elif failure == "lost_ack":
        commit = guard.commit_condition
        def lost_ack():
            commit()
            raise KeyboardInterrupt()
        monkeypatch.setattr(guard, "commit_condition", lost_ack)
        error = KeyboardInterrupt
    else:
        def fail_second_row(_model, ids):
            rows.append(ids)
            if len(rows) == 2:
                raise OSError("edited row failed")
            return np.zeros(4, dtype="<f4").tobytes()
        monkeypatch.setattr(v1, "_logit_row", fail_second_row)
        error = OSError
    with pytest.raises(error):
        writer._evaluation_conditions(tmp_path, identity, snapshot, guard, updates, applied, v1, v2, evidence, gate)
    assert rows == ([(1,), (1,)] if failure == "edited_row" else [(1,)])
    assert torch.equal(model.weight, original)
    assert guard.last_cleanup == "verified"
    assert guard.poisoned is (failure != "lost_ack")
    assert bool((tmp_path / "condition_states.jsonl").read_bytes()) is (failure != "before_state")
    assert not (tmp_path / "terminal.json").exists()
    assert (tmp_path / "selected.f32").exists() is (failure == "edited_row")


@pytest.mark.parametrize("mutation", ["selection_commit", "selection_terminal", "source_commit", "manifest_sha",
                                      "witness_shape", "same_run", "applied", "update", "rounded_away"])
def test_binding_and_all_edits_are_checked_before_any_held_out_row(monkeypatch, tmp_path, mutation):
    _model, guard, _original, updates, applied, identity, snapshot = _real_guard_inputs(monkeypatch)
    if mutation == "selection_commit":
        identity["selection_commit"] = "0" * 40
    elif mutation == "selection_terminal":
        identity["selection_terminal_sha256"] = "0" * 64
    elif mutation == "source_commit":
        identity["source_commit"] = "0" * 40
    elif mutation == "manifest_sha":
        identity["manifest_sha256"] = "0" * 64
    elif mutation == "witness_shape":
        identity["evaluation_witness_commit"] = "bad"
    elif mutation == "same_run":
        identity["run_id"] = "9" * 32
    elif mutation == "applied":
        applied["opposite"] = "0" * 64
    elif mutation == "update":
        updates["v1_1_percent"] = updates["v1_1_percent"] + 1
    else:
        old = updates["v1_1_percent"]
        changed = old.clone()
        changed.view(-1)[0] = torch.nextafter(changed.view(-1)[0], torch.tensor(float("inf")))
        assert v1._tensor_sha(old) != v1._tensor_sha(changed)
        assert torch.equal((guard.original + old).view(torch.int32), (guard.original + changed).view(torch.int32))
        updates["v1_1_percent"] = changed
    rows = []
    monkeypatch.setattr(v1, "_logit_row", lambda *_args: rows.append(True))
    with pytest.raises(owner.OwnerError):
        writer._evaluation_conditions(tmp_path, identity, snapshot, guard, updates, applied, v1, v2, evidence, gate)
    assert rows == [] and not (tmp_path / "observations.jsonl").exists()


def test_update_drift_after_baseline_stops_before_that_control(monkeypatch, tmp_path):
    model, guard, original, updates, applied, identity, snapshot = _real_guard_inputs(monkeypatch)
    rows = []
    def row(_model, ids):
        rows.append(ids)
        if len(rows) == 1:
            updates["opposite"].view(-1)[0] = torch.nextafter(
                updates["opposite"].view(-1)[0], torch.tensor(float("inf")))
            assert torch.equal((original + updates["opposite"]).view(torch.int32),
                               (original + torch.full_like(original, 3 / 16)).view(torch.int32))
        return np.zeros(4, dtype="<f4").tobytes()
    monkeypatch.setattr(v1, "_logit_row", row)
    with pytest.raises(owner.OwnerError, match="intended update"):
        writer._evaluation_conditions(tmp_path, identity, snapshot, guard, updates, applied, v1, v2, evidence, gate)
    assert rows == [(1,), (1,), (1,)] and torch.equal(model.weight, original)
    assert not (tmp_path / "opposite.f32").exists() and not (tmp_path / "terminal.json").exists()
