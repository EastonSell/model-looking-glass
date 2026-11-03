"""Cooperative stops occur only after complete controlled operations."""
import threading

import pytest
import torch

from looking_glass.app_cross_patch import patch
from looking_glass.app_sensitivity import run_sensitivity
from looking_glass.app_sensitivity_repeat import run_repeated_sensitivity
from test_app_cross_patch import pair
from test_app_sensitivity import request, state


def snapshot(owner):
    reference = owner.post("pin-reference", {"capture_id": owner._capture_id, "event": 0, "vector": 1,
                                             "expected_reference_id": None})
    comparison = owner.post("intervene", {"capture_id": owner._capture_id, "event": 0, "vector": 1,
                                         "operation": "zero"})
    return {"capture": owner.store.summary(), "comparison": owner.experiment_store.summary(),
            "capture_bits": [v.view(torch.int32).clone() for v in owner.store._values],
            "comparison_bits": [v.view(torch.int32).clone() for v in owner.experiment_store._values],
            "reference_bits": owner.reference._value.view(torch.int32).clone(),
            "weights": {k: v.view(torch.int32).clone() for k, v in owner.runtime.model.state_dict().items()},
            "binding": owner._live_binding, "reference_id": reference["reference_id"],
            "experiment_id": comparison["experiment_id"]}


def assert_preserved(owner, saved):
    assert owner.store.summary() == saved["capture"] and owner.experiment_store.summary() == saved["comparison"]
    assert owner._reference_id == saved["reference_id"] and owner._experiment_id == saved["experiment_id"]
    assert owner.runtime.capture_binding() is saved["binding"] is owner._live_binding
    assert owner.runtime.info()["state"] == "ready"
    assert torch.equal(owner.reference._value.view(torch.int32), saved["reference_bits"])
    assert all(torch.equal(v.view(torch.int32), b) for v, b in zip(owner.store._values, saved["capture_bits"]))
    assert all(torch.equal(v.view(torch.int32), b) for v, b in zip(owner.experiment_store._values, saved["comparison_bits"]))
    assert all(torch.equal(v.view(torch.int32), saved["weights"][k]) for k, v in owner.runtime.model.state_dict().items())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in owner.runtime.model.modules())


def test_single_prestart_stop_skips_all_without_forward(state, monkeypatch):
    saved = snapshot(state)
    stop = threading.Event(); stop.set()
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", lambda *a, **k: pytest.fail("no forward expected"))
    result = run_sensitivity(state, request(state, layers=[0]), cancel_event=stop)
    assert result["status"] == "cancelled" and result["completed_runs"] == result["completed_model_forwards"] == 0
    assert all(row["status"] == "skipped" and "request" in row["reason"] for row in result["results"])
    assert_preserved(state, saved)


@pytest.mark.parametrize("sites, expected_status, expected_rows", [
    (["attention", "mlp", "block"], "cancelled", ["ok", "skipped", "skipped"]),
    (["attention"], "complete", ["ok"]),
])
def test_stop_during_single_cell_finishes_controls_and_handles_last_cell_race(state, monkeypatch, sites, expected_status, expected_rows):
    import looking_glass.app_interventions as engine
    saved = snapshot(state)
    stop, calls = threading.Event(), []
    execute = engine.execute

    def controlled(owner, validated):
        forward = owner.model.forward
        def signal(*args, **kwargs):
            calls.append(None)
            if len(calls) == 1:
                stop.set()
            return forward(*args, **kwargs)
        with monkeypatch.context() as local:
            local.setattr(owner.model, "forward", signal)
            return execute(owner, validated)

    monkeypatch.setattr(engine, "execute", controlled)
    result = run_sensitivity(state, request(state, layers=[0], sites=sites), cancel_event=stop)
    assert len(calls) == 4 and result["status"] == expected_status
    assert [row["status"] for row in result["results"]] == expected_rows
    assert result["completed_runs"] == 1 and result["completed_model_forwards"] == 4
    assert all(result["results"][0]["controls"].values())
    assert_preserved(state, saved)


def test_real_control_failure_takes_precedence_over_stop(state, monkeypatch):
    import looking_glass.app_interventions as engine
    original = state.store.summary()
    stop, calls = threading.Event(), []
    execute = engine.execute

    def controlled(owner, validated):
        forward = owner.model.forward
        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            calls.append(None)
            if len(calls) == 2:
                stop.set()
                result.logits = result.logits + 1
            return result
        with monkeypatch.context() as local:
            local.setattr(owner.model, "forward", corrupt)
            return execute(owner, validated)

    monkeypatch.setattr(engine, "execute", controlled)
    result = run_sensitivity(state, request(state, layers=[0]), cancel_event=stop)
    assert result["status"] == "inconclusive"
    assert [row["status"] for row in result["results"]] == ["failed", "skipped", "skipped"]
    assert "null control" in result["results"][0]["reason"].lower()
    assert state.runtime.info()["state"] == "quarantined" and state.store.summary() == original


def repeat_request(owner, *, sites=None):
    return {"capture_id": owner._capture_id, "layers": [0], "sites": sites or ["attention", "mlp", "block"],
            "prompts": ["hello world", "world hello"]}


def test_repeat_prestart_stop_has_no_capture_forward(state, monkeypatch):
    saved = snapshot(state)
    stop = threading.Event(); stop.set()
    monkeypatch.setattr(state.runtime, "run", lambda *a, **k: pytest.fail("no capture expected"))
    report = run_repeated_sensitivity(state, repeat_request(state), cancel_event=stop)
    assert report["status"] == "cancelled" and report["completed_model_forwards"] == 0
    assert [row["status"] for row in report["results"]] == ["skipped", "skipped"]
    assert_preserved(state, saved)


def test_repeat_stop_during_capture_keeps_one_completed_forward_and_skips_cells(state, monkeypatch):
    saved = snapshot(state)
    stop = threading.Event()
    run = state.runtime.run
    def capture(*args, **kwargs):
        result = run(*args, **kwargs)
        stop.set()
        return result
    monkeypatch.setattr(state.runtime, "run", capture)
    report = run_repeated_sensitivity(state, repeat_request(state), cancel_event=stop)
    assert report["status"] == "cancelled" and report["completed_model_forwards"] == 1
    assert [row["status"] for row in report["results"]] == ["cancelled", "skipped"]
    assert all(row["status"] == "skipped" for row in report["results"][0]["report"]["results"])
    assert_preserved(state, saved)


def test_repeat_stop_after_first_cell_preserves_partial_report(state, monkeypatch):
    saved = snapshot(state)
    stop = threading.Event()
    execute = state.runtime.intervene_fixed_forward
    def cell(*args, **kwargs):
        result = execute(*args, **kwargs)
        stop.set()
        return result
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", cell)
    report = run_repeated_sensitivity(state, repeat_request(state), cancel_event=stop)
    assert report["status"] == "cancelled" and report["completed_model_forwards"] == 5
    assert [row["status"] for row in report["results"]] == ["cancelled", "skipped"]
    assert [row["status"] for row in report["results"][0]["report"]["results"]] == ["ok", "skipped", "skipped"]
    assert_preserved(state, saved)


def test_repeat_capture_failure_is_not_reported_as_cancellation(state, monkeypatch):
    original = state.store.summary()
    stop = threading.Event()
    def failed_capture(*args, **kwargs):
        stop.set()
        raise ValueError("synthetic capture failure")
    monkeypatch.setattr(state.runtime, "run", failed_capture)
    report = run_repeated_sensitivity(state, repeat_request(state), cancel_event=stop)
    assert report["status"] == "inconclusive"
    assert [row["status"] for row in report["results"]] == ["failed", "skipped"]
    assert "synthetic capture failure" in report["results"][0]["reason"]
    assert state.store.summary() == original


def test_repeat_stop_after_last_cell_is_complete(state, monkeypatch):
    saved = snapshot(state)
    stop, calls = threading.Event(), []
    execute = state.runtime.intervene_fixed_forward
    def cell(*args, **kwargs):
        result = execute(*args, **kwargs)
        calls.append(None)
        if len(calls) == 2:
            stop.set()
        return result
    monkeypatch.setattr(state.runtime, "intervene_fixed_forward", cell)
    report = run_repeated_sensitivity(state, repeat_request(state, sites=["attention"]), cancel_event=stop)
    assert report["status"] == "complete" and report["completed_runs"] == 2
    assert report["completed_model_forwards"] == 10
    assert_preserved(state, saved)


@pytest.mark.parametrize("stop_after_capture", [0, 1, 2])
def test_patch_stop_before_pair_reports_only_completed_captures(state, monkeypatch, stop_after_capture):
    saved = snapshot(state)
    data = pair(state)
    stop, calls = threading.Event(), []
    if stop_after_capture == 0:
        stop.set()
    run = state.runtime.run
    def capture(*args, **kwargs):
        result = run(*args, **kwargs)
        calls.append(None)
        if len(calls) == stop_after_capture:
            stop.set()
        return result
    monkeypatch.setattr(state.runtime, "run", capture)
    monkeypatch.setattr(state.runtime, "intervene_cross_prompt", lambda *a, **k: pytest.fail("patch must not begin"))
    report = patch(state, data, cancel_event=stop)
    assert report["status"] == "cancelled" and report["total_model_forwards"] == stop_after_capture
    assert report["model_forwards_upper_bound"] == 10 and "controls" not in report
    assert len(calls) == stop_after_capture
    assert_preserved(state, saved)


def test_patch_stop_during_pair_finishes_all_controls(state, monkeypatch):
    import looking_glass.app_interventions as engine
    saved = snapshot(state)
    data = pair(state)
    stop, calls = threading.Event(), []
    execute = engine.execute
    def paired(owner, validated):
        calls.append(None)
        if len(calls) == 1:
            stop.set()  # The first controlled operation has already begun.
        return execute(owner, validated)
    monkeypatch.setattr(engine, "execute", paired)
    report = patch(state, data, cancel_event=stop)
    assert stop.is_set() and len(calls) == 2
    assert report["status"] == "complete" and report["total_model_forwards"] == 10
    assert all(report["controls"].values())
    assert_preserved(state, saved)


def test_patch_control_failure_is_not_masked_by_stop(state, monkeypatch):
    import looking_glass.app_interventions as engine
    original = state.store.summary()
    data = pair(state)
    stop, calls = threading.Event(), []
    execute = engine.execute
    def controlled(owner, validated):
        forward = owner.model.forward
        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            calls.append(None)
            if len(calls) == 2:
                stop.set()
                result.logits = result.logits + 1
            return result
        with monkeypatch.context() as local:
            local.setattr(owner.model, "forward", corrupt)
            return execute(owner, validated)
    monkeypatch.setattr(engine, "execute", controlled)
    with pytest.raises(ValueError, match="null control"):
        patch(state, data, cancel_event=stop)
    assert stop.is_set() and state.runtime.info()["state"] == "quarantined"
    assert state.store.summary() == original
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in state.runtime.model.modules())


def test_llama_sweep_stop_preserves_rotary_controls(tmp_path, monkeypatch):
    import looking_glass.app_interventions as engine
    from looking_glass.app import AppState
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_llama_http import LLAMA3_SCALING, llama_checkpoint
    checkpoint = tmp_path / "llama"; llama_checkpoint(checkpoint, 2, LLAMA3_SCALING)
    owner = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports", download_root=tmp_path / "downloads")
    owner.post("load", {"path": str(checkpoint), "device": "cpu"})
    capture = owner.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    saved = owner.store.summary(); binding = owner._live_binding
    stop = threading.Event(); execute = engine.execute
    def cell(runtime, validated):
        result = execute(runtime, validated)
        stop.set()
        return result
    monkeypatch.setattr(engine, "execute", cell)
    result = run_sensitivity(owner, {"capture_id": capture["capture_id"], "event": 0, "vector": 1,
                                    "layers": [0], "sites": ["attention", "mlp", "block"]}, cancel_event=stop)
    assert result["status"] == "cancelled" and result["completed_runs"] == 1
    assert result["results"][0]["controls"]["static_rotary_state_unchanged"]
    assert result["results"][0]["controls"]["fixed_input_ids_mask_unchanged"]
    assert owner.store.summary() == saved and owner.runtime.capture_binding() is binding


def test_llama_patch_stop_during_pair_finishes_rotary_controls(tmp_path, monkeypatch):
    import looking_glass.app_interventions as engine
    from looking_glass.app import AppState
    from looking_glass.app_runtime import LocalModelRuntime
    from test_app_llama_http import LLAMA3_SCALING, llama_checkpoint
    checkpoint = tmp_path / "llama"; llama_checkpoint(checkpoint, 2, LLAMA3_SCALING)
    owner = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports", download_root=tmp_path / "downloads")
    owner.post("load", {"path": str(checkpoint), "device": "cpu"})
    owner.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    saved, binding, data = owner.store.summary(), owner._live_binding, pair(owner)
    stop = threading.Event(); execute = engine.execute; calls = []
    def controlled(runtime, validated):
        calls.append(None)
        if len(calls) == 1:
            stop.set()
        return execute(runtime, validated)
    monkeypatch.setattr(engine, "execute", controlled)
    report = patch(owner, data, cancel_event=stop)
    assert report["status"] == "complete" and report["total_model_forwards"] == 10
    assert report["controls"]["donor_static_rotary_state_unchanged"]
    assert report["controls"]["receiver_static_rotary_state_unchanged"]
    assert report["controls"]["fixed_input_ids_masks_unchanged"]
    assert owner.store.summary() == saved and owner.runtime.capture_binding() is binding
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in owner.runtime.model.modules())
