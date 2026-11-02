"""Reference lifecycle and authenticated HTTP tests; no model/GPU execution."""
import copy
import json
import threading

import pytest
import torch

from looking_glass.app import AppState, create_server
from test_app_http import call
from test_app_intervention_http import ExperimentRuntime


class ReferenceRuntime(ExperimentRuntime):
    def load(self, path, **kwargs):
        super().load(path, **kwargs)
        self.loaded.update(device="cpu", dtype="float32", execution_numerics={"dtype": "float32", "test": "synthetic"})
        return self.info()

    def run(self, prompt, settings, image_bytes=None):
        if prompt == "fail":
            raise RuntimeError("synthetic failed forward")
        result = super().run(prompt, settings, image_bytes)
        result["model"] = copy.deepcopy(self.loaded)
        result["prompt"] = prompt
        result["events"][0].metadata().update(output_path=[0], device="cpu", saved_device="cpu", dtype="torch.float32", saved_dtype="torch.float32")
        if prompt == "second":
            result["events"][0].value.add_(1)
        return result


@pytest.fixture
def reference_state(tmp_path):
    state = AppState(runtime=ReferenceRuntime(), export_root=tmp_path / "exports", max_trace_bytes=1024)
    state.post("load", {"path": "fixture"})
    state.post("run", {"prompt": "first", "settings": {"max_trace_bytes": 512}})
    return state


def pin(state):
    return state.post("pin-reference", {"capture_id": state.status()["run"]["capture_id"], "event": 0, "vector": 0,
                                        "expected_reference_id": (state.status()["reference"] or {}).get("reference_id")})


def compare(state, reference, **overrides):
    args = dict(reference_id=reference["reference_id"], capture_id=state.status()["run"]["capture_id"], event=0, vector=0)
    args.update(overrides)
    return state.compare_reference(**args)


def test_pin_survives_second_prompt_and_comparison_preserves_both(reference_state):
    state = reference_state
    original = state.store.tensor(0)
    reference = pin(state)
    assert reference["total_bytes"] == 16 and reference["prompt"] == "first"
    state.post("run", {"prompt": "second", "settings": {"max_trace_bytes": 512}})
    current = state.store.tensor(0)
    paired = compare(state, reference)
    assert paired["reference_values"] == original["values"]
    assert paired["current_values"] == current["values"]
    assert paired["deltas"] == [1., 1., 1., 1.]
    assert paired["metrics"]["raw_bit_changed_count"] == 4
    assert state.reference_tensor(reference["reference_id"])["raw_bits"] == original["raw_bits"]
    assert state.store.tensor(0) == current
    assert state.status()["reference"]["reference_id"] == reference["reference_id"]


def test_stale_capture_and_reference_revisions_reject(reference_state):
    state = reference_state; first = pin(state); old_capture = state.status()["run"]["capture_id"]
    second = pin(state)
    assert first["reference_id"] != second["reference_id"]
    with pytest.raises(ValueError, match="stale"):
        state.post("pin-reference", {"capture_id": state.status()["run"]["capture_id"], "event": 0, "vector": 0,
                                     "expected_reference_id": first["reference_id"]})
    with pytest.raises(ValueError, match="stale"):
        state.reference_tensor(first["reference_id"])
    with pytest.raises(ValueError, match="stale"):
        state.post("clear-reference", {"reference_id": first["reference_id"]})
    state.post("run", {"prompt": "second", "settings": {"max_trace_bytes": 512}})
    with pytest.raises(ValueError, match="stale"):
        compare(state, second, capture_id=old_capture)
    before = state.status()["reference"]
    with pytest.raises(ValueError, match="stale"):
        state.post("pin-reference", {"capture_id": old_capture, "event": 0, "vector": 0})
    assert state.status()["reference"] == before


@pytest.mark.parametrize("action", ["unload", "reload", "open", "quarantine", "clear"])
def test_invalid_authority_keeps_reference_readable(reference_state, action):
    state = reference_state; reference = pin(state)
    before = state.reference_tensor(reference["reference_id"])
    if action == "reload":
        state.post("load", {"path": "same-path-new-owner"})
        state.post("run", {"prompt": "second", "settings": {"max_trace_bytes": 512}})
    elif action == "open":
        saved = state.post("save", {"name": "original"})
        state.post("open", {"path": saved["path"]})
    elif action == "quarantine":
        state.runtime.loaded["state"] = "quarantined"
    else:
        state.post(action, {})
    assert not state.status()["reference"]["ready_for_compare"]
    assert state.reference_tensor(reference["reference_id"]) == before
    capture = state.status()["run"]
    with pytest.raises(ValueError):
        state.compare_reference(reference["reference_id"], capture["capture_id"] if capture else "none", 0, 0)
    state.post("clear-reference", {"reference_id": reference["reference_id"]})
    assert state.status()["reference"] is None


def test_run_budget_rejects_before_losing_current_capture(reference_state):
    state = reference_state; reference = pin(state); before = state.status()["run"]
    with pytest.raises(ValueError, match="reference|allowance|budget"):
        state.post("run", {"prompt": "second", "settings": {"max_trace_bytes": 1024}})
    assert state.status()["run"] == before
    assert state.status()["reference"]["reference_id"] == reference["reference_id"]


def test_normal_large_server_budget_still_allows_a_small_reference(reference_state):
    state = reference_state; state.max_trace_bytes = 512 * 1024**2
    assert pin(state)["total_bytes"] == 16


def test_pin_admission_atomicity_and_experiment_remainder(reference_state):
    state = reference_state; reference = pin(state)
    source_bytes = state.store.summary()["total_bytes"]
    result = state.post("intervene", {"capture_id": state.status()["run"]["capture_id"], "event": 0, "vector": 0, "operation": "zero"})
    assert state.runtime.last_budget == 1024 - source_bytes - reference["total_bytes"]
    state.max_trace_bytes = source_bytes + state.experiment_store.summary()["total_bytes"] + 15
    before = state.reference_tensor(reference["reference_id"])
    with pytest.raises(ValueError): pin(state)
    assert state.reference_tensor(reference["reference_id"]) == before
    assert state.status()["experiment"]["experiment_id"] == result["experiment_id"]


def test_failed_run_keeps_reference_and_save_does_not_include_it(reference_state):
    state = reference_state; reference = pin(state)
    saved = state.post("save", {"name": "source"})
    from pathlib import Path
    manifest = json.loads((Path(saved["path"]) / "manifest.json").read_text())
    assert "reference_id" not in json.dumps(manifest)
    with pytest.raises(RuntimeError, match="failed"):
        state.post("run", {"prompt": "fail", "settings": {"max_trace_bytes": 512}})
    assert state.status()["run"] is None
    assert state.reference_tensor(reference["reference_id"])["values"] == [0., 1., 2., 3.]


def test_oversized_open_preserves_capture_reference_and_restores_limit(reference_state, tmp_path):
    from looking_glass.app_storage import TraceStore
    from test_app_storage import coordinate_result
    state = reference_state; reference = pin(state)
    other = TraceStore(tmp_path / "larger"); other.publish(coordinate_result())
    saved = other.save("two-vectors")
    before = state.store.tensor(0); limit = state.store.max_bytes
    state.max_trace_bytes = state.store.summary()["total_bytes"] + reference["total_bytes"]
    with pytest.raises(ValueError): state.post("open", {"path": saved["path"]})
    assert state.store.tensor(0) == before and state.store.max_bytes == limit
    assert state.status()["reference"]["reference_id"] == reference["reference_id"]


def test_authenticated_reference_routes(reference_state):
    state = reference_state; httpd = create_server(state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    server = state, httpd
    try:
        assert call(server, "/api/reference", token=False)[0] == 403
        assert call(server, "/api/pin-reference", "POST", {}, token=False)[0] == 403
        code, body, _ = call(server, "/api/pin-reference", "POST", {"capture_id": state.status()["run"]["capture_id"], "event": 0, "vector": 0})
        assert code == 200, body
        reference = json.loads(body)
        capture = state.status()["run"]["capture_id"]
        query = f"reference_id={reference['reference_id']}&capture_id={capture}&event=0&vector=0"
        assert call(server, "/api/reference-tensor?reference_id=" + reference["reference_id"])[0] == 200
        code, body, _ = call(server, "/api/compare-vector?" + query)
        assert code == 200 and json.loads(body)["metrics"]["l2"] == 0
        for route in ("/api/reference-tensor", "/api/compare-vector?reference_id=" + reference["reference_id"]):
            assert call(server, route)[0] == 400
        assert call(server, "/api/compare-vector?" + query + "&capture_id=duplicate")[0] == 400
        assert call(server, "/api/clear-reference", "POST", {"reference_id": reference["reference_id"]})[0] == 200
    finally:
        httpd.shutdown(); thread.join(5); httpd.server_close()
        assert not thread.is_alive()


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required; CPU reference tests are not GPU execution")
def test_reference_from_real_cuda_capture_when_available(tmp_path, family):
    from looking_glass.app_runtime import LocalModelRuntime
    from looking_glass.app_selftest import _checkpoint
    checkpoint = tmp_path / family; _checkpoint(checkpoint, family)
    state = AppState(runtime=LocalModelRuntime(), export_root=tmp_path / "exports")
    try:
        state.post("load", {"path": str(checkpoint), "device": "cuda:0", "max_vram_bytes": 1024**3})
        state.post("run", {"prompt": "hello", "settings": {"max_new_tokens": 1}})
        reference = pin(state)
        original = state.reference_tensor(reference["reference_id"])
        state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
        current = state.store.tensor(0, 0)
        pair = compare(state, reference)
        assert pair["reference_raw_bits"] == original["raw_bits"]
        assert pair["current_raw_bits"] == current["raw_bits"]
        assert pair["deltas"] == [float(b) - float(a) for a, b in zip(original["values"], current["values"])]
        assert pair["reference"]["model"]["device"] == pair["current"]["model"]["device"] == "cuda:0"
    finally:
        state.runtime.unload()
