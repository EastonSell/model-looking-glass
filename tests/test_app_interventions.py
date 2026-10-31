"""Real tiny CPU fixed-forward experiments; CUDA cases require real hardware."""
import copy

import pytest
import torch

from looking_glass.app_runtime import LocalModelRuntime
from looking_glass.app_storage import TraceStore
from test_app_runtime import local_checkpoint


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load(str(path), device="cpu")
    result = runtime.run("hello world", {"max_new_tokens": 2})
    store = TraceStore(tmp_path / "exports"); store.publish(result)
    source = store.intervention_source(0, 1)
    return runtime, store, source, path


@pytest.mark.parametrize("operation,feature,value", [("zero", None, None), ("add", 2, 1.), ("replace", 2, .5), ("add", 1, 0.)])
def test_real_fixed_forward_landing_controls_and_unchanged_source(fixture, operation, feature, value):
    runtime, store, source, _ = fixture
    params = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    original = source["reference_value"].clone()
    original_summary = store.summary()
    result = runtime.intervene_fixed_forward(runtime.capture_binding(), source, operation, feature, value)
    assert result["run_kind"] == "intervention_comparison"
    assert all(result["experiment"]["controls"].values())
    assert [e.metadata()["experiment_phase"] for e in result["events"]] == ["null_before", "null_after", "intervention_before", "intervention_after"]
    assert torch.equal(original.view(torch.int32), source["reference_value"].view(torch.int32))
    assert store.summary() == original_summary
    assert all(torch.equal(v.view(torch.uint8), runtime.model.state_dict()[k].view(torch.uint8)) for k, v in params.items())
    last = result["events"][-1].value.flatten()
    expected = original.clone()
    if operation == "zero": expected.zero_()
    elif operation == "add": expected[feature] += value
    else: expected[feature] = value
    assert torch.equal(last.view(torch.int32), expected.view(torch.int32))
    if value == 0.: assert not result["experiment"]["comparison"]["selected_vector_changed"]
    assert runtime.capture_binding() is not None and runtime.info()["state"] == "ready"
    target = TraceStore(store.export_root / "experiments"); target.publish(result)
    assert target.vectors(3)["vectors"][0]["coordinate"]["status"] == "known"


def test_replace_original_is_exact_noop(fixture):
    runtime, _, source, _ = fixture
    result = runtime.intervene_fixed_forward(runtime.capture_binding(), source, "replace", 3, float(source["reference_value"][3]))
    assert not result["experiment"]["comparison"]["selected_vector_changed"]
    assert result["experiment"]["full_logits"]["bitwise_equal"]


@pytest.mark.parametrize("mutation", ["unknown", "path", "mask", "numerics", "token", "reference", "budget", "nan", "feature"])
def test_invalid_preflight_never_forwards_or_quarantines(fixture, monkeypatch, mutation):
    runtime, _, source, _ = fixture
    operation, feature, value, maximum = "add", 0, 1., 128 * 1024**2
    if mutation == "unknown": source["coordinate"]["status"] = "unknown"
    elif mutation == "path": source["coordinate"]["module_path"] = "transformer.wte"
    elif mutation == "mask": source["step"].pop("input_attention_mask")
    elif mutation == "numerics": source["model"]["execution_numerics"] = {}
    elif mutation == "token": source["coordinate"]["token_id"] = 100
    elif mutation == "reference": source["reference_value"] = torch.zeros(2)
    elif mutation == "budget": maximum = 1
    elif mutation == "nan": value = float("nan")
    else: feature = 999
    import looking_glass.app_interventions as module
    monkeypatch.setattr(module, "execute", lambda *a, **k: pytest.fail("preflight must reject"))
    binding = runtime.capture_binding()
    with pytest.raises(ValueError): runtime.intervene_fixed_forward(binding, source, operation, feature, value, maximum)
    assert runtime.capture_binding() is binding and runtime.info()["state"] == "ready"


def test_stale_imported_reload_and_unload_bindings_rejected(fixture):
    runtime, _, source, path = fixture
    old = runtime.capture_binding()
    for binding in (None, object()):
        with pytest.raises(ValueError): runtime.intervene_fixed_forward(binding, source, "zero")
    runtime.load(str(path), device="cpu")
    with pytest.raises(ValueError): runtime.intervene_fixed_forward(old, source, "zero")
    runtime.unload()
    assert runtime.capture_binding() is None


def test_mutated_reference_quarantines_and_preserves_source(fixture):
    runtime, store, source, _ = fixture
    original_summary = store.summary()
    source["reference_value"][0] += 1
    with pytest.raises(ValueError, match="original source"):
        runtime.intervene_fixed_forward(runtime.capture_binding(), source, "zero")
    assert runtime.capture_binding() is None and runtime.info()["state"] == "quarantined"
    assert store.summary() == original_summary


@pytest.mark.parametrize("trial", [2, 4])
def test_null_or_restoration_logit_mismatch_quarantines(fixture, monkeypatch, trial):
    runtime, _, source, _ = fixture
    original = runtime.model.forward
    calls = []
    def changed(*args, **kwargs):
        result = original(*args, **kwargs); calls.append(None)
        if len(calls) == trial: result.logits = result.logits + 1
        return result
    import looking_glass.app_interventions as module
    execute = module.execute
    def after_preflight(owner, request):
        monkeypatch.setattr(runtime.model, "forward", changed)
        return execute(owner, request)
    monkeypatch.setattr(module, "execute", after_preflight)
    with pytest.raises(ValueError, match="full logits"):
        runtime.intervene_fixed_forward(runtime.capture_binding(), source, "zero")
    assert runtime.capture_binding() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device/build required, no mocked GPU validation")
def test_real_cuda_fixed_forward_controls(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load(str(path), device="cuda:0", max_vram_bytes=1024**3)
    store = TraceStore(tmp_path / "exports"); store.publish(runtime.run("hello world", {"max_new_tokens": 1}))
    source = store.intervention_source(0, 1)
    prior = torch.cuda.get_per_process_memory_fraction(0)
    result = runtime.intervene_fixed_forward(runtime.capture_binding(), source, "add", 1, 0.)
    assert result["gpu_memory"]["synchronized"] and result["experiment"]["full_logits"]["bitwise_equal"]
    assert torch.cuda.get_per_process_memory_fraction(0) == prior
    assert all(e.value.device.type == "cpu" for e in result["events"])
    runtime.unload()


def test_single_feature_add_preserves_other_negative_zero_bits():
    from looking_glass.app_interventions import replacement
    fresh = torch.tensor([-0.0, 2.0, -0.0], dtype=torch.float32)
    source, desired = replacement(fresh, "add", 1, 1.)
    assert desired[1] == 3
    assert torch.equal(desired.view(torch.int32)[[0, 2]], fresh.view(torch.int32)[[0, 2]])
    assert int(desired.view(torch.int32)[0]) == -2147483648
    assert source.data_ptr() != fresh.data_ptr() and desired.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("event,vector", [(1, 0), (2, 1), (6, 2)])
def test_other_supported_boundaries_and_later_saved_prefix(fixture, event, vector):
    runtime, store, _, _ = fixture
    source = store.intervention_source(event, vector)
    result = runtime.intervene_fixed_forward(runtime.capture_binding(), source, "add", 1, 0.)
    assert result["experiment"]["full_logits"]["bitwise_equal"]
    assert result["steps"][0]["input_token_ids"] == source["step"]["input_token_ids"]
    assert result["steps"][0]["input_attention_mask"] == source["step"]["input_attention_mask"]
    assert result["experiment"]["readout_position"] == len(source["step"]["input_token_ids"]) - 1


@pytest.mark.parametrize("mutation", ["eager", "hook", "forward", "buffer", "autocast"])
def test_modified_execution_contract_rejected_before_forward(fixture, monkeypatch, mutation):
    from contextlib import nullcontext
    runtime, _, source, _ = fixture
    context = nullcontext()
    handle = None
    if mutation == "eager": runtime.model.config._attn_implementation = "sdpa"
    elif mutation == "hook": handle = runtime.model.register_forward_hook(lambda *args: pytest.fail("must not forward"))
    elif mutation == "forward": monkeypatch.setattr(runtime.model, "forward", lambda *args, **kwargs: pytest.fail("must not forward"))
    elif mutation == "buffer": runtime.model.transformer.h[0].attn.masked_bias = torch.tensor(-1e4, dtype=torch.float64)
    else: context = torch.autocast("cpu", dtype=torch.bfloat16)
    binding = runtime.capture_binding()
    try:
        with context, pytest.raises(ValueError): runtime.intervene_fixed_forward(binding, source, "zero")
    finally:
        if handle is not None: handle.remove()
    assert runtime.capture_binding() is binding and runtime.info()["state"] == "ready"
