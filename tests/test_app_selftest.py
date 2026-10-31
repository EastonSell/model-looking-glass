"""CPU contract checks do not substitute for the explicitly gated CUDA integration."""
import json
import subprocess
import sys

import pytest
import torch

from looking_glass import app_selftest as subject


def test_cpu_prerequisite_never_reports_pass_or_constructs_fixture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(subject, "_checkpoint", lambda *args: pytest.fail("must not build before CUDA prerequisite"))
    report = subject.run_selftest()
    assert report["status"] == "skipped"
    assert report["stages"] == [{"stage": "cuda_prerequisite", "status": "skipped"}]


@pytest.mark.parametrize("device,budget", [("cpu", 1024), ("auto", 1024), ("cuda:-1", 1024), ("cuda:0", 0), ("cuda:0", True)])
def test_invalid_request_is_failed_not_skipped(device, budget):
    assert subject.run_selftest(device, budget)["status"] == "failed"


@pytest.mark.parametrize("status,code", [("passed", 0), ("failed", 1), ("skipped", 2)])
def test_cli_classification_and_stdout_is_only_json(monkeypatch, capsys, status, code):
    def synthetic(*args):
        print("synthetic dependency diagnostic")
        return {"status": status, "fixture": "mocked CLI classification; not CUDA execution"}
    monkeypatch.setattr(subject, "run_selftest", synthetic)
    assert subject.main([]) == code
    streams = capsys.readouterr()
    assert json.loads(streams.out)["status"] == status
    assert "diagnostic" in streams.err


def _synthetic_prerequisite(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 8 * 1024**3))


def test_stage_failure_stops_remaining_family_and_removes_temporary_directory(monkeypatch):
    _synthetic_prerequisite(monkeypatch)
    roots, called = [], []
    def failure(root, family, device, budget, stages):
        roots.append(root); called.append(family)
        (root / "synthetic-marker").write_text("test only")
        raise RuntimeError("/private/path should not be reported")
    monkeypatch.setattr(subject, "_family", failure)
    report = subject.run_selftest()
    assert report["status"] == "failed" and called == ["gpt2"]
    assert not roots[0].exists()
    assert "/private/path" not in json.dumps(report)


def test_synthetic_success_requires_all_five_families_and_selected_budget(monkeypatch):
    _synthetic_prerequisite(monkeypatch)
    calls = []
    def synthetic(root, family, device, budget, stages):
        calls.append((family, str(device), budget))
        stages.extend({"family": family, "stage": stage, "status": "passed",
                       "test_only": "synthetic ledger; not GPU execution"} for stage in subject.EXPECTED_STAGES)
    monkeypatch.setattr(subject, "_family", synthetic)
    report = subject.run_selftest(max_vram_mib=512)
    assert report["status"] == "passed"
    assert len(report["stages"]) == 25
    assert calls == [(family, "cuda:0", 512 * 1024**2) for family in ("gpt2", "t5", "distilbert", "vit", "llama")]


@pytest.mark.parametrize("family", ["gpt2", "t5", "distilbert", "vit", "llama"])
def test_synthetic_local_assets_are_safe_and_deterministic(tmp_path, family):
    from looking_glass.app_runtime import LocalModelRuntime
    one, two = tmp_path / "one", tmp_path / "two"
    before = torch.random.get_rng_state().clone()
    subject._checkpoint(one, family); subject._checkpoint(two, family)
    assert torch.equal(before, torch.random.get_rng_state())
    assert (one / "model.safetensors").read_bytes() == (two / "model.safetensors").read_bytes()
    assert not list(one.glob("*.bin"))
    runtime = LocalModelRuntime()
    runtime.load(str(one), device="cpu")
    prompt = "hello [MASK]" if family == "distilbert" else "hello world"
    result = runtime.run(prompt, {"max_new_tokens": 2}, subject._image() if family == "vit" else None)
    assert result["model"]["family"] == family
    if family == "llama":
        assert result["model"]["config_summary"]["rope_scaling"] == {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8}
    assert result["coverage"]["forward_count"] == (1 if family in ("distilbert", "vit") else 2)
    runtime.unload()


def test_cpu_cli_subprocess_is_machine_readable():
    if torch.cuda.is_available(): pytest.skip("CPU prerequisite behavior only")
    result = subprocess.run([sys.executable, "-B", "-m", "looking_glass.app_selftest"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "skipped"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA device/build required; CPU policy tests are not GPU validation")
def test_real_cuda_end_to_end_selftest():
    report = subject.run_selftest()
    if report["status"] == "skipped": pytest.skip(report["reason"])
    assert report["status"] == "passed", report
    assert len(report["stages"]) == 25
    assert all(stage["status"] == "passed" for stage in report["stages"])


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "failed", "skipped", "reversed"])
def test_incomplete_or_unsuccessful_ledger_never_passes(monkeypatch, mutation):
    _synthetic_prerequisite(monkeypatch)
    def synthetic(root, family, device, budget, stages):
        rows = [{"family": family, "stage": stage, "status": "passed"} for stage in subject.EXPECTED_STAGES]
        if mutation == "empty": rows = []
        elif mutation == "missing": rows.pop()
        elif mutation == "duplicate": rows.append(dict(rows[-1]))
        elif mutation in ("failed", "skipped"): rows[-1]["status"] = mutation
        elif mutation == "reversed": rows.reverse()
        stages.extend(rows)
    monkeypatch.setattr(subject, "_family", synthetic)
    report = subject.run_selftest()
    assert report["status"] == "failed"
    assert report["stages"][-1]["status"] == "failed"


def test_boundary_comparison_rejects_changed_bits_missing_and_duplicate_coordinates():
    from types import SimpleNamespace
    expected = {"model.layers.0.mlp": torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)}
    metadata = {"module_path": "model.layers.0.mlp", "batch_positions": [0], "token_positions": [0, 1]}
    record = SimpleNamespace(value=expected[metadata["module_path"]].clone(), metadata=lambda: metadata)
    subject._verify_boundaries([record], expected, "llama")
    record.value[0, 1, 7] += 1
    with pytest.raises(RuntimeError, match="boundary differs"): subject._verify_boundaries([record], expected, "llama")
    record.value = expected[metadata["module_path"]].clone()
    with pytest.raises(RuntimeError, match="coordinates"): subject._verify_boundaries([record, record], expected, "llama")
    with pytest.raises(RuntimeError, match="coordinates"): subject._verify_boundaries([], expected, "llama")
    metadata["token_positions"] = [1, 0]
    with pytest.raises(RuntimeError, match="coordinates"): subject._verify_boundaries([record], expected, "llama")


def test_llama_fixture_configuration_is_explicit_static_llama3_rope_and_gqa(tmp_path):
    subject._checkpoint(tmp_path / "llama", "llama")
    config = json.loads((tmp_path / "llama" / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert {name: config[name] for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "vocab_size")} == dict(hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32, vocab_size=8)
    assert config["rope_scaling"] == {"rope_type": "llama3", "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8}
    assert config["eos_token_id"] is None and config["bos_token_id"] == 2


def test_llama_fixture_cpu_full_logits_every_boundary_and_hook_cleanup(tmp_path):
    """Real CPU evidence only; does not satisfy the actual-CUDA ledger."""
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    from looking_glass.llama_trace import trace_llama_causal_lm
    path = tmp_path / "llama"
    subject._checkpoint(path, "llama")
    model = LlamaForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                            attn_implementation="eager", torch_dtype=torch.float32).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    original_inv_freq = model.model.rotary_emb.inv_freq.detach().clone()
    for length in (2, 8, 9, 16, 3):
        ids = tokenizer(" ".join(["hello", "world"] * length), return_tensors="pt")["input_ids"][:, :length]
        mask = torch.ones_like(ids)
        args = dict(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        with torch.inference_mode(): baseline = model(**args).logits
        assert subject._bits_equal(original_inv_freq, model.model.rotary_emb.inv_freq)
        expected, handles = {}, []
        for layer in range(2):
            for suffix in (".self_attn", ".mlp", ""):
                name = f"model.layers.{layer}{suffix}"
                def observe(_module, _args, output, name=name):
                    expected[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
                handles.append(model.get_submodule(name).register_forward_hook(observe))
        try:
            with torch.inference_mode(): hooked = model(**args).logits
        finally:
            for handle in handles: handle.remove()
        assert subject._bits_equal(baseline, hooked)
        assert subject._bits_equal(original_inv_freq, model.model.rotary_emb.inv_freq)
        for _ in range(2):
            result = trace_llama_causal_lm(model, ids, mask, run_id="fixture-cpu-only", layers=(0, 1),
                modules=("attention", "mlp", "block"), max_events=6,
                max_trace_bytes=1024**2, max_tensor_bytes=1024**2, max_logits_bytes=1024**2)
            assert subject._bits_equal(baseline, result.output.logits)
            subject._verify_boundaries(result.events, expected, "llama")
            assert len(result.events) == 6
            assert {record.event.module_kind for record in result.events} == {"llama_attention", "llama_mlp", "llama_block"}
            assert all(tuple(record.value.shape) == (1, length, 8) for record in result.events)
            assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
            assert subject._bits_equal(original_inv_freq, model.model.rotary_emb.inv_freq)
