"""RAM-only onboarding demo; no model download or checkpoint files needed."""
import builtins
import io
import os
import tempfile
import socket

import pytest
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from looking_glass.app_runtime import LocalModelRuntime
from looking_glass.app_storage import TraceStore


def test_demo_in_memory_without_loader_saver_network_or_file_writes(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("external I/O forbidden for demo")
    monkeypatch.setattr(GPT2LMHeadModel, "from_pretrained", forbidden)
    monkeypatch.setattr(GPT2LMHeadModel, "save_pretrained", forbidden)
    monkeypatch.setattr(PreTrainedTokenizerFast, "from_pretrained", forbidden)
    monkeypatch.setattr(PreTrainedTokenizerFast, "save_pretrained", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", forbidden)
    monkeypatch.setattr(tempfile, "mkstemp", forbidden)
    original = builtins.open
    def read_only(file, mode="r", *args, **kwargs):
        assert not any(x in mode for x in "wax+")
        return original(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", read_only)
    original_io = io.open
    def io_read_only(file, mode="r", *args, **kwargs):
        assert not any(x in mode for x in "wax+")
        return original_io(file, mode, *args, **kwargs)
    monkeypatch.setattr(io, "open", io_read_only)
    original_os = os.open
    def os_read_only(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return original_os(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", os_read_only)
    runtime = LocalModelRuntime()
    info = runtime.load_demo()
    assert info["path"] is None and info["device"] == "cpu"
    assert info["provenance"]["kind"] == "synthetic_untrained_demo"
    assert "untrained" in info["warning"]
    assert runtime.tokenizer.decode(runtime.tokenizer("Hello 🧪 漢字", add_special_tokens=False)["input_ids"]) == "Hello 🧪 漢字"
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in runtime.model.parameters())


def test_demo_rng_and_same_environment_determinism():
    state = torch.random.get_rng_state().clone()
    one = LocalModelRuntime(); one.load_demo()
    assert torch.equal(state, torch.random.get_rng_state())
    two = LocalModelRuntime(); two.load_demo()
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(v.view(torch.uint8), two.model.state_dict()[k].view(torch.uint8)) for k, v in one.model.state_dict().items())
    a = one.run("Hi", {"max_new_tokens": 2})
    b = two.run("Hi", {"max_new_tokens": 2})
    assert a["generated_token_ids"] == b["generated_token_ids"]
    assert all(torch.equal(x.value.view(torch.int32), y.value.view(torch.int32)) for x, y in zip(a["events"], b["events"]))
    assert a["model"]["provenance"]["kind"] == "synthetic_untrained_demo"
    assert "untrained" in a["warning"]


def test_demo_capture_navigation_and_intervention(tmp_path):
    runtime = LocalModelRuntime(); runtime.load_demo()
    store = TraceStore(tmp_path / "exports")
    store.publish(runtime.run("Hello", {"max_new_tokens": 1}))
    assert store.vectors(0)["vectors"][0]["coordinate"]["status"] == "known"
    source = store.intervention_source(0, 0)
    result = runtime.intervene_fixed_forward(runtime.capture_binding(), source, "add", 0, 0.)
    assert all(result["experiment"]["controls"].values())
    assert result["experiment"]["full_logits"]["bitwise_equal"]
    assert result["model"]["provenance"]["kind"] == "synthetic_untrained_demo"


def test_demo_binding_reload_unload_and_preflight_budget():
    runtime = LocalModelRuntime(); runtime.load_demo()
    original = runtime.capture_binding()
    model = runtime.model
    runtime.max_model_bytes = 1
    with pytest.raises(ValueError, match="byte limit"): runtime.load_demo()
    assert runtime.model is model and runtime.capture_binding() is original
    runtime.max_model_bytes = 1024**2
    runtime.load_demo()
    assert runtime.capture_binding() is not original
    runtime.unload()
    assert runtime.capture_binding() is None and runtime.model is None


def test_demo_quarantine_is_not_bypassed():
    runtime = LocalModelRuntime(); runtime.load_demo(); runtime._quarantine()
    with pytest.raises(RuntimeError, match="quarantined"): runtime.load_demo()
    assert runtime.capture_binding() is None


def test_demo_construction_failure_quarantines_after_retirement(monkeypatch):
    runtime = LocalModelRuntime(); runtime.load_demo()
    original = GPT2LMHeadModel.__init__
    def fail_real(self, *args, **kwargs):
        if torch.empty(()).device.type != "meta": raise RuntimeError("synthetic constructor failure")
        original(self, *args, **kwargs)
    monkeypatch.setattr(GPT2LMHeadModel, "__init__", fail_real)
    with pytest.raises(RuntimeError, match="constructor failure"): runtime.load_demo()
    assert runtime.capture_binding() is None and runtime.info()["state"] == "quarantined"
