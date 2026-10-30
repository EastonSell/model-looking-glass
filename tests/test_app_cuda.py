"""Actual CUDA integration checks: never replaced by CPU or mocked GPU execution.

Requires a CUDA PyTorch build, a working NVIDIA GPU and adequate free headroom.
The ordinary CPU environment explicitly skips these two device integrations.
"""
import json
import math

import pytest
import torch

from looking_glass.app_runtime import LocalModelRuntime
from looking_glass.app_storage import TraceStore
from test_app_runtime import local_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device/build required; CPU/mocked policy checks are not GPU validation")


@pytest.mark.parametrize("family", ["gpt2", "t5"])
def test_real_cuda_same_device_capture_offload_and_export_roundtrip(tmp_path, family):
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    reserve = max(256 * 1024**2, math.ceil(total * 0.10))
    if free < reserve + 64 * 1024**2:
        pytest.skip("real GPU lacks the required free-memory/headroom prerequisite")
    path = tmp_path / family
    local_checkpoint(path, family)
    runtime = LocalModelRuntime()
    previous_fraction = torch.cuda.get_per_process_memory_fraction(device)
    expected = {}
    handles = []
    try:
        info = runtime.load(str(path), device="cuda:0", max_vram_bytes=1024**3)
        assert info["device"] == "cuda:0" and info["gpu_memory"]["synchronized"]
        assert torch.cuda.get_per_process_memory_fraction(device) == previous_fraction
        assert next(runtime.model.parameters()).device == device
        ids = runtime.tokenizer("hello world", return_tensors="pt")["input_ids"].to(device)
        if family == "gpt2":
            paths = [f"transformer.h.{layer}" + suffix for layer in range(2) for suffix in (".attn", ".mlp", "")]
        else:
            paths = [f"{stack}.block.{layer}" + suffix
                for stack, depth in (("encoder", 2), ("decoder", 1)) for layer in range(depth)
                for suffix in (".layer.0", ".layer.1" if stack == "encoder" else ".layer.2", "")]
        for name in paths:
            def hook(_module, _args, out, name=name):
                expected[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            handles.append(runtime.model.get_submodule(name).register_forward_hook(hook))
        with torch.inference_mode():
            if family == "gpt2":
                baseline = runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            else:
                decoder = torch.tensor([[0]], dtype=torch.int64, device=device)
                baseline = runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), decoder_input_ids=decoder,
                    decoder_attention_mask=torch.ones_like(decoder), use_cache=False).logits
        baseline_id = int(baseline[0, -1].argmax())
        torch.cuda.synchronize(device)
        expected = {name: value.cpu() for name, value in expected.items()}
        del baseline
        for handle in handles: handle.remove()
        handles.clear()
        result = runtime.run("hello world", {"max_new_tokens": 1})
        assert result["generated_token_ids"] == [baseline_id]
        assert result["gpu_memory"]["synchronized"]
        assert result["gpu_memory"]["peak_reserved_bytes"] <= result["gpu_memory"]["effective_limit_bytes"]
        assert torch.cuda.get_per_process_memory_fraction(device) == previous_fraction
        for capture in result["events"]:
            metadata = capture.metadata()
            assert metadata["device"] == "cuda:0" and metadata["saved_device"] == "cpu"
            assert capture.value.device.type == "cpu"
            wanted = expected[metadata["module_path"]]
            if family == "t5":
                wanted = wanted[metadata["batch_positions"][0], metadata["token_positions"][0]]
                assert metadata["app_offloaded_from_device"] == "cuda:0"
            assert torch.equal(capture.value.view(torch.int32), wanted.view(torch.int32))
        store = TraceStore(tmp_path / "exports")
        store.publish(result)
        original = [[store.tensor(i, vector=j, limit=4096)["raw_bits"]
                     for j in range(event.value.numel() // event.value.shape[-1])]
                    for i, event in enumerate(result["events"])]
        saved = store.save("cuda-trace")
        store.clear(); store.open(saved["path"])
        assert store.summary()["model"]["device"] == "cuda:0"
        assert store.summary()["gpu_memory"]["synchronized"]
        for i, vectors in enumerate(original):
            for j, bits in enumerate(vectors):
                assert store.tensor(i, vector=j, limit=4096)["raw_bits"] == bits
        # A second run must work with allocator cache retained from the first.
        again = runtime.run("hello world", {"max_new_tokens": 2})
        assert again["coverage"]["forward_count"] == 2
        assert again["gpu_memory"]["synchronized"]
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in runtime.model.modules())
        json.dumps({k: v for k, v in again.items() if k != "events"})
    finally:
        for handle in handles: handle.remove()
        if runtime.info() is not None and runtime.info()["state"] == "ready":
            runtime.unload()
