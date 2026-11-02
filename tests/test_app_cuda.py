"""Actual CUDA integration checks: never replaced by CPU or mocked GPU execution.

Requires a CUDA PyTorch build, a working NVIDIA GPU and adequate free headroom.
The ordinary CPU environment explicitly skips these four device integrations.
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


def test_real_distilbert_cuda_full_logits_captures_offload_and_export(tmp_path):
    from looking_glass.app_runtime import _cuda_operation
    from looking_glass.distilbert_trace import trace_distilbert_masked_lm
    from collections import Counter
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < max(256 * 1024**2, math.ceil(total * 0.10)) + 64 * 1024**2:
        pytest.skip("real GPU lacks required free-memory/headroom prerequisite")
    path = tmp_path / "distilbert"; local_checkpoint(path, "distilbert")
    runtime = LocalModelRuntime()
    previous = torch.cuda.get_per_process_memory_fraction(device)
    runtime.load(str(path), device=str(device), max_vram_bytes=1024**3)
    handles, expected = [], {}
    try:
        with _cuda_operation(device, 1024**3), torch.inference_mode():
            encoded = runtime.tokenizer("hello [MASK]", return_tensors="pt")
            ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
            with pytest.raises(ValueError, match="CPU required by default"):
                trace_distilbert_masked_lm(runtime.model, ids, mask, run_id="default-rejects-cuda")
            baseline = runtime.model(input_ids=ids, attention_mask=mask).logits.cpu()
            for layer in range(2):
                for role in ("attention", "ffn", "block"):
                    name = f"distilbert.transformer.layer.{layer}" + ("" if role == "block" else f".{role}")
                    def save(_module, _args, output, name=name):
                        expected[name] = (output[0] if type(output) is tuple else output).detach().cpu().clone()
                    handles.append(runtime.model.get_submodule(name).register_forward_hook(save))
            hooked = runtime.model(input_ids=ids, attention_mask=mask).logits.cpu()
            assert torch.equal(baseline.view(torch.int32), hooked.view(torch.int32))
            for handle in handles: handle.remove()
            handles.clear()
            traced = trace_distilbert_masked_lm(runtime.model, ids, mask, run_id="cuda-check",
                layers=(0, 1), tokens=tuple(range(ids.shape[1])), synchronous_cpu=False)
            assert traced.encoder_context is None
            assert traced.output.logits.device == device
            assert torch.equal(baseline.view(torch.int32), traced.output.logits.cpu().view(torch.int32))
            assert len(traced.events) == 2 * 3 * ids.shape[1]
            assert Counter((x.event.module_path, x.event.batch_positions, x.event.token_positions) for x in traced.events) == Counter(
                (name, (0,), (token,)) for name in expected for token in range(ids.shape[1]))
            for observation in traced.events:
                event = observation.event
                wanted = expected[event.module_path][event.batch_positions[0], event.token_positions[0]]
                assert observation.value.device == device and event.device == str(device)
                assert torch.equal(observation.value[0, 0].cpu().view(torch.int32), wanted.view(torch.int32))
            del traced, hooked
        result = runtime.run("hello [MASK]", {})
        assert result["gpu_memory"]["synchronized"] and runtime.info()["device"] == str(device)
        assert result["estimates"]["cuda_logits_bytes"] == ids.shape[1] * runtime.model.config.vocab_size * 4
        assert len(result["events"]) == len(expected) * ids.shape[1]
        assert Counter((x.metadata()["module_path"], tuple(x.metadata()["batch_positions"]), tuple(x.metadata()["token_positions"])) for x in result["events"]) == Counter(
            (name, (0,), (token,)) for name in expected for token in range(ids.shape[1]))
        wanted_predictions = [(position, int(baseline[0, position].argmax())) for position in range(ids.shape[1])
                              if int(ids[0, position]) == runtime.tokenizer.mask_token_id]
        assert [(x["position"], x["token_id"]) for x in result["steps"][0]["mask_predictions"]] == wanted_predictions
        for capture in result["events"]:
            meta = capture.metadata()
            assert meta["device"] == str(device) and meta["saved_device"] == "cpu" and meta["app_offloaded_from_device"] == str(device)
            wanted = expected[meta["module_path"]][meta["batch_positions"][0], meta["token_positions"][0]]
            assert torch.equal(capture.value[0, 0].view(torch.int32), wanted.view(torch.int32))
        store = TraceStore(tmp_path / "exports"); store.publish(result)
        bits = [store.tensor(i)["raw_bits"] for i in range(len(result["events"]))]
        saved = store.save("distilbert-cuda"); store.clear(); store.open(saved["path"])
        assert bits == [store.tensor(i)["raw_bits"] for i in range(len(bits))]
        again = runtime.run("hello [MASK]", {})
        assert [(x["position"], x["token_id"]) for x in again["steps"][0]["mask_predictions"]] == wanted_predictions
        assert again["gpu_memory"]["synchronized"]
        assert torch.cuda.get_per_process_memory_fraction(device) == previous
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
    finally:
        for handle in handles: handle.remove()
        if runtime.info() and runtime.info()["state"] == "ready":
            runtime.unload()
            assert torch.cuda.get_per_process_memory_fraction(device) == previous


def test_real_vit_cuda_logits_pixels_grid_capture_offload_export_and_repeat(tmp_path):
    import io
    from collections import Counter
    from PIL import Image
    from looking_glass.app_runtime import _cuda_operation
    from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier
    from test_app_runtime import image_bytes
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < max(256 * 1024**2, math.ceil(total * .10)) + 64 * 1024**2:
        pytest.skip("real GPU lacks required free-memory/headroom prerequisite")
    path = tmp_path / "vit"; local_checkpoint(path, "vit")
    runtime = LocalModelRuntime(); prior = torch.cuda.get_per_process_memory_fraction(device)
    runtime.load(str(path), device=str(device), max_vram_bytes=1024**3)
    handles, expected = [], {}
    try:
        with _cuda_operation(device, 1024**3), torch.inference_mode():
            with Image.open(io.BytesIO(image_bytes())) as image:
                pixels = runtime.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
            original_bits = pixels.view(torch.int32).cpu().clone()
            points = [ViTTracePoint(layer, role, 0, token) for layer in range(2)
                      for role in ("attention", "intermediate", "block") for token in range(5)]
            with pytest.raises(ValueError, match="CPU required by default"):
                trace_vit_classifier(runtime.model, pixels, points=points, run_id="default")
            baseline_output = runtime.model(pixel_values=pixels).logits
            expected_class = int(torch.topk(baseline_output[0].softmax(dim=-1), min(5, baseline_output.shape[-1])).indices[0])
            baseline = baseline_output.cpu()
            del baseline_output
            for layer in range(2):
                for role in ("attention", "intermediate", "block"):
                    name = f"vit.encoder.layer.{layer}" + ("" if role == "block" else f".{role}")
                    def save(_module, _args, output, name=name): expected[name] = output.detach().cpu().clone()
                    handles.append(runtime.model.get_submodule(name).register_forward_hook(save))
            hooked = runtime.model(pixel_values=pixels).logits.cpu()
            assert torch.equal(baseline.view(torch.int32), hooked.view(torch.int32))
            for handle in handles: handle.remove()
            handles.clear()
            traced = trace_vit_classifier(runtime.model, pixels, points=points, run_id="cuda", synchronous_cpu=False)
            assert torch.equal(baseline.view(torch.int32), traced.output.logits.cpu().view(torch.int32))
            assert torch.equal(original_bits, pixels.view(torch.int32).cpu())
            assert len(traced.records) == 30
            assert Counter((r.event.module_path, r.event.token_positions) for r in traced.records) == Counter(
                (name, (token,)) for name in expected for token in range(5))
            for record in traced.records:
                assert record.value.device == device and record.event.device == str(device)
                assert torch.equal(record.value.cpu().view(torch.int32), expected[record.event.module_path][0, record.event.token_positions[0]].view(torch.int32))
            geometry = traced.image_metadata
            del traced, hooked
        result = runtime.run("", {}, image_bytes())
        assert result["image_metadata"] == geometry and result["steps"][0]["classes"][0]["class_id"] == expected_class
        assert result["gpu_memory"]["synchronized"] and result["estimates"]["cuda_logits_bytes"] == 12
        assert len(result["events"]) == 30
        for capture in result["events"]:
            metadata = capture.metadata()
            assert metadata["device"] == str(device) and metadata["saved_device"] == "cpu"
            assert metadata["app_offloaded_from_device"] == str(device)
            assert torch.equal(capture.value.view(torch.int32), expected[metadata["module_path"]][0, metadata["token_positions"][0]].view(torch.int32))
        store = TraceStore(tmp_path / "exports"); store.publish(result)
        assert all(store.vectors(i)["vectors"][0]["coordinate"]["status"] == "known" for i in range(30))
        coordinates = [store.vectors(i)["vectors"][0]["coordinate"] for i in range(30)]
        bits = [store.tensor(i)["raw_bits"] for i in range(30)]
        saved = store.save("vit-cuda"); store.clear(); store.open(saved["path"])
        assert bits == [store.tensor(i)["raw_bits"] for i in range(30)]
        assert coordinates == [store.vectors(i)["vectors"][0]["coordinate"] for i in range(30)]
        assert store.summary()["image_metadata"] == geometry
        again = runtime.run("", {}, image_bytes())
        assert again["steps"][0]["classes"] == result["steps"][0]["classes"] and again["gpu_memory"]["synchronized"]
        assert torch.cuda.get_per_process_memory_fraction(device) == prior
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
    finally:
        for handle in handles: handle.remove()
        if runtime.info() and runtime.info()["state"] == "ready":
            runtime.unload()
            assert torch.cuda.get_per_process_memory_fraction(device) == prior
