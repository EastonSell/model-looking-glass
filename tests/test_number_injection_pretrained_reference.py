"""Opt-in, disclosed-site CPU mechanics check on an already exposed input.

This is not a discovery, held-out case, GPU benchmark, or behavioral claim.
"""
import hashlib
import json
import os
import socket
import struct
import unittest
from contextlib import ExitStack
from unittest.mock import patch


CACHE_ENV = "LOOKING_GLASS_TEST_CACHE"
SUMMARY_ENV = "LOOKING_GLASS_REFERENCE_SUMMARY"
IDS = (15496, 995)
SITE = 2


def _raw(tensor):
    return tensor.detach().contiguous().view(__import__("torch").uint8).numpy().tobytes()


def _direction(torch):
    # Independently reconstruct the frozen sign vector, not the executor's
    # helper or its saved tensors.
    bits = []
    for chunk in range(3):
        digest = hashlib.sha256(b"MLG-v1-direction\0" + (20261028).to_bytes(8, "big") +
                                chunk.to_bytes(4, "big")).digest()
        bits.extend(0x3c93cd3a if byte & (1 << bit) else 0xbc93cd3a
                    for byte in digest for bit in range(7, -1, -1))
    return torch.frombuffer(bytearray(struct.pack("<768I", *bits)), dtype=torch.int32).clone().view(torch.float32)


def _reference(torch, model, condition, *, donor=None):
    """Plain PyTorch hooks, independent of ActivationPlan and the executor."""
    captures, witnesses, counts, handles = {}, {}, {}, []
    delta = _direction(torch)
    inject = condition in ("injected", "clean_to_injected")
    patch_site = condition in ("clean_to_injected", "injected_to_clean")
    for layer, block in enumerate(model.transformer.h):
        def attention(_module, _args, output, layer=layer):
            name = f"attention_{layer}"
            counts[name] = counts.get(name, 0) + 1
            if type(output) is not tuple:
                raise AssertionError("unexpected attention output")
            captures[name] = _raw(output[0][0, -1])

        def mlp(_module, _args, output, layer=layer):
            name = f"mlp_{layer}"
            counts[name] = counts.get(name, 0) + 1
            if type(output) is not torch.Tensor:
                raise AssertionError("unexpected MLP output")
            value = output
            if inject and layer == SITE:
                before = _raw(value)
                value = value.clone()
                value[0, -1] = value[0, -1] + delta
                witnesses["add"] = (before, _raw(value), None)
            if patch_site and layer == SITE:
                before = _raw(value)
                value = value.clone()
                value[0, -1] = torch.frombuffer(bytearray(donor), dtype=torch.int32).clone().view(torch.float32)
                witnesses["patch"] = (before, _raw(value), donor)
            captures[name] = _raw(value[0, -1])
            return value

        handles.extend((block.attn.register_forward_hook(attention), block.mlp.register_forward_hook(mlp)))

    def final_block(_module, _args, output):
        counts["block_5"] = counts.get("block_5", 0) + 1
        if type(output) is not tuple:
            raise AssertionError("unexpected final block output")
        captures["block_5"] = _raw(output[0][0, -1])

    handles.append(model.transformer.h[5].register_forward_hook(final_block))
    try:
        ids = torch.tensor([IDS], dtype=torch.int64, device="cpu")
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
        if tuple(output.logits.shape) != (1, 2, 50257):
            raise AssertionError("unexpected model output shape")
        logits = _raw(output.logits[0, -1])
    finally:
        for handle in handles:
            handle.remove()
    if len(counts) != 13 or set(counts.values()) != {1}:
        raise AssertionError("reference hooks did not observe each boundary exactly once")
    return captures, witnesses, logits


@unittest.skipUnless(os.environ.get(CACHE_ENV), "set LOOKING_GLASS_TEST_CACHE to opt in")
class PinnedInjectionReferenceTests(unittest.TestCase):
    def test_disclosed_site_matches_independent_serial_hooks(self):
        import torch
        import transformers
        import huggingface_hub
        from looking_glass import live_profile, pretrained
        from looking_glass.agreement_numeric import compare_f32_le
        from looking_glass.pretrained_number_injection import NumberInjectionExecutor

        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or torch.cuda._initialized:
            self.fail("CPU-only startup was not established")
        calls = []
        def denied(name):
            def fail(*_args, **_kwargs):
                calls.append(name)
                raise AssertionError("guarded CUDA or network entry point")
            return fail

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        tiny = torch.tensor([1], dtype=torch.int32).view(torch.float32)
        was_flushed = int((tiny * 1.0).view(torch.int32).item()) == 0
        if not torch.set_flush_denormal(False):
            self.fail("cannot establish CPU denormal handling")
        self.addCleanup(torch.set_flush_denormal, was_flushed)
        torch.random.default_generator.manual_seed(20261028)
        with ExitStack() as stack:
            for name in ("init", "_lazy_init", "device_count", "is_available", "get_device_properties",
                         "current_device", "set_device", "synchronize", "manual_seed_all", "_is_in_bad_fork"):
                stack.enter_context(patch.object(torch.cuda, name, denied("cuda." + name)))
            for name in ("_cuda_init", "_cuda_getDeviceCount"):
                if hasattr(torch._C, name):
                    stack.enter_context(patch.object(torch._C, name, denied("torch._C." + name)))
            for name in ("connect", "connect_ex"):
                stack.enter_context(patch.object(socket.socket, name, denied("socket." + name)))
            for name in ("create_connection", "getaddrinfo"):
                stack.enter_context(patch.object(socket, name, denied("socket." + name)))
            cache = os.environ[CACHE_ENV]
            paths = tuple((name, huggingface_hub.hf_hub_download(
                repo_id=pretrained.MODEL_ID, revision=pretrained.REVISION, filename=name,
                cache_dir=cache, local_files_only=True, token=False)) for name in sorted(live_profile._PINNED_ASSETS))
            preflight = live_profile.SourcePreflight("", cache, "", "", b"{}", b"{}", paths)
            bundle = pretrained.load_distilgpt2(cache_dir=cache, allow_download=False)
            model = bundle.model
            if type(model) is not transformers.GPT2LMHeadModel or not live_profile.verify_loaded_model(preflight, model):
                self.fail("pinned CPU model verification failed")
            reference_clean = _reference(torch, model, "clean")
            reference_injected = _reference(torch, model, "injected")
            reference_forward = _reference(torch, model, "clean_to_injected", donor=reference_clean[0]["mlp_2"])
            reference_reverse = _reference(torch, model, "injected_to_clean", donor=reference_injected[0]["mlp_2"])
            references = (reference_clean, reference_injected, reference_forward, reference_reverse)
            if hashlib.sha256(_raw(_direction(torch))).hexdigest() != "fee167bdb27a7dc195a0d280df78e07fe6f6fff5218f6f07299e7d4bce6f98b7":
                self.fail("independent injection direction does not match the frozen definition")
            maxima = []
            with NumberInjectionExecutor(model, site=SITE) as executor:
                clean = executor.forward(IDS, request_id="1" * 32, condition="clean")
                injected = executor.forward(IDS, request_id="2" * 32, condition="injected")
                forward = executor.forward(IDS, request_id="3" * 32, condition="clean_to_injected", layer=SITE,
                                           donor=clean.mlp_donor(SITE))
                reverse = executor.forward(IDS, request_id="4" * 32, condition="injected_to_clean", layer=SITE,
                                           donor=injected.mlp_donor(SITE))
                for observation, (expected, witnesses, logits) in zip((clean, injected, forward, reverse), references):
                    paths = [name for layer in range(6) for name in
                             (f"transformer.h.{layer}.attn", f"transformer.h.{layer}.mlp")] + ["transformer.h.5"]
                    if len(observation.trace) != len(paths):
                        self.fail("public trace is incomplete")
                    for order, (event, path) in enumerate(zip(observation.trace, paths)):
                        selector = (0,) if path.endswith(".attn") or path == "transformer.h.5" else ()
                        if ((event.module_path, event.capture_order, event.invocation, event.batch_index,
                             event.token_index, event.output_selector, event.output_shape, event.selected_shape,
                             event.dtype, event.device) !=
                                (path, order, 0, 0, 1, selector, (1, 2, 768), (768,), "torch.float32", "cpu")):
                            self.fail("public trace coordinates do not match the executed path")
                    comparisons = [(observation.attention[i], expected[f"attention_{i}"]) for i in range(6)]
                    comparisons += [(observation.mlp[i], expected[f"mlp_{i}"]) for i in range(6)]
                    comparisons += [(observation.block, expected["block_5"]), (observation.logits, logits)]
                    differences = [compare_f32_le(left, right) for left, right in comparisons]
                    if not all(item["within_tolerance"] for item in differences):
                        self.fail("plain-hook and executor outputs differ beyond the preregistered tolerance")
                    maxima.append(max(item["max_abs_difference_binary64"] for item in differences))
                    evidence = [item for item in executor.private_evidence() if item.request_id == observation.request_id]
                    if {item.operation for item in evidence} != set(witnesses):
                        self.fail("private witness operations differ from plain hooks")
                    for item in evidence:
                        before, after, source = witnesses[item.operation]
                        if (item.module_path != "transformer.h.2.mlp" or item.token_index != 1 or
                                (item.before, item.after, item.donor) != (before, after, source)):
                            self.fail("private action witness differs from plain hooks")
                if not compare_f32_le(clean.logits, forward.logits)["within_tolerance"]:
                    self.fail("same-site clean-to-injected patch did not restore clean logits")
                if not compare_f32_le(injected.logits, reverse.logits)["within_tolerance"]:
                    self.fail("same-site reverse patch did not restore injected logits")
            if not live_profile.verify_loaded_model(preflight, model):
                self.fail("model verification failed after hook cleanup")
            if calls:
                self.fail("a guarded CUDA or network helper was invoked")
            if torch.cuda._initialized:
                self.fail("CUDA initialized during the CPU check")
            if os.environ.get(SUMMARY_ENV) == "1":
                case = {"schema": "resource-bootstrap-case-v1", "input_ids": list(IDS),
                        "shape": [1, 2], "dtype": "torch.int64"}
                print(json.dumps({"model_revision": pretrained.REVISION, "weight_sha256": pretrained.ASSETS["model.safetensors"][1],
                                  "input_sha256": hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":"),
                                                                             ensure_ascii=True).encode()).hexdigest(),
                                  "device": "cpu", "site": SITE, "reference_forwards": 4, "executor_forwards": 4,
                                  "max_abs_error_by_condition": maxima, "guarded_calls": len(calls)}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
