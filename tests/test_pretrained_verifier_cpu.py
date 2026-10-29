"""Opt-in real-checkpoint CPU challenge of the loaded-model verifier."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest


CACHE_ENV = "LOOKING_GLASS_TEST_CACHE"
SUMMARY_ENV = "LOOKING_GLASS_VERIFIER_CPU_SUMMARY"


def _child(phase):
    import hashlib
    import json
    import socket
    from contextlib import ExitStack
    from unittest.mock import patch

    phase[0] = "startup"
    repo = Path(os.environ["LOOKING_GLASS_VERIFIER_REPO"]).resolve(strict=True)
    site = Path(os.environ["LOOKING_GLASS_VERIFIER_SITE"]).resolve(strict=True)
    cache = Path(os.environ[CACHE_ENV]).resolve(strict=True)
    prefix = Path(os.environ["LOOKING_GLASS_VERIFIER_PREFIX"]).resolve(strict=True)
    assert sys.flags.no_site == 1 and sys.flags.isolated == 1 and sys.dont_write_bytecode
    assert Path(sys.pycache_prefix).resolve(strict=True) == prefix and not any(prefix.iterdir())
    sys.path[:0] = [str(repo), str(site)]
    verifier_path = repo / "looking_glass" / "live_profile.py"
    verifier_digest = hashlib.sha256(verifier_path.read_bytes()).hexdigest()

    # CUDA is hidden from interpreter startup. Import torch, then deny the
    # listed initialization/device-query helpers throughout the CPU challenge.
    phase[0] = "torch_import"
    import torch
    assert not torch.cuda._initialized
    calls = []

    def denied(name):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError("guarded CUDA or network entry point")
        return fail

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
        phase[0] = "imports"
        import huggingface_hub
        from looking_glass import live_profile, pretrained
        assert Path(live_profile.__file__).resolve(strict=True) == verifier_path
        assert pretrained.MODEL_ID == "distilbert/distilgpt2"
        assert pretrained.REVISION == "2290a62682d06624634c1f46a6ad5be0f47f38aa"

        # This is a verifier fixture, not a full SourcePreflight or an
        # attestation of the installed source and execution environment.
        phase[0] = "fixture"
        paths = tuple((name, huggingface_hub.hf_hub_download(
            repo_id=pretrained.MODEL_ID, revision=pretrained.REVISION, filename=name,
            cache_dir=str(cache), local_files_only=True, token=False))
            for name in sorted(live_profile._PINNED_ASSETS))
        preflight = live_profile.SourcePreflight("", str(cache), "", "", b"{}", b"{}", paths)
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(17)
        phase[0] = "model_load"
        with patch.object(pretrained, "load_distilgpt2", wraps=pretrained.load_distilgpt2) as load:
            bundle = pretrained.load_distilgpt2(cache_dir=cache, allow_download=False)
            assert load.call_count == 1
            direct_loads = load.call_count
        model = bundle.model
        assert live_profile.verify_loaded_model(preflight, model) is True
        ids = torch.tensor([[15496, 995]], dtype=torch.int64, device="cpu")
        manifest = {"schema": "resource-bootstrap-case-v1", "input_ids": ids.tolist()[0],
                    "shape": list(ids.shape), "dtype": str(ids.dtype)}
        case_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=True).encode()).hexdigest()
        assert case_hash == "a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4"
        with torch.inference_mode():
            baseline = model(input_ids=ids).logits.clone()
        assert tuple(baseline.shape) == (1, 2, 50257) and bool(torch.isfinite(baseline).all())

        outcomes = []
        differences = {}

        def reject(name, mutate, restore, *, observe=False):
            phase[0] = name
            mutate()
            try:
                try:
                    live_profile.verify_loaded_model(preflight, model)
                except ValueError:
                    pass
                else:
                    raise AssertionError("altered model was accepted")
                if observe:
                    with torch.inference_mode():
                        changed = model(input_ids=ids).logits
                    assert bool(torch.isfinite(changed).all())
                    differences[name] = float((changed - baseline).abs().max())
                outcomes.append(name)
            finally:
                restore()

        def valid_after_group():
            phase[0] = "restored_model"
            assert live_profile.verify_loaded_model(preflight, model) is True

        attn = model.transformer.h[0].attn
        param = attn.c_attn.bias
        original_scalar = param[0].detach().clone()
        def change_weight():
            with torch.no_grad():
                param[0].add_(20.0)
        def restore_weight():
            with torch.no_grad():
                param[0].copy_(original_scalar)
        reject("parameter_value", change_weight, restore_weight, observe=True)
        assert differences["parameter_value"] > 0

        mask = attn.bias
        original_bit = mask[0, 0, 1, 0].detach().clone()
        assert bool(original_bit)
        def change_mask():
            with torch.no_grad():
                mask[0, 0, 1, 0] = False
        def restore_mask():
            with torch.no_grad():
                mask[0, 0, 1, 0].copy_(original_bit)
        reject("attention_mask_value", change_mask, restore_mask, observe=True)
        assert differences["attention_mask_value"] > 0

        masked_bias = attn.masked_bias
        original_bias = masked_bias.detach().clone()
        def change_masked_bias():
            with torch.no_grad():
                masked_bias.fill_(-9999.0)
        def restore_masked_bias():
            with torch.no_grad():
                masked_bias.copy_(original_bias)
        reject("masked_bias_value", change_masked_bias, restore_masked_bias)
        valid_after_group()

        vocab = model.config.vocab_size
        reject("vocab_config", lambda: setattr(model.config, "vocab_size", vocab - 1),
               lambda: setattr(model.config, "vocab_size", vocab))
        reject("training_mode", model.train, model.eval)
        scaling = attn.scale_attn_weights
        assert scaling is True
        reject("attention_scaling", lambda: setattr(attn, "scale_attn_weights", False),
               lambda: setattr(attn, "scale_attn_weights", scaling), observe=True)
        assert differences["attention_scaling"] > 0
        valid_after_group()

        original_weight = attn.c_attn.weight
        strided = original_weight.detach().transpose(0, 1).contiguous().transpose(0, 1)
        assert torch.equal(original_weight.detach().contiguous().view(torch.uint8),
                           strided.contiguous().view(torch.uint8))
        assert original_weight.stride() != strided.stride()
        strided_parameter = torch.nn.Parameter(strided, requires_grad=original_weight.requires_grad)
        reject("parameter_stride", lambda: setattr(attn.c_attn, "weight", strided_parameter),
               lambda: setattr(attn.c_attn, "weight", original_weight))
        del strided_parameter, strided

        strided_mask = mask.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert torch.equal(mask, strided_mask) and mask.stride() != strided_mask.stride()
        reject("attention_mask_stride", lambda: setattr(attn, "bias", strided_mask),
               lambda: setattr(attn, "bias", mask))
        del strided_mask

        backing = torch.empty(original_weight.numel() + 1, dtype=original_weight.dtype, device="cpu")
        offset = backing[1:].view(original_weight.shape)
        with torch.no_grad():
            offset.copy_(original_weight)
        assert torch.equal(offset.contiguous().view(torch.uint8),
                           original_weight.detach().contiguous().view(torch.uint8))
        assert offset.storage_offset() == 1
        offset_parameter = torch.nn.Parameter(offset, requires_grad=original_weight.requires_grad)
        reject("parameter_storage_offset", lambda: setattr(attn.c_attn, "weight", offset_parameter),
               lambda: setattr(attn.c_attn, "weight", original_weight))
        del offset_parameter, offset, backing

        other_attn = model.transformer.h[1].attn
        other_mask = other_attn.bias
        alias = mask.view_as(mask)
        assert alias is not mask and torch.equal(other_mask, alias)
        reject("attention_mask_storage_alias", lambda: setattr(other_attn, "bias", alias),
               lambda: setattr(other_attn, "bias", other_mask))
        del alias
        valid_after_group()

        original_head = model.lm_head.weight
        shared_head = torch.nn.Parameter(original_head.detach(), requires_grad=original_head.requires_grad)
        assert shared_head is not original_head
        assert shared_head.untyped_storage().data_ptr() == original_head.untyped_storage().data_ptr()
        assert shared_head.shape == original_head.shape and shared_head.stride() == original_head.stride()
        assert shared_head.storage_offset() == original_head.storage_offset()
        reject("shared_storage_distinct_parameter", lambda: setattr(model.lm_head, "weight", shared_head),
               lambda: setattr(model.lm_head, "weight", original_head))
        del shared_head

        untied = torch.nn.Parameter(original_head.detach().clone(), requires_grad=original_head.requires_grad)
        assert torch.equal(original_head.detach().contiguous().view(torch.uint8),
                           untied.detach().contiguous().view(torch.uint8))
        assert original_head.untyped_storage().data_ptr() != untied.untyped_storage().data_ptr()
        reject("untied_head", lambda: setattr(model.lm_head, "weight", untied),
               lambda: setattr(model.lm_head, "weight", original_head))
        del untied

        hook = []
        def add_local_hook():
            hook.append(attn.register_forward_hook(lambda module, inputs, output: output))
        def remove_hook():
            hook.pop().remove()
        reject("local_hook", add_local_hook, remove_hook)
        def add_global_hook():
            hook.append(torch.nn.modules.module.register_module_forward_hook(
                lambda module, inputs, output: output))
        reject("global_hook", add_global_hook, remove_hook)
        valid_after_group()

        phase[0] = "final"
        with torch.inference_mode():
            restored = model(input_ids=ids).logits
        assert restored.shape == baseline.shape and restored.dtype == baseline.dtype == torch.float32
        assert restored.is_contiguous() and baseline.is_contiguous()
        assert torch.equal(restored.view(torch.uint8), baseline.view(torch.uint8))
        assert not calls and not torch.cuda._initialized
        assert not any(prefix.iterdir())
        assert hashlib.sha256(verifier_path.read_bytes()).hexdigest() == verifier_digest

    return {"status": "passed", "verifier_source_sha256": verifier_digest,
            "case_sha256": case_hash, "rejected_cases": outcomes,
            "max_absolute_logit_changes": differences, "restored_output_bitwise_equal": True,
            "direct_loader_calls": direct_loads, "guarded_cuda_and_network_calls": 0}


@unittest.skipUnless(os.environ.get(CACHE_ENV), "set LOOKING_GLASS_TEST_CACHE to opt in")
class PretrainedVerifierCpuTests(unittest.TestCase):
    def test_real_checkpoint_rejects_model_mutations_on_cpu(self):
        import json

        repo = Path.cwd().resolve(strict=True)
        site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve(strict=True)
        scratch = os.environ.get("LOOKING_GLASS_TEST_SCRATCH") or None
        with tempfile.TemporaryDirectory(prefix="looking-glass-verifier-cpu-", dir=scratch) as temp:
            prefix = Path(temp).resolve(strict=True) / "pycache"
            prefix.mkdir()
            env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("CUDA_", "PYTORCH_", "TORCH_", "CUBLAS_", "HF_",
                                                  "TRANSFORMERS_", "OMP_", "MKL_", "OPENBLAS_", "PYTHON",
                                                  "LOOKING_GLASS_"))}
            env.update({CACHE_ENV: str(Path(os.environ[CACHE_ENV]).absolute()),
                        "LOOKING_GLASS_VERIFIER_REPO": str(repo), "LOOKING_GLASS_VERIFIER_SITE": str(site),
                        "LOOKING_GLASS_VERIFIER_PREFIX": str(prefix), "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "HF_HOME": str(Path(os.environ[CACHE_ENV]).absolute().parent)})
            result = subprocess.run([str(base), "-I", "-B", "-S", "-X", f"pycache_prefix={prefix}",
                                     str(Path(__file__).resolve()), "--verifier-child"], cwd=repo, env=env,
                                    capture_output=True, text=True, timeout=240)
            self.assertFalse(any(prefix.iterdir()), "private bytecode prefix was modified")
        try:
            summary = json.loads(result.stdout)
        except ValueError:
            self.fail("CPU verifier child did not return an aggregate summary")
        if result.returncode != 0 or summary.get("status") != "passed":
            phases = {"startup", "torch_import", "imports", "fixture", "model_load", "parameter_value",
                      "attention_mask_value", "masked_bias_value", "vocab_config", "training_mode",
                      "attention_scaling", "parameter_stride", "attention_mask_stride",
                      "parameter_storage_offset", "attention_mask_storage_alias", "untied_head",
                      "shared_storage_distinct_parameter", "local_hook", "global_hook", "restored_model", "final"}
            phase = summary.get("phase") if summary.get("phase") in phases else "unknown"
            self.fail(f"CPU verifier child failed at {phase}; private output withheld")
        if os.environ.get(SUMMARY_ENV) == "1":
            print("PRETRAINED_VERIFIER_CPU_V2 " + json.dumps(summary, sort_keys=True), flush=True)


@unittest.skipUnless(sys.implementation.name == "cpython", "test uses CPython timestamp pyc headers")
class VerifierCollectorIsolationTests(unittest.TestCase):
    @staticmethod
    def collector_namespace():
        collector = Path(__file__).resolve().parent.parent / "tools" / "pretrained_verifier_cpu_evidence.py"
        namespace = {"__name__": "_verifier_collector_under_test", "__file__": str(collector)}
        exec(compile(collector.read_bytes(), str(collector), "exec"), namespace)
        return namespace

    def test_collector_parent_ignores_valid_stale_test_bytecode(self):
        import marshal
        import py_compile

        command_builder = self.collector_namespace()["_unittest_command"]
        scratch = os.environ.get("LOOKING_GLASS_TEST_SCRATCH") or None
        with tempfile.TemporaryDirectory(prefix="verifier-bytecode-test-", dir=scratch) as temp:
            root = Path(temp)
            tests = root / "tests"
            tests.mkdir()
            source = tests / "test_pretrained_verifier_cpu.py"
            source.write_text("import unittest\nclass Fixture(unittest.TestCase):\n"
                              "    def test_source(self):\n        pass\n", encoding="utf-8")
            cached = tests / "__pycache__" / ("test_pretrained_verifier_cpu." + sys.implementation.cache_tag + ".pyc")
            cached.parent.mkdir()
            py_compile.compile(str(source), cfile=str(cached), doraise=True, optimize=0,
                               invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            header = cached.read_bytes()[:16]
            stale = compile("raise RuntimeError('STALE_VERIFIER_TEST_BYTECODE')\n", str(source), "exec")
            cached.write_bytes(header + marshal.dumps(stale))
            control = subprocess.run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", "tests",
                                      "-p", "test_pretrained_verifier_cpu.py", "-v"], cwd=root,
                                     capture_output=True, text=True, timeout=30)
            self.assertNotEqual(control.returncode, 0)
            self.assertIn("STALE_VERIFIER_TEST_BYTECODE", control.stderr)
            prefix = root / "private-prefix"
            prefix.mkdir()
            result = subprocess.run(command_builder(sys.executable, prefix), cwd=root,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ran 1 test", result.stderr)
            self.assertNotIn("STALE_VERIFIER_TEST_BYTECODE", result.stderr)
            self.assertFalse(any(prefix.iterdir()))

    def test_collector_accepts_only_finite_aggregate_summary_fields(self):
        import copy

        validate = self.collector_namespace()["_validate_summary"]
        valid = {"status": "passed", "verifier_source_sha256": "verifier", "case_sha256": "case",
                 "rejected_cases": ["parameter_value", "attention_mask_value", "masked_bias_value", "vocab_config",
                                    "training_mode", "attention_scaling", "parameter_stride", "attention_mask_stride",
                                    "parameter_storage_offset", "attention_mask_storage_alias",
                                    "shared_storage_distinct_parameter", "untied_head", "local_hook", "global_hook"],
                 "max_absolute_logit_changes": {"parameter_value": 1.0, "attention_mask_value": 2.0,
                                                "attention_scaling": 3.0},
                 "restored_output_bitwise_equal": True, "direct_loader_calls": 1,
                 "guarded_cuda_and_network_calls": 0}
        validate(valid, "case", "verifier")
        for field, value in (("extra", "private"), ("direct_loader_calls", True),
                             ("guarded_cuda_and_network_calls", False)):
            invalid = copy.deepcopy(valid)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(invalid, "case", "verifier")
        for value in (float("nan"), float("inf"), 0.0, True):
            invalid = copy.deepcopy(valid)
            invalid["max_absolute_logit_changes"]["parameter_value"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(invalid, "case", "verifier")
        invalid = copy.deepcopy(valid)
        invalid["max_absolute_logit_changes"]["extra"] = 4.0
        with self.assertRaises(ValueError):
            validate(invalid, "case", "verifier")


if __name__ == "__main__":
    if sys.argv[1:] == ["--verifier-child"]:
        import json
        phase = ["startup"]
        try:
            result = _child(phase)
        except BaseException:
            print(json.dumps({"status": "failed", "phase": phase[0]}), flush=True)
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True), flush=True)
    else:
        unittest.main()
