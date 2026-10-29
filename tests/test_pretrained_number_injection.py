"""CPU-only tests for the known-injection number-study executor."""
import hashlib
import os
import random
import struct
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace

import torch
from torch import nn

from looking_glass.agreement_numeric import compare_f32_le
from looking_glass.model_ownership import ModelLease
from looking_glass.pretrained_number_injection import (
    NumberInjectionExecutor,
    _check_plan_metadata,
    _check_witness_chain,
    _plan_steps,
    add_float32_bits,
    injection_direction_bits,
    verify_cpu_float32_addition,
    verify_action_scope,
    verify_injection_bits,
)


_original_torch_threads = None


def setUpModule():
    global _original_torch_threads
    _original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_original_torch_threads)


class Float32InjectionArithmeticTests(unittest.TestCase):
    def test_addition_rounds_exactly_including_subnormals_and_signed_zero(self):
        rows = [
            (0x00000000, 0x80000000, 0x00000000),
            (0x80000000, 0x80000000, 0x80000000),
            (0x3f800000, 0xbf800000, 0x00000000),
            (0x3f800000, 0x33800000, 0x3f800000),
            (0x3f800001, 0x33800000, 0x3f800002),
            (0x00000001, 0x00000001, 0x00000002),
            (0x00800000, 0x807fffff, 0x00000001),
            (0x7f7fffff, 0x7f7fffff, 0x7f800000),
            (0xff7fffff, 0xff7fffff, 0xff800000),
        ]
        for left, right, expected in rows:
            with self.subTest(left=hex(left), right=hex(right)):
                self.assertEqual(add_float32_bits(left, right), expected)
                self.assertEqual(add_float32_bits(right, left), expected)

    def test_nonfinite_and_noninteger_inputs_are_rejected(self):
        for left in (True, -1, 1 << 32, 0x7f800000, 0x7fc00001):
            with self.subTest(left=left), self.assertRaises(ValueError):
                add_float32_bits(left, 0)

    def test_fixed_direction_bits_and_seed_are_stable(self):
        bits = injection_direction_bits()
        self.assertEqual(len(bits), 768)
        self.assertEqual(bits[:8], (0x3c93cd3a, 0x3c93cd3a, 0xbc93cd3a, 0xbc93cd3a,
                                    0xbc93cd3a, 0xbc93cd3a, 0x3c93cd3a, 0xbc93cd3a))
        self.assertEqual(hashlib.sha256(struct.pack("<768I", *bits)).hexdigest(),
                         "fee167bdb27a7dc195a0d280df78e07fe6f6fff5218f6f07299e7d4bce6f98b7")

    def test_cpu_runtime_matches_reference_for_varied_finite_values(self):
        verify_cpu_float32_addition()
        rng = random.Random(29)
        rows = []
        while len(rows) < 4096:
            left, right = rng.getrandbits(32), rng.getrandbits(32)
            if ((left >> 23) & 255) != 255 and ((right >> 23) & 255) != 255:
                rows.append((left, right))
        signed = lambda value: value if value < (1 << 31) else value - (1 << 32)
        left = torch.tensor([signed(a) for a, _ in rows], dtype=torch.int32).view(torch.float32)
        right = torch.tensor([signed(b) for _, b in rows], dtype=torch.int32).view(torch.float32)
        actual = torch.add(left, right).view(torch.int32).tolist()
        self.assertEqual([value & 0xffffffff for value in actual],
                         [add_float32_bits(a, b) for a, b in rows])

    def test_runtime_guard_rejects_flush_to_zero_when_supported(self):
        code = """import sys, torch
from looking_glass.pretrained_number_injection import verify_cpu_float32_addition
if not torch.set_flush_denormal(True):
    sys.exit(77)
try:
    verify_cpu_float32_addition()
except RuntimeError:
    sys.exit(0)
sys.exit(1)
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        result = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, timeout=30)
        if result.returncode == 77:
            self.skipTest("runtime does not expose denormal flushing")
        self.assertEqual(result.returncode, 0, "the runtime sentinel did not reject flush-to-zero")

    def test_injection_requires_exact_bits_finite_values_and_actual_change(self):
        direction = injection_direction_bits()
        before = (0,) * 768
        pre = struct.pack("<768I", *before)
        post = struct.pack("<768I", *direction)
        self.assertEqual(verify_injection_bits(pre, post), {"changed_coordinates": 768})
        bad = list(direction)
        bad[0] ^= 1
        with self.assertRaises(ValueError):
            verify_injection_bits(pre, struct.pack("<768I", *bad))
        bad[0] = 0x7f800000
        with self.assertRaises(ValueError):
            verify_injection_bits(pre, struct.pack("<768I", *bad))
        unchanged = struct.pack("<768I", *((0x7f000000,) * 768))
        with self.assertRaises(ValueError):
            verify_injection_bits(unchanged, unchanged)
        with self.assertRaises(ValueError):
            verify_injection_bits(b"", post)


class OrderedInjectionPlanTests(unittest.TestCase):
    def test_adjacent_actions_and_public_observation_are_cross_checked(self):
        zero = struct.pack("<768I", *((0,) * 768))
        changed = struct.pack("<768I", *injection_direction_bits())
        full_zero, full_changed = zero * 2, zero + changed
        events = {"inject_add": {"module_path": "transformer.h.2.mlp"},
                  "action_patch": {"module_path": "transformer.h.2.mlp"}}
        raw = {"inject_after": full_changed, "action_before": full_changed,
               "action_after": full_zero, "mlp_2": zero}
        _check_witness_chain(raw, events, 1)
        with self.assertRaises(ValueError):
            _check_witness_chain(dict(raw, action_before=full_zero), events, 1)
        with self.assertRaises(ValueError):
            _check_witness_chain(dict(raw, mlp_2=changed), events, 1)

    def test_metadata_requires_exact_keys_and_coordinate_types(self):
        import copy

        step = {"name": "a", "operation": "capture", "module_path": "transformer.h.0.attn",
                "invocation": 0, "batch_index": 0, "token_index": 0}
        event = {"name": "a", "operation": "capture", "order": 0, "module_path": "transformer.h.0.attn",
                 "module_aliases": ["transformer.h.0.attn"], "call_site_known": False, "forward_index": 0,
                 "invocation": 0, "batch_index": 0, "token_index": 0, "output_selector": [0],
                 "output_shape": [1, 1, 768], "selected_shape": [768], "dtype": "torch.float32",
                 "device": "cpu", "source_capture": None, "preceding_same_site_actions": []}
        _check_plan_metadata([step], [event], 1)
        for key in ("order", "invocation", "batch_index", "token_index"):
            invalid = copy.deepcopy(event)
            invalid[key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                _check_plan_metadata([step], [invalid], 1)
        invalid = copy.deepcopy(event)
        invalid["output_shape"][0] = True
        with self.assertRaises(ValueError):
            _check_plan_metadata([step], [invalid], 1)
        invalid = copy.deepcopy(event)
        del invalid["source_capture"]
        with self.assertRaises(ValueError):
            _check_plan_metadata([step], [invalid], 1)
        invalid = copy.deepcopy(event)
        invalid["extra"] = None
        with self.assertRaises(ValueError):
            _check_plan_metadata([step], [invalid], 1)
        for key in ("module_aliases", "output_selector", "output_shape", "selected_shape",
                    "preceding_same_site_actions"):
            invalid = copy.deepcopy(event)
            invalid[key] = tuple(invalid[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                _check_plan_metadata([step], [invalid], 1)

    def test_full_output_witness_detects_outside_token_modifications(self):
        zero = struct.pack("<768I", *((0,) * 768))
        direction = struct.pack("<768I", *injection_direction_bits())
        pre, post = zero * 3, zero + direction + zero
        self.assertEqual(verify_action_scope(pre, post, token=1, operation="add")["changed_coordinates"], 768)
        self.assertEqual(verify_action_scope(pre, post, token=1, operation="patch", donor=direction), {})
        with self.assertRaises(ValueError):
            verify_action_scope(pre, direction + direction + zero, token=1, operation="add")
        with self.assertRaises(ValueError):
            verify_action_scope(pre, post, token=0, operation="patch", donor=direction)
        with self.assertRaises(ValueError):
            verify_action_scope(pre, post, token=1, operation="null")
        bad = bytearray(post)
        bad[-4:] = struct.pack("<I", 0x7f800000)
        with self.assertRaises(ValueError):
            verify_action_scope(pre, bytes(bad), token=1, operation="add")

    def test_same_site_injects_before_patch_and_observes_after_both(self):
        donor = struct.pack("<768I", *((0,) * 768))
        steps = _plan_steps(site=2, token=4, condition="clean_to_injected", layer=2, donor=donor)
        names = [step["name"] for step in steps]
        selected = [name for name in names if name.startswith(("inject_", "action_")) or name == "mlp_2"]
        self.assertEqual(selected, ["inject_before", "inject_add", "inject_after", "action_before",
                                    "action_patch", "action_after", "mlp_2"])
        for step in steps:
            if step["name"] in {"inject_before", "inject_after", "action_before", "action_after"}:
                self.assertNotIn("batch_index", step)
                self.assertNotIn("token_index", step)
        self.assertEqual([step["name"] for step in steps if step["name"].startswith("attention_")],
                         [f"attention_{layer}" for layer in range(6)])

    def test_reverse_is_clean_and_actions_follow_execution_order(self):
        donor = struct.pack("<768I", *((0,) * 768))
        reverse = _plan_steps(site=3, token=1, condition="injected_to_clean", layer=1, donor=donor)
        self.assertFalse(any(step["name"].startswith("inject_") for step in reverse))
        self.assertLess([step["name"] for step in reverse].index("action_patch"),
                        [step["name"] for step in reverse].index("mlp_1"))
        forward = _plan_steps(site=3, token=1, condition="clean_to_injected", layer=1, donor=donor)
        names = [step["name"] for step in forward]
        self.assertLess(names.index("action_patch"), names.index("inject_add"))
        late = _plan_steps(site=1, token=1, condition="downstream", donor=donor)
        names = [step["name"] for step in late]
        self.assertLess(names.index("inject_add"), names.index("action_patch"))
        self.assertLess(names.index("action_patch"), names.index("block_5"))

    def test_null_and_self_use_distinct_same_forward_actions(self):
        null = _plan_steps(site=1, token=0, condition="null")
        same = _plan_steps(site=1, token=0, condition="self")
        null_action = next(step for step in null if step["name"] == "action_null")
        self_action = next(step for step in same if step["name"] == "action_patch")
        self.assertEqual(null_action["operation"], "null")
        self.assertEqual(self_action["source_capture"], "self_source")
        self.assertLess([x["name"] for x in same].index("self_source"),
                        [x["name"] for x in same].index("action_patch"))

    def test_rejects_unsupported_site_selector_or_donor(self):
        for kwargs in ({"site": 0, "token": 0, "condition": "clean"},
                       {"site": 1, "token": 32, "condition": "clean"},
                       {"site": 1, "token": 0, "condition": "invalid"},
                       {"site": 1, "token": 0, "condition": "clean_to_injected", "layer": 5, "donor": b""},
                       {"site": 1, "token": 0, "condition": "injected_to_clean", "layer": 1, "donor": b""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _plan_steps(**kwargs)


class ToyAttention(nn.Module):
    def forward(self, hidden):
        return (hidden * 0.125,)


class ToyMLP(nn.Module):
    def forward(self, hidden):
        return hidden * 0.25


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden):
        hidden = hidden + self.attn(hidden)[0]
        return (hidden + self.mlp(hidden),)


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList(ToyBlock() for _ in range(6))

    def forward(self, hidden):
        for block in self.h:
            hidden = block(hidden)[0]
        return hidden


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = ToyTransformer()
        self.config = SimpleNamespace(n_layer=6, n_embd=768, vocab_size=50257,
                                      use_cache=False, _attn_implementation="eager")
        self.register_buffer("indices", torch.arange(50257, dtype=torch.int64) % 768)
        self.is_gradient_checkpointing = False
        self.calls = 0
        self.fail = False
        self.flip_denormal = False
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private failure")
        assert not torch.is_grad_enabled() and not use_cache
        assert bool(torch.all(attention_mask == 1))
        features = torch.arange(768, dtype=torch.float32).view(1, 1, -1) / 1024
        hidden = input_ids.to(torch.float32).unsqueeze(-1) + features
        hidden = self.transformer(hidden)
        if self.flip_denormal:
            torch.set_flush_denormal(True)
        return SimpleNamespace(logits=hidden[:, :, self.indices])


def serial_toy_reference(ids, *, site, condition, layer=None, donor=None):
    """Independent explicit residual arithmetic, with no hooks or plans."""
    def from_bits(raw):
        values = struct.unpack("<768I", raw)
        signed = [value if value < 1 << 31 else value - (1 << 32) for value in values]
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32)

    def raw(tensor):
        return tensor.contiguous().view(torch.uint8).numpy().tobytes()

    direction = from_bits(struct.pack("<768I", *injection_direction_bits()))
    injected = condition in {"injected", "clean_to_injected", "upstream", "downstream", "null", "self"}
    selected_patch = (0 if condition == "upstream" else layer) if condition in {
        "upstream", "clean_to_injected", "injected_to_clean"} else None
    hidden = torch.tensor([ids], dtype=torch.float32).unsqueeze(-1) + \
             torch.arange(768, dtype=torch.float32).view(1, 1, -1) / 1024
    attention, mlp = [], []
    for current in range(6):
        attention_value = hidden * 0.125
        attention.append(raw(attention_value[0, -1]))
        hidden = hidden + attention_value
        mlp_value = hidden * 0.25
        if injected and current == site:
            mlp_value = mlp_value.clone()
            mlp_value[0, -1] = mlp_value[0, -1] + direction
        if selected_patch == current:
            mlp_value = mlp_value.clone()
            mlp_value[0, -1] = from_bits(donor)
        mlp.append(raw(mlp_value[0, -1]))
        hidden = hidden + mlp_value
        if current == 5 and condition == "downstream":
            hidden = hidden.clone()
            hidden[0, -1] = from_bits(donor)
    indices = torch.arange(50257, dtype=torch.int64) % 768
    return tuple(attention), tuple(mlp), raw(hidden[0, -1]), raw(hidden[0, -1, indices])


class OrderedInjectionExecutorTests(unittest.TestCase):
    def test_composite_actions_land_and_clean_up_on_toy_model(self):
        model = ToyModel()
        with NumberInjectionExecutor(model, site=2) as executor:
            clean = executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
            injected = executor.forward([2, 3], request_id=f"{2:032x}", condition="injected")
            self.assertFalse(compare_f32_le(clean.logits, injected.logits)["within_tolerance"])
            restored = executor.forward([2, 3], request_id=f"{3:032x}", condition="clean_to_injected", layer=2,
                                        donor=clean.mlp_donor(2))
            reverse = executor.forward([2, 3], request_id=f"{4:032x}", condition="injected_to_clean", layer=2,
                                       donor=injected.mlp_donor(2))
            upstream = executor.forward([2, 3], request_id=f"{5:032x}", condition="upstream",
                                        donor=clean.mlp_donor(0))
            downstream = executor.forward([2, 3], request_id=f"{6:032x}", condition="downstream",
                                          donor=clean.block_donor())
            null = executor.forward([2, 3], request_id=f"{7:032x}", condition="null")
            same = executor.forward([2, 3], request_id=f"{8:032x}", condition="self")
            references = ((clean, "clean", None, None), (injected, "injected", None, None),
                          (restored, "clean_to_injected", 2, clean.mlp[2]),
                          (reverse, "injected_to_clean", 2, injected.mlp[2]),
                          (upstream, "upstream", None, clean.mlp[0]),
                          (downstream, "downstream", None, clean.block),
                          (null, "null", None, None), (same, "self", None, None))
            for observation, condition, layer, donor in references:
                expected = serial_toy_reference([2, 3], site=2, condition=condition, layer=layer, donor=donor)
                self.assertEqual((observation.attention, observation.mlp, observation.block, observation.logits),
                                 expected, condition)
            for observation in (restored, downstream):
                self.assertTrue(compare_f32_le(observation.logits, clean.logits)["within_tolerance"])
            for observation in (reverse, upstream, null, same):
                self.assertTrue(compare_f32_le(observation.logits, injected.logits)["within_tolerance"])
            self.assertEqual(len(clean.attention), 6)
            self.assertEqual(len(clean.mlp), 6)
            expected_paths = [name for i in range(6)
                              for name in (f"transformer.h.{i}.attn", f"transformer.h.{i}.mlp")]
            self.assertEqual([item.module_path for item in injected.trace], expected_paths + ["transformer.h.5"])
            self.assertEqual([item.capture_order for item in injected.trace], list(range(13)))
            self.assertEqual(injected.trace[0].output_shape, (1, 2, 768))
            self.assertEqual(injected.trace[0].selected_shape, (768,))
            self.assertEqual(injected.trace[0].output_selector, (0,))
            self.assertEqual(injected.trace[1].output_selector, ())
            self.assertEqual(clean.input_sha256, injected.input_sha256)
            self.assertIsNone(clean.action)
            self.assertIsNone(injected.action)
            for observation, operation, path, selector, donor in (
                (restored, "patch", "transformer.h.2.mlp", (), clean.mlp_donor(2).identity),
                (reverse, "patch", "transformer.h.2.mlp", (), injected.mlp_donor(2).identity),
                (upstream, "patch", "transformer.h.0.mlp", (), clean.mlp_donor(0).identity),
                (downstream, "patch", "transformer.h.5", (0,), clean.block_donor().identity),
                (null, "null", "transformer.h.1.mlp", (), None),
                (same, "patch", "transformer.h.1.mlp", (), same.mlp_donor(1).identity),
            ):
                action = observation.action
                self.assertEqual(action.operation, operation)
                self.assertEqual(action.module_path, path)
                self.assertEqual((action.invocation, action.batch_index, action.token_index), (0, 0, 1))
                self.assertEqual(action.output_selector, selector)
                self.assertEqual(action.output_shape, (1, 2, 768))
                self.assertEqual(action.selected_shape, (768,))
                self.assertEqual((action.dtype, action.device), ("torch.float32", "cpu"))
                self.assertEqual(action.donor_identity, donor)
                self.assertFalse(hasattr(action, "site"))
                self.assertFalse(hasattr(action, "order"))
            self.assertEqual(len(executor.completed()), 8)
            self.assertGreater(executor.retained_tensor_bytes, 0)
            self.assertTrue(any(item.operation == "add" for item in executor.private_evidence()))
            evidence = next(item for item in executor.private_evidence()
                            if item.request_id == f"{3:032x}" and item.operation == "patch")
            self.assertEqual(evidence.donor_identity.request_id, clean.request_id)
            self.assertEqual(evidence.input_sha256, clean.input_sha256)
            self.assertEqual(evidence.donor_identity.module_path, "transformer.h.2.mlp")
            self.assertEqual(evidence.donor_identity.output_selector, ())
            block_donor = clean.block_donor()
            self.assertIs(model.get_submodule(block_donor.identity.module_path), model.transformer.h[5])
            self.assertEqual(block_donor.identity.output_selector, (0,))
            self.assertEqual(block_donor.identity.invocation, 0)
            self.assertEqual(block_donor.identity.token_index, 1)
            add = next(item for item in executor.private_evidence()
                       if item.request_id == injected.request_id and item.operation == "add")
            self.assertEqual(len(add.realized_delta_f64_le), 768 * 8)
            self.assertTrue(any(value != 0 for value in struct.unpack("<768d", add.realized_delta_f64_le)))
            self.assertFalse(hasattr(injected, "site"))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))

    def test_public_action_descriptor_is_independent_of_hidden_site(self):
        actions = []
        for site in range(1, 5):
            with self.subTest(site=site), NumberInjectionExecutor(ToyModel(), site=site) as executor:
                clean = executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
                patched = executor.forward([2, 3], request_id=f"{2:032x}", condition="clean_to_injected",
                                           layer=3, donor=clean.mlp_donor(3))
                actions.append(patched.action)
        self.assertTrue(all(action == actions[0] for action in actions))

    def test_budget_rejects_before_model_execution(self):
        model = ToyModel()
        with NumberInjectionExecutor(model, site=1, max_study_bytes=200000) as executor:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
            self.assertEqual(model.calls, 0)
            self.assertEqual(executor.retained_tensor_bytes, 0)
            self.assertTrue(executor.poisoned)
            self.assertEqual(executor.failures()[0]["phase"], "budget")

    def test_failure_is_sanitized_and_invalidates_the_executor(self):
        model = ToyModel()
        with NumberInjectionExecutor(model, site=1) as executor:
            executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
            model.fail = True
            with self.assertRaisesRegex(RuntimeError, "forward failed") as caught:
                executor.forward([2, 3], request_id=f"{2:032x}", condition="injected")
            self.assertNotIn("private failure", str(caught.exception))
            self.assertTrue(executor.poisoned)
            self.assertEqual(len(executor.completed()), 1)
            self.assertEqual(executor.failures()[0]["partial_forward_captures"], "discarded")
            self.assertEqual(executor.private_failures()[0].exception_type, "RuntimeError")
            self.assertEqual(executor.private_failures()[0].exception_message, "private failure")
            self.assertNotIn("private failure", str(executor.failures()))
            with self.assertRaises(RuntimeError):
                executor.forward([2, 3], request_id=f"{3:032x}", condition="clean")

    def test_existing_hook_and_invalid_inputs_are_rejected(self):
        model = ToyModel()
        handle = model.transformer.h[0].mlp.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaises(ValueError):
                with NumberInjectionExecutor(model, site=1):
                    pass
        finally:
            handle.remove()
        for ids in ([], [True], [-1], [50257], [1] * 33):
            candidate = ToyModel()
            with self.subTest(ids=ids), NumberInjectionExecutor(candidate, site=1) as executor:
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    executor.forward(ids, request_id=f"{1:032x}", condition="clean")
                self.assertEqual(candidate.calls, 0)
                self.assertTrue(executor.poisoned)
                self.assertEqual(executor.failures()[0]["phase"], "request")

    def test_donor_must_come_from_matching_input_site_and_prior_condition(self):
        for scenario in ("input", "boundary", "source_condition", "reverse_condition", "duplicate", "typed_identity"):
            model = ToyModel()
            with self.subTest(scenario=scenario), NumberInjectionExecutor(model, site=2) as executor:
                clean = executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
                injected = executor.forward([2, 3], request_id=f"{2:032x}", condition="injected")
                ids, condition, donor = [2, 3], "clean_to_injected", clean.mlp_donor(2)
                if scenario == "input":
                    ids = [2, 4]
                elif scenario == "boundary":
                    donor = clean.mlp_donor(1)
                elif scenario == "source_condition":
                    donor = injected.mlp_donor(2)
                elif scenario == "reverse_condition":
                    condition = "injected_to_clean"
                elif scenario == "typed_identity":
                    donor = replace(donor, identity=replace(donor.identity, batch_index=False))
                request_id = f"{1 if scenario == 'duplicate' else 3:032x}"
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    executor.forward(ids, request_id=request_id, condition=condition, layer=2, donor=donor)
                self.assertEqual(model.calls, 2)
                self.assertTrue(executor.poisoned)

    def test_input_is_frozen_before_digest_and_model_tensor_construction(self):
        model = ToyModel()
        ids = [2, 3]
        original_tensor = torch.tensor
        def mutate_before_tensor(data, *args, **kwargs):
            if (type(data) is list and len(data) == 1 and data[0] == (2, 3)
                    and kwargs.get("dtype") is torch.int64):
                ids[:] = [9, 9]
            return original_tensor(data, *args, **kwargs)
        with NumberInjectionExecutor(model, site=2) as executor:
            with patch.object(torch, "tensor", side_effect=mutate_before_tensor):
                observation = executor.forward(ids, request_id=f"{1:032x}", condition="clean")
            self.assertEqual(ids, [9, 9])
            expected = serial_toy_reference([2, 3], site=2, condition="clean")
            self.assertEqual(observation.logits, expected[3])
            digest = hashlib.sha256(b"MLG-number-input-v1\0" + bytes([2]) +
                                    (2).to_bytes(4, "big") + (3).to_bytes(4, "big")).hexdigest()
            self.assertEqual(observation.input_sha256, digest)

    def test_interrupted_acquisition_releases_lease(self):
        model = ToyModel()
        executor = NumberInjectionExecutor(model, site=1)
        acquire = executor._lease.acquire
        def interrupt_after_acquire():
            acquire()
            raise KeyboardInterrupt()
        with patch.object(executor._lease, "acquire", side_effect=interrupt_after_acquire):
            with self.assertRaises(KeyboardInterrupt):
                executor.__enter__()
        with NumberInjectionExecutor(model, site=1):
            pass

    def test_second_enter_and_foreign_exit_cannot_release_active_owner(self):
        model = ToyModel()
        executor = NumberInjectionExecutor(model, site=1)
        acquired, resume, inside, finish = (threading.Event() for _ in range(4))
        errors = []
        acquire = executor._lease.acquire
        def paused_acquire():
            acquire()
            acquired.set()
            if not resume.wait(10):
                raise RuntimeError("test entry did not resume")
        def owner():
            try:
                with executor:
                    inside.set()
                    if not finish.wait(10):
                        raise RuntimeError("test owner did not finish")
            except BaseException as error:
                errors.append(error)
        with patch.object(executor._lease, "acquire", side_effect=paused_acquire):
            thread = threading.Thread(target=owner)
            thread.start()
            try:
                self.assertTrue(acquired.wait(10))
                with self.assertRaises(RuntimeError):
                    executor.__enter__()
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
                resume.set()
                self.assertTrue(inside.wait(10))
                with self.assertRaises(RuntimeError):
                    executor.__exit__(None, None, None)
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
            finally:
                resume.set()
                finish.set()
                thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        lease = ModelLease(model)
        lease.acquire()
        lease.release()

    def test_cleanup_failure_preserves_body_exception(self):
        model = ToyModel()
        executor = NumberInjectionExecutor(model, site=1)
        executor.__enter__()
        original = ValueError("original body error")
        with patch.object(executor._lease, "_quarantine", side_effect=RuntimeError("cleanup error")):
            with self.assertRaises(ValueError) as caught:
                executor.__exit__(ValueError, original, None)
        self.assertIs(caught.exception, original)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "cleanup error")

    def test_system_exit_type_and_code_are_preserved(self):
        model = ToyModel()
        with NumberInjectionExecutor(model, site=1) as executor:
            with patch.object(model, "forward", side_effect=SystemExit(17)):
                with self.assertRaises(SystemExit) as caught:
                    executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
            self.assertEqual(caught.exception.code, 17)
            self.assertTrue(executor.poisoned)

    def test_all_candidate_orders_and_hidden_layers_restore_at_the_injected_site(self):
        for site in range(1, 5):
            with self.subTest(site=site), NumberInjectionExecutor(ToyModel(), site=site) as executor:
                clean = executor.forward([2, 3], request_id=f"{1:032x}", condition="clean")
                injected = executor.forward([2, 3], request_id=f"{2:032x}", condition="injected")
                for layer in range(1, 5):
                    patched = executor.forward([2, 3], request_id=f"{2 * layer + 1:032x}",
                                               condition="clean_to_injected", layer=layer,
                                               donor=clean.mlp_donor(layer))
                    reverse = executor.forward([2, 3], request_id=f"{2 * layer + 2:032x}",
                                               condition="injected_to_clean", layer=layer,
                                               donor=injected.mlp_donor(layer))
                    if layer == site:
                        self.assertTrue(compare_f32_le(clean.logits, patched.logits)["within_tolerance"])
                        self.assertTrue(compare_f32_le(injected.logits, reverse.logits)["within_tolerance"])

    def test_32_token_composite_snapshots_stay_within_forward_payload_cap(self):
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            ids = [2] * 32
            clean = executor.forward(ids, request_id=f"{1:032x}", condition="clean")
            old = executor.retained_tensor_bytes
            executor.forward(ids, request_id=f"{2:032x}", condition="clean_to_injected", layer=2,
                             donor=clean.mlp_donor(2))
            retained = executor.retained_tensor_bytes - old
            self.assertGreaterEqual(retained, 4 * 32 * 768 * 4 + 50257 * 4)
            self.assertLessEqual(retained, 1024 * 1024)

    def test_post_forward_validation_failure_keeps_available_private_witnesses(self):
        with NumberInjectionExecutor(ToyModel(), site=2) as executor:
            with patch("looking_glass.pretrained_number_injection.verify_action_scope", side_effect=ValueError("private")):
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    executor.forward([2, 3], request_id=f"{1:032x}", condition="injected")
            self.assertTrue(executor.poisoned)
            self.assertEqual(executor.failures()[0]["partial_forward_captures"], "retained_privately")
            failure = executor.private_failures()[0]
            self.assertIn("inject_before", dict(failure.captures))
            self.assertIn("inject_after", dict(failure.captures))
            self.assertEqual(len(failure.logits), 50257 * 4)

    def test_failure_recording_error_keeps_original_and_quarantines(self):
        model = ToyModel()
        model.fail = True
        with NumberInjectionExecutor(model, site=2) as executor:
            with patch("looking_glass.pretrained_number_injection.json.dumps", side_effect=MemoryError("record OOM")):
                with self.assertRaisesRegex(RuntimeError, "recording also failed") as caught:
                    executor.forward([2, 3], request_id=f"{1:032x}", condition="injected")
            original = caught.exception.__cause__
            self.assertIsInstance(original, RuntimeError)
            self.assertEqual(str(original), "private failure")
            self.assertIsInstance(original.__cause__, MemoryError)
            self.assertTrue(executor.poisoned)

    def test_denormal_mode_change_during_forward_is_terminal(self):
        if not torch.set_flush_denormal(False):
            self.skipTest("runtime does not expose denormal flushing")
        model = ToyModel()
        model.flip_denormal = True
        try:
            with NumberInjectionExecutor(model, site=2) as executor:
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    executor.forward([2, 3], request_id=f"{1:032x}", condition="injected")
                self.assertTrue(executor.poisoned)
                self.assertEqual(executor.failures()[0]["phase"], "runtime_after")
                self.assertEqual(executor.failures()[0]["partial_forward_captures"], "retained_privately")
        finally:
            torch.set_flush_denormal(False)


if __name__ == "__main__":
    unittest.main()
