"""Synthetic CPU-only custodian tests; no checkpoint or reserved inputs."""
import types
import hashlib
import json
from threading import Event
import unittest
from unittest import mock

import torch
from torch import nn

from looking_glass import pretrained_causal_experiment as study
from looking_glass import pretrained_custodian as custodian


class SyntheticMLP(nn.Module):
    def forward(self, value):
        return value * 0.03125 + 0.125


class SyntheticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SyntheticMLP()

    def forward(self, value):
        return (value + self.mlp(value),)


class SyntheticModel(nn.Module):
    """Untrained arithmetic test double with protocol-compatible shapes."""
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([SyntheticBlock() for _ in range(6)])
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.config = types.SimpleNamespace(n_layer=6, n_embd=768, vocab_size=50257, use_cache=False,
                                            _attn_implementation="eager")
        self.is_gradient_checkpointing = False
        self.fail = False
        self.mutate_input = False
        self.interrupt = False
        self.cancel_event = None
        self.zero_logits = False
        self.forward_calls = 0
        self.eval()

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        self.forward_calls += 1
        if self.fail:
            raise RuntimeError("PRIVATE-SITE-DETAILS")
        if self.interrupt:
            raise KeyboardInterrupt("PRIVATE-SITE-DETAILS")
        if self.mutate_input:
            input_ids[0, 0] += 1
        if self.cancel_event is not None:
            self.cancel_event.set()
        offsets = torch.arange(768, dtype=torch.float32).reshape(1, 1, 768) / 4096
        value = input_ids.unsqueeze(-1).float() / 1024 + offsets * self.weight
        for block in self.transformer.h:
            value = block(value)[0]
        logits = value[..., 0:1].expand(-1, -1, 50257).clone()
        logits[..., 257] = value[..., 1]
        if self.zero_logits:
            logits.zero_()
        return types.SimpleNamespace(logits=logits, past_key_values=None)


class DevelopmentCustodianTests(unittest.TestCase):
    def test_all_controls_and_same_site_order_with_synthetic_model(self):
        for site in (1, 3):
            with self.subTest(site=site):
                model = SyntheticModel()
                owner = custodian.DevelopmentCustodian(model, synthetic_site=site)
                case = owner.run_case(index=0, input_ids=torch.tensor([[11, 12, 13]], dtype=torch.long))
                analysis = study.analyze_discovery_case(case, token_index=2, source_device="cpu")
                self.assertEqual(analysis["first_observed_layer"], site)
                self.assertEqual(analysis["first_difference"], site)
                self.assertEqual(case.conditions["injected"].events[site].module_path,
                                 f"transformer.h.{site}.mlp")
                self.assertTrue(torch.equal(case.conditions[f"patch_{site}"].events[site].value,
                                            case.conditions["injected"].events[site].value))
                self.assertTrue(torch.equal(case.conditions["late"].logits, case.conditions["clean"].logits))
                self.assertEqual(len(case.conditions), 15)
                self.assertEqual(len(case.post_clean), 13)
                self.assertEqual(len(owner.private_evidence()), 28)
                self.assertTrue(all(row["site"] == site for row in owner.private_evidence()))
                evidence = owner.private_evidence()
                injection_evidence = [row for row in evidence if "pre_f32le" in row]
                self.assertEqual(len(injection_evidence), 9)
                self.assertIs(type(injection_evidence[0]["pre_f32le"]), bytes)
                private_meta = json.loads(evidence[0]["plan_metadata_json"])
                self.assertEqual(private_meta["case_index"], 0)
                self.assertEqual(private_meta["forward_label"], "clean")
                expected_input_sha = hashlib.sha256(torch.tensor([[11, 12, 13]], dtype=torch.long).numpy().astype("<i8").tobytes()).hexdigest()
                self.assertEqual(private_meta["input_sha256"], expected_input_sha)
                self.assertEqual(len({json.loads(row["plan_metadata_json"])["forward_label"] for row in evidence}), 28)
                self.assertEqual(len([row for row in evidence if "control_after_f32le" in row]), 12)
                evidence[0]["site"] = 99
                self.assertEqual(owner.private_evidence()[0]["site"], site)
                self.assertEqual(case.conditions["clean"].events[6].output_selector, (0,))
                self.assertTrue(all(event.invocation == 0 and event.token_index == 2 and
                                    event.source_device == "cpu" for event in case.conditions["injected"].events))
                self.assertFalse(owner.poisoned)
                self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                    for module in model.modules()))

    def test_failed_forward_is_sanitized_and_quarantines_owner(self):
        model = SyntheticModel()
        owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
        model.fail = True
        with self.assertRaises(custodian.CustodianExecutionError) as caught:
            owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(owner.poisoned)
        model.fail = False
        with self.assertRaises(custodian.CustodianExecutionError):
            owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    def test_cpu_only_input_validation_and_trace_cap_fail_closed(self):
        owner = custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=1, max_study_bytes=1)
        with self.assertRaises(custodian.CustodianExecutionError):
            owner.run_case(index=0, input_ids=torch.tensor([[1]], dtype=torch.long))
        self.assertTrue(owner.poisoned)
        with self.assertRaises(ValueError):
            custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=True)

    def test_input_mutation_and_cancellation_are_quarantined(self):
        for flag, expected in (("mutate_input", custodian.CustodianExecutionError),
                               ("interrupt", custodian.CustodianCancelled)):
            with self.subTest(flag=flag):
                model = SyntheticModel()
                owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
                setattr(model, flag, True)
                with self.assertRaises(expected) as caught:
                    owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))
                self.assertTrue(owner.poisoned)
                self.assertNotIn("PRIVATE", str(caught.exception))
                self.assertIsNone(caught.exception.__context__)
                if flag == "interrupt":
                    # An interrupted CPU forward can retain hooks/lease; never
                    # reuse this model or owner without a proven recovery path.
                    self.assertTrue(any(module._forward_hooks or module._forward_pre_hooks
                                        for module in model.modules()))
                    with self.assertRaises(custodian.CustodianExecutionError):
                        owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    def test_cooperative_cancel_stops_before_the_next_forward(self):
        event = Event()
        event.set()
        model = SyntheticModel()
        owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
        with self.assertRaises(custodian.CustodianCancelled):
            owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long), cancel_event=event)
        self.assertEqual(model.forward_calls, 0)
        self.assertTrue(owner.poisoned)
        model = SyntheticModel()
        event = Event()
        model.cancel_event = event
        owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
        with self.assertRaises(custodian.CustodianCancelled):
            owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long), cancel_event=event)
        self.assertEqual(model.forward_calls, 1)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))

    def test_post_action_capture_catches_an_ignored_patch_even_when_logits_are_equal(self):
        model = SyntheticModel()
        model.zero_logits = True
        owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
        original = custodian.ActivationPlan._apply_step
        def ignore_patch(plan, step, output, preceding):
            replacement, changed = original(plan, step, output, preceding)
            return (output, False) if step["operation"] == "patch" else (replacement, changed)
        with mock.patch.object(custodian.ActivationPlan, "_apply_step", new=ignore_patch):
            with self.assertRaises(custodian.CustodianExecutionError):
                owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))
        self.assertTrue(owner.poisoned)

    def test_self_patch_source_must_match_the_pre_action_observation(self):
        model = SyntheticModel()
        model.zero_logits = True
        owner = custodian.DevelopmentCustodian(model, synthetic_site=2)
        original = custodian.ActivationPlan._apply_step
        def corrupt_self_source(plan, step, output, preceding):
            result = original(plan, step, output, preceding)
            if step["name"] == "self_source":
                plan._captures["self_source"][0] += 1.0
            return result
        with mock.patch.object(custodian.ActivationPlan, "_apply_step", new=corrupt_self_source):
            with self.assertRaises(custodian.CustodianExecutionError):
                owner.run_case(index=0, input_ids=torch.tensor([[1, 2]], dtype=torch.long))
        row = owner.private_evidence()[-1]
        self.assertEqual(row["condition"], "self_patch")
        self.assertFalse(row["control_integrity_verified"])
        self.assertEqual(owner.private_failures()[-1]["error_type"], "ValueError")

    def test_private_integrity_failure_retains_available_pre_post_and_diagnostic(self):
        owner = custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=2)
        with mock.patch.object(study, "verify_injection_integrity", side_effect=ValueError("PRIVATE INTEGRITY ERROR")):
            with self.assertRaises(custodian.CustodianExecutionError) as caught:
                owner.run_case(index=3, input_ids=torch.tensor([[4, 5]], dtype=torch.long))
        self.assertNotIn("PRIVATE", str(caught.exception))
        row = owner.private_evidence()[-1]
        self.assertEqual(row["condition"], "injected")
        self.assertFalse(row["injection_integrity_verified"])
        self.assertIs(type(row["pre_f32le"]), bytes)
        self.assertIs(type(row["post_f32le"]), bytes)
        failure = owner.private_failures()[-1]
        self.assertEqual(failure["case_index"], 3)
        self.assertEqual(failure["forward_label"], "injected")
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertIn("PRIVATE", failure["error_message"])
        self.assertTrue(failure["captures_available"])

    def test_invalid_metadata_retains_unverified_private_captures(self):
        owner = custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=2)
        getter = custodian.ActivationPlan.metadata.fget
        def wrong_shape(plan):
            metadata = getter(plan)
            if any(step["name"] == "private_before" for step in metadata["steps"]):
                metadata["steps"][0]["output_shape"][0] = True
            return metadata
        with mock.patch.object(custodian.ActivationPlan, "metadata", new=property(wrong_shape)):
            with self.assertRaises(custodian.CustodianExecutionError):
                owner.run_case(index=1, input_ids=torch.tensor([[4, 5]], dtype=torch.long))
        row = owner.private_evidence()[-1]
        self.assertEqual(row["condition"], "injected")
        self.assertIs(type(row["pre_f32le"]), bytes)
        self.assertIs(type(row["post_f32le"]), bytes)
        self.assertFalse(row["injection_integrity_verified"])
        self.assertNotIn("plan_metadata_json", row)
        self.assertTrue(owner.private_failures()[-1]["captures_available"])

    def test_invalid_order_or_forward_metadata_is_not_normalized(self):
        getter = custodian.ActivationPlan.metadata.fget
        for key, bad in (("order", True), ("forward_index", False),
                         ("preceding_same_site_actions", ["unrecorded"])):
            with self.subTest(key=key):
                owner = custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=2)
                def bad_metadata(plan):
                    metadata = getter(plan)
                    if any(step["name"] == "private_before" for step in metadata["steps"]):
                        metadata["steps"][0][key] = bad
                    return metadata
                with mock.patch.object(custodian.ActivationPlan, "metadata", new=property(bad_metadata)):
                    with self.assertRaises(custodian.CustodianExecutionError):
                        owner.run_case(index=1, input_ids=torch.tensor([[4, 5]], dtype=torch.long))
                row = owner.private_evidence()[-1]
                self.assertEqual(row["condition"], "injected")
                self.assertFalse(row["injection_integrity_verified"])
                self.assertIs(type(row["pre_f32le"]), bytes)

    def test_non_eager_attention_configuration_is_rejected(self):
        model = SyntheticModel()
        model.config._attn_implementation = "sdpa"
        with self.assertRaises(ValueError):
            custodian.DevelopmentCustodian(model, synthetic_site=1)

    def test_global_hooks_are_rejected_and_test_hooks_removed(self):
        module = torch.nn.modules.module
        for register in (module.register_module_forward_pre_hook, module.register_module_forward_hook,
                         module.register_module_full_backward_pre_hook, module.register_module_full_backward_hook):
            with self.subTest(register=register.__name__):
                handle = register(lambda *args: None)
                try:
                    with self.assertRaises(ValueError):
                        custodian.DevelopmentCustodian(SyntheticModel(), synthetic_site=1)
                finally:
                    handle.remove()


if __name__ == "__main__":
    unittest.main()
