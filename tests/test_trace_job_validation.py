"""CPU-only batch coverage and metadata validation for one GPT-2 job."""
from dataclasses import replace
import importlib.util
import gc
import sys
import unittest
from unittest.mock import patch
import weakref

import torch

from looking_glass.capture_dispatcher import CaptureDispatcher, CapturedSlice
from looking_glass.trace_job_validation import make_trace_job_layout
from looking_glass.traced_broker_worker import (
    BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec, bind_trace_plan,
)


class TraceJobValidationTests(unittest.TestCase):
    def setUp(self):
        identity = BrokerJobIdentity("a" * 32, 7, "team", "client-request")
        samples = (SampleTrace("sample-a", (0, 2)), SampleTrace("sample-b", (1, 2)))
        sites = (SiteTrace(0, "attn"), SiteTrace(1, "mlp"))
        spec = TraceSpec("unique-run", samples, sites, 4, 12, 5, 4, 128)
        limits = TraceLimits(2, 3, 2, 2, 4, 128, 3)
        self.plan = bind_trace_plan(identity, spec, limits, batch_size=2,
                                    sequence_length=3, layer_count=2)
        self.layout = make_trace_job_layout(self.plan, batch_size=2, sequence_length=3,
                                            hidden_width=4, layer_count=2,
                                            max_slots=8, max_event_bytes=64)

    def events(self):
        dispatcher = CaptureDispatcher((self.plan.capture_request(),), batch_size=2,
                                       sequence_length=3, max_events=4,
                                       max_trace_bytes=128,
                                       sequence_starts=self.plan.sequence_starts())
        source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        def forward():
            dispatcher.record("transformer.h.0.attn", layer_index=0, module_kind="attn",
                              output_path=(0,), output=source)
            dispatcher.record("transformer.h.1.mlp", layer_index=1, module_kind="mlp",
                              output_path=(), output=source + 1)
        _, events = dispatcher.run(forward)
        return events

    def test_expected_slots_bytes_and_complete_metadata(self):
        self.assertEqual(len(self.layout.slots), 4)
        self.assertEqual(self.layout.total_payload_bytes, 128)
        self.assertEqual([slot.payload_bytes for slot in self.layout.slots], [32] * 4)
        events = self.events()
        metadata = self.layout.validate(tuple(reversed(events)))
        self.assertEqual([item.sequence for item in metadata], [12, 13, 14, 15])
        self.assertEqual([item.identity.job_id for item in metadata],
                         [self.plan.worker_request_id] * 4)
        self.assertEqual([item.batch_positions for item in metadata], [(0,), (1,), (0,), (1,)])
        self.assertEqual([item.token_positions for item in metadata],
                         [(0, 2), (1, 2), (0, 2), (1, 2)])
        self.assertTrue(all(item.generation.generation_step == 5
                            and item.generation.stage == "unknown"
                            and item.generation.cache_position is None for item in metadata))

    def test_layout_isolated_from_later_plan_and_returned_slot_mutation(self):
        events = self.events()
        exported = self.layout.slots
        object.__setattr__(exported[0], "sample_id", "forged")
        object.__setattr__(self.plan, "_samples", (("other", (1,)),))
        self.assertEqual(self.layout.slots[0].sample_id, "sample-a")
        self.assertEqual(len(self.layout.validate(events)), 4)

    def test_oversized_plan_is_rejected_before_selection_accessors(self):
        object.__setattr__(self.plan, "_sites", ((0, "attn"),) * 100000)
        with patch("looking_glass.traced_broker_worker.BoundTracePlan.capture_request",
                   side_effect=AssertionError("must not copy selection")) as copy_request:
            with self.assertRaises(ValueError):
                make_trace_job_layout(self.plan, batch_size=2, sequence_length=3,
                                      hidden_width=4, layer_count=2,
                                      max_slots=8, max_event_bytes=64)
        copy_request.assert_not_called()

    def test_retained_validation_errors_do_not_hold_source_tensors_via_tracebacks(self):
        for mode in ("metadata", "finite", "schema", "interrupt", "list"):
            events = self.events()
            refs = [weakref.ref(event.value) for event in events]
            if mode == "metadata":
                inputs = (events[0], replace(events[1], sample_id="wrong"), *events[2:])
                manager = patch("looking_glass.trace_job_validation._finite_chunks",
                                new=lambda value: True)
            elif mode == "finite":
                inputs = events
                def fail_finite(value):
                    raise RuntimeError("injected finite-check failure")
                manager = patch("looking_glass.trace_job_validation.torch.isfinite", new=fail_finite)
            elif mode == "schema":
                inputs = events
                def fail_conversion(self):
                    raise RuntimeError("injected nested conversion failure")
                manager = patch.object(CapturedSlice, "to_trace_event", new=fail_conversion)
            elif mode == "interrupt":
                inputs = events
                def interrupt(value):
                    raise KeyboardInterrupt("injected interruption")
                manager = patch("looking_glass.trace_job_validation._finite_chunks", new=interrupt)
            else:
                inputs = list(events)
                manager = patch("looking_glass.trace_job_validation._finite_chunks",
                                new=lambda value: True)
            retained = None
            with manager:
                try:
                    self.layout.validate(inputs)
                except (ValueError, RuntimeError, KeyboardInterrupt) as error:
                    retained = error
            self.assertIsNotNone(retained, mode)
            events = inputs = None
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs), mode)
            self.assertIsNotNone(retained.__traceback__)
            self.assertIsNone(retained.__context__)
            self.assertIsNone(retained.__cause__)

    def test_exception_owned_payload_is_not_mistaken_for_disposal(self):
        events = self.events()
        ref = weakref.ref(events[0].value)
        def fail(value):
            error = RuntimeError("injected payload-owning failure")
            error.payload = value
            raise error
        retained = None
        with patch("looking_glass.trace_job_validation._finite_chunks", new=fail):
            try:
                self.layout.validate(events)
            except RuntimeError as error:
                retained = error
        events = None
        gc.collect()
        self.assertIsNotNone(ref())
        retained.payload = None
        gc.collect()
        self.assertIsNone(ref())

    def test_outer_exception_is_not_attached_to_validation_failure(self):
        events = self.events()
        retained = None
        try:
            raise RuntimeError("unrelated outer failure")
        except RuntimeError:
            try:
                self.layout.validate(list(events))
            except ValueError as error:
                retained = error
        self.assertIsNotNone(retained)
        self.assertIsNone(retained.__context__)
        self.assertIsNone(retained.__cause__)

    @unittest.skipUnless(sys.version_info >= (3, 11), "requires built-in exception groups")
    def test_retained_exception_group_does_not_hold_child_tracebacks(self):
        events = self.events()
        refs = [weakref.ref(event.value) for event in events]
        def fail(value):
            try:
                raise RuntimeError("inner failure")
            except RuntimeError as inner:
                raise ExceptionGroup("nested failure", [inner])
        retained = None
        with patch("looking_glass.trace_job_validation._finite_chunks", new=fail):
            try:
                self.layout.validate(events)
            except ExceptionGroup as error:
                retained = error
        events = None
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIsNotNone(retained)
        self.assertIsNone(retained.exceptions[0].__traceback__)

    @unittest.skipUnless(sys.version_info >= (3, 11), "requires built-in exception groups")
    def test_oversized_exception_group_still_requires_custody(self):
        events = self.events()
        ref = weakref.ref(events[0].value)
        def fail(value):
            try:
                raise RuntimeError("tensor-bearing child")
            except RuntimeError as inner:
                raise ExceptionGroup("large group", [RuntimeError("other") for _ in range(128)] + [inner])
        retained = None
        with patch("looking_glass.trace_job_validation._finite_chunks", new=fail):
            try:
                self.layout.validate(events)
            except ExceptionGroup as error:
                retained = error
        events = None
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIsNotNone(retained.exceptions[-1].__traceback__)
        retained = None
        gc.collect()
        self.assertIsNone(ref())

    def test_missing_duplicate_and_extra_events_fail_entire_batch(self):
        events = self.events()
        for candidate in (events[:-1], events + (events[0],),
                          (events[0], events[1], events[2], events[2])):
            with self.subTest(count=len(candidate)):
                with self.assertRaises(ValueError):
                    self.layout.validate(candidate)
        with self.assertRaises(ValueError):
            self.layout.validate((replace(events[0], sequence=13),
                                  replace(events[1], sequence=12), *events[2:]))

    def test_wrong_identity_coordinates_and_source_are_rejected(self):
        events = self.events()
        first = events[0]
        bad = (
            replace(first, run_id="other-run"), replace(first, job_id="other-job"),
            replace(first, request_id="other-request"), replace(first, sample_id="other"),
            replace(first, batch_position=1), replace(first, token_positions=(2, 0)),
            replace(first, forward_index=5), replace(first, generation_step=6),
            replace(first, sequence=99), replace(first, module_path="transformer.h.0.mlp"),
            replace(first, layer_index=1), replace(first, module_kind="block"),
            replace(first, output_path=()), replace(first, invocation=1),
            replace(first, source_shape=(2, 4, 4)), replace(first, source_dtype="torch.float64"),
            replace(first, source_device="cuda:0"), replace(first, saved_shape=(1, 4)),
            replace(first, saved_dtype="torch.float64"), replace(first, saved_device="cuda:0"),
            replace(first, token_positions=(False, 2)),
            replace(first, output_path=(False,)),
        )
        for event in bad:
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    self.layout.validate((event, *events[1:]))

    def test_rejects_aliases_views_invalid_payload_and_nonfinite_values(self):
        events = self.events()
        aliased = replace(events[1], value=events[0].value)
        with self.assertRaises(ValueError):
            self.layout.validate((events[0], aliased, *events[2:]))
        backing = bytearray(48)
        first = torch.frombuffer(memoryview(backing)[:32], dtype=torch.float32).reshape(2, 4)
        second = torch.frombuffer(memoryview(backing)[16:48], dtype=torch.float32).reshape(2, 4)
        self.assertNotEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        with self.assertRaises(ValueError):
            self.layout.validate((replace(events[0], value=first),
                                  replace(events[1], value=second), *events[2:]))
        oversized_view = torch.arange(64, dtype=torch.float32)[:8].reshape(2, 4)
        for value in (oversized_view, torch.ones(2, 4, dtype=torch.float64),
                      torch.ones(2, 4, device="meta"), torch.ones(2, 4, requires_grad=True)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.layout.validate((replace(events[0], value=value), *events[1:]))
        with torch.inference_mode():
            inference_value = torch.ones(2, 4)
        with self.assertRaises(ValueError):
            self.layout.validate((replace(events[0], value=inference_value), *events[1:]))
        nonfinite = events[0].value.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.layout.validate((replace(events[0], value=nonfinite), *events[1:]))

    def test_finite_validation_uses_bounded_chunks(self):
        identity = BrokerJobIdentity("b" * 32, 1, "team", "request")
        spec = TraceSpec("run", (SampleTrace("sample", (0,)),), (SiteTrace(0, "mlp"),),
                         0, 0, None, 1, 20000)
        plan = bind_trace_plan(identity, spec, TraceLimits(1, 1, 1, 1, 1, 20000, 1),
                               batch_size=1, sequence_length=1, layer_count=1)
        layout = make_trace_job_layout(plan, batch_size=1, sequence_length=1,
                                       hidden_width=5000, layer_count=1,
                                       max_slots=1, max_event_bytes=20000)
        dispatcher = CaptureDispatcher((plan.capture_request(),), batch_size=1,
                                       sequence_length=1, max_events=1,
                                       max_trace_bytes=20000,
                                       sequence_starts=plan.sequence_starts())
        def forward():
            dispatcher.record("transformer.h.0.mlp", layer_index=0, module_kind="mlp",
                              output_path=(), output=torch.ones(1, 1, 5000))
        _, events = dispatcher.run(forward)
        real_isfinite = torch.isfinite
        with patch("looking_glass.trace_job_validation.torch.isfinite", wraps=real_isfinite) as finite:
            layout.validate(events)
        self.assertEqual([call.args[0].numel() for call in finite.call_args_list], [4096, 904])

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                         "requires the optional transformer dependency")
    def test_real_gpt2_unsorted_sites_match_execution_order(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker
        sites = (SiteTrace(1, "block"), SiteTrace(0, "mlp"), SiteTrace(1, "attn"),
                 SiteTrace(0, "block"), SiteTrace(0, "attn"), SiteTrace(1, "mlp"))
        samples = (SampleTrace("row-a", (2, 0)), SampleTrace("row-b", (1, 2)))
        spec = TraceSpec("unique-real-run", samples, sites, 0, 7, None, 12, 1024)
        identity = BrokerJobIdentity("c" * 32, 2, "team", "external")
        plan = bind_trace_plan(identity, spec, TraceLimits(2, 3, 2, 6, 12, 1024, 3),
                               batch_size=2, sequence_length=3, layer_count=2)
        layout = make_trace_job_layout(plan, batch_size=2, sequence_length=3, hidden_width=8,
                                       layer_count=2, max_slots=12, max_event_bytes=64)
        config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=8, n_layer=2,
                            n_head=2, use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            model = GPT2LMHeadModel(config).eval()
        worker = TracedGPT2Worker("tiny-gpt2", model, device="cpu", max_batch_size=2,
                                  max_sequence_length=3)
        try:
            result = worker.execute_traced(
                ForwardRequest(plan.worker_request_id,
                               {"input_ids": torch.tensor([[1, 3, 5], [2, 4, 6]])}, seed=7),
                captures=(plan.capture_request(),),
                sites=tuple(TraceSite(site.layer_index, site.module_kind) for site in sites),
                sequence_starts=plan.sequence_starts(), max_events=12, max_trace_bytes=1024)
            metadata = layout.validate(result.events)
            self.assertEqual([(event.batch_positions[0], event.layer_index, event.module_kind)
                              for event in metadata],
                             [(slot.batch_position, slot.layer_index, slot.module_kind)
                              for slot in layout.slots])
            self.assertEqual([event.sequence for event in metadata], list(range(7, 19)))
        finally:
            worker.close()

    def test_slot_and_byte_caps_are_checked_before_running(self):
        for kwargs in (dict(hidden_width=4, max_slots=3, max_event_bytes=64),
                       dict(hidden_width=4, max_slots=8, max_event_bytes=31),
                       dict(hidden_width=5, max_slots=8, max_event_bytes=64)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    make_trace_job_layout(self.plan, batch_size=2, sequence_length=3,
                                          layer_count=2, **kwargs)


if __name__ == "__main__":
    unittest.main()
