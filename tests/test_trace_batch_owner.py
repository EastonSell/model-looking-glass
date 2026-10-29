"""Trusted CPU-only batch custody; no broker, IPC, or GPU claims."""
import gc
from dataclasses import replace
import dis
import importlib
import sys
import threading
import time
import unittest
from unittest.mock import patch
import weakref

import torch

from looking_glass.capture_dispatcher import CaptureDispatcher
from looking_glass.host_budget import HostBudget
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import (
    BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec, bind_trace_plan,
)


def fixture_plan(job_id=7):
    identity = BrokerJobIdentity("a" * 32, job_id, "team", "external")
    samples = (SampleTrace("sample-a", (0, 2)), SampleTrace("sample-b", (1, 2)))
    sites = (SiteTrace(1, "mlp"), SiteTrace(0, "attn"))
    spec = TraceSpec("run", samples, sites, 4, 12, None, 4, 128)
    limits = TraceLimits(2, 3, 2, 2, 4, 128, 3)
    return bind_trace_plan(identity, spec, limits, batch_size=2,
                           sequence_length=3, layer_count=2)


def capture(plan):
    dispatcher = CaptureDispatcher((plan.capture_request(),), batch_size=2,
                                   sequence_length=3, max_events=4,
                                   max_trace_bytes=128,
                                   sequence_starts=plan.sequence_starts())
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    def forward():
        dispatcher.record("transformer.h.0.attn", layer_index=0, module_kind="attn",
                          output_path=(0,), output=source)
        dispatcher.record("transformer.h.1.mlp", layer_index=1, module_kind="mlp",
                          output_path=(), output=source + 1)
    _, events = dispatcher.run(forward)
    return events


class TraceBatchOwnerTests(unittest.TestCase):
    def test_rejected_overlapping_stage_does_not_cancel_the_active_stage(self):
        from looking_glass import trace_store
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        entered, resume = threading.Event(), threading.Event()
        errors = []
        original_clone = trace_store._clone_tensor
        def blocked_clone(value):
            if not entered.is_set():
                entered.set()
                if not resume.wait(3):
                    raise RuntimeError("timed out waiting for the test")
            return original_clone(value)
        def first_stage(values):
            try:
                job.stage(values)
            except BaseException as error:
                errors.append(error)
        with patch.object(trace_store, "_clone_tensor", new=blocked_clone):
            thread = threading.Thread(target=first_stage, args=(events,))
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                with self.assertRaises(BatchStateError):
                    job.stage(events)
            finally:
                resume.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        events = None
        job.confirm_sources_disposed()
        job.seal()
        self.assertEqual(len(job.manifest().events), 4)
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_lost_prepare_and_seal_returns_are_recoverable_by_exact_key(self):
        from looking_glass.trace_batch_owner import TraceBatchRegistry
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing Python tracing session")
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        def interrupt_return(method):
            code = method.__func__.__code__
            offset = next(item.offset for item in dis.get_instructions(code)
                          if item.opname == "RETURN_VALUE")
            def trace(frame, event, arg):
                if event == "call":
                    frame.f_trace_opcodes = True
                if frame.f_code is code and event == "opcode" and frame.f_lasti == offset:
                    sys.settrace(None)
                    raise KeyboardInterrupt("lost successful return")
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    method()
            finally:
                sys.settrace(None)
        interrupt_return(job.prepare)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        self.assertEqual(job.state, "prepared")
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        interrupt_return(job.seal)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        self.assertEqual(job.state, "sealed")
        self.assertEqual([item.event.sequence for item in job.manifest().events],
                         [12, 13, 14, 15])
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_boolean_child_row_cannot_impersonate_integer_row_zero(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        child = job._deliveries.lookup(*job._keys[0]).adopt()
        object.__setattr__(child.identity, "batch_position", False)
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        with self.assertRaises(BatchStateError):
            job.seal()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_metadata_equality_spoof_cannot_seal_a_wrong_event(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_transport import TraceDelivery
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        class Spoof:
            def __init__(self, event):
                self.event = event
            def __eq__(self, other):
                return True
            def to_json(self):
                return replace(self.event, sequence=self.event.sequence + 1000).to_json()
        original_publish = TraceDelivery.publish
        def spoofed_metadata(child):
            manifest = original_publish(child)
            return replace(manifest, event=Spoof(manifest.event))
        with patch.object(TraceDelivery, "publish", new=spoofed_metadata):
            with self.assertRaises(BatchStateError):
                job.seal()
        with self.assertRaises(BatchStateError):
            job.manifest()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_uncertain_source_reservation_quarantines_until_supervised_reconciliation(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_transport import TraceDelivery
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        original_reserve = TraceDelivery.reserve_source
        def interrupted_reserve(child, nbytes):
            original_reserve(child, nbytes)
            raise KeyboardInterrupt("after source admission")
        with patch.object(TraceDelivery, "reserve_source", new=interrupted_reserve):
            with self.assertRaises(KeyboardInterrupt):
                job.prepare()
        with self.assertRaises(BatchStateError):
            job.confirm_sources_disposed()
        self.assertEqual(job.state, "quarantined")
        self.assertFalse(job.dispose(timeout=0))
        self.assertGreater(budget.snapshot().used_bytes, 0)
        self.assertEqual(registry.active_jobs(), (7,))
        # Test-only supervised reconciliation: no forward was started and the
        # exact child is known. Production has no automatic quarantine release.
        slot = job._deliveries.lookup(*job._keys[0])
        child = slot.owned_delivery()
        child.cancel()
        child.confirm_source_disposed()
        self.assertTrue(child.drain(5))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_inflight_registration_consumes_capacity_before_layout_reconstruction(self):
        from looking_glass import trace_batch_owner
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = trace_batch_owner.TraceBatchRegistry(
            budget=budget, store=store, broker_id="a" * 32,
            process_generation="b" * 32, max_jobs=1, max_slots_per_job=8)
        entered, resume = threading.Event(), threading.Event()
        failures = []
        original_layout = trace_batch_owner.make_trace_job_layout
        def blocked_layout(*args, **kwargs):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError("timed out waiting for the test")
            return original_layout(*args, **kwargs)
        def first_registration():
            try:
                registry.register(fixture_plan(), registration_id="c" * 32, generation=1,
                                  batch_size=2, sequence_length=3, hidden_width=4,
                                  layer_count=2, max_event_bytes=64)
            except BaseException as error:
                failures.append(error)
        with patch.object(trace_batch_owner, "make_trace_job_layout", new=blocked_layout):
            thread = threading.Thread(target=first_registration)
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                with self.assertRaises(trace_batch_owner.BatchStateError):
                    registry.register(fixture_plan(8), registration_id="d" * 32,
                                      generation=2, batch_size=2, sequence_length=3,
                                      hidden_width=4, layer_count=2, max_event_bytes=64)
                self.assertEqual(registry.active_jobs(), (7,))
                self.assertEqual(budget.snapshot().used_bytes, 0)
            finally:
                resume.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.cancel()
        job.confirm_sources_disposed()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)

    def test_store_copy_capacity_failure_keeps_sources_charged_until_disposal(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_transport import DeliveryError
        budget = HostBudget(max_bytes=4 * 65536 + 4 * 32, max_items=64,
                            max_requests=16, max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        with self.assertRaises(DeliveryError):
            job.stage(events)
        self.assertEqual(budget.snapshot().used_bytes, 4 * 65536 + 4 * 32)
        with self.assertRaises(BatchStateError):
            job.manifest()
        events = None
        job.confirm_sources_disposed()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_authorized_manifest_read_may_finish_after_concurrent_cancel(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_schema import TraceEvent
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        job.seal()
        entered, resume = threading.Event(), threading.Event()
        result, failures = [], []
        original_from_json = TraceEvent.from_json
        def blocked_copy(value):
            if not entered.is_set():
                entered.set()
                if not resume.wait(3):
                    raise RuntimeError("timed out waiting for the test")
            return original_from_json(value)
        def read():
            try:
                result.append(job.manifest())
            except BaseException as error:
                failures.append(error)
        with patch.object(TraceEvent, "from_json", new=blocked_copy):
            reader = threading.Thread(target=read)
            reader.start()
            self.assertTrue(entered.wait(3))
            job.cancel()
            resume.set()
            reader.join(3)
        self.assertFalse(reader.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].events), 4)
        with self.assertRaises(BatchStateError):
            job.manifest()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_distinct_children_with_duplicate_real_ids_prevent_batch_seal(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        first = job._deliveries.lookup(*job._keys[0]).adopt()
        second = job._deliveries.lookup(*job._keys[1]).adopt()
        object.__setattr__(second.identity, "delivery_id", first.identity.delivery_id)
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        with self.assertRaises(BatchStateError):
            job.seal()
        with self.assertRaises(BatchStateError):
            job.manifest()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_forged_or_duplicate_child_delivery_ids_prevent_batch_seal(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_transport import TraceDelivery
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        original_publish = TraceDelivery.publish
        def forged_identity(child):
            manifest = original_publish(child)
            return replace(manifest, identity=replace(manifest.identity, delivery_id="f" * 32))
        with patch.object(TraceDelivery, "publish", new=forged_identity):
            with self.assertRaises(BatchStateError):
                job.seal()
        with self.assertRaises(BatchStateError):
            job.manifest()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_same_row_site_swaps_are_rejected_before_batch_seal(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_transport import TraceDelivery
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        misroute = {12: events[2], 14: events[0]}
        real_stage = TraceDelivery.stage
        def wrong_child(child, event):
            return real_stage(child, misroute.get(event.sequence, event))
        with patch.object(TraceDelivery, "stage", new=wrong_child):
            job.stage(events)
        events = None
        misroute.clear()
        job.confirm_sources_disposed()
        with self.assertRaises(BatchStateError):
            job.seal()
        with self.assertRaises(BatchStateError):
            job.manifest()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_during_copy_keeps_claims_until_stage_and_disposal_finish(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass import trace_store
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        entered, resume = threading.Event(), threading.Event()
        failures = []
        original_clone = trace_store._clone_tensor
        def blocked_clone(value):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError("timed out waiting for the test")
            return original_clone(value)
        def run(values):
            try:
                job.stage(values)
            except BaseException as error:
                failures.append(error)
        with patch.object(trace_store, "_clone_tensor", new=blocked_clone):
            thread = threading.Thread(target=run, args=(events,))
            thread.start()
            self.assertTrue(entered.wait(3))
            job.cancel()
            with self.assertRaises(BatchStateError):
                job.confirm_sources_disposed()
            self.assertFalse(job.dispose(timeout=0))
            snapshot = budget.snapshot()
            self.assertGreater(snapshot.used_bytes, 0)
            self.assertTrue(any(claim.stage == "source" and claim.active_operations == 1
                                for claim in snapshot.claims))
            self.assertTrue(any(claim.stage == "store-copy" and claim.active_operations == 1
                                for claim in snapshot.claims))
            self.assertEqual(len(snapshot.operations), 1)
            resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        events = None
        failures.clear()
        job.confirm_sources_disposed()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_stage_acquisition_does_not_strand_owner(self):
        from looking_glass.trace_batch_owner import TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        original_begin = job._begin
        def interrupted_begin(*args):
            original_begin(*args)
            raise KeyboardInterrupt("after start ownership")
        with patch.object(job, "_begin", new=interrupted_begin):
            with self.assertRaises(KeyboardInterrupt):
                job.stage(events)
        events = None
        self.assertEqual(job.state, "failed")
        job.confirm_sources_disposed()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_reused_job_key_needs_fresh_generation_and_old_lookup_is_stale(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        args = dict(registration_id="c" * 32, generation=1, batch_size=2,
                    sequence_length=3, hidden_width=4, layer_count=2, max_event_bytes=64)
        registry.register(fixture_plan(), **args)
        first = registry.lookup(7, registration_id="c" * 32, generation=1)
        first.cancel()
        first.confirm_sources_disposed()
        self.assertTrue(first.dispose(timeout=5))
        registry.retire(first)
        with self.assertRaises(BatchStateError):
            registry.register(fixture_plan(), **args)
        args["generation"] = 2
        registry.register(fixture_plan(), **args)
        with self.assertRaises(BatchStateError):
            registry.lookup(7, registration_id="c" * 32, generation=1)
        second = registry.lookup(7, registration_id="c" * 32, generation=2)
        second.cancel()
        second.confirm_sources_disposed()
        self.assertTrue(second.dispose(timeout=5))
        registry.retire(second)

    def test_cancel_before_capture_allows_explicit_no_source_disposal(self):
        from looking_glass.trace_batch_owner import TraceBatchRegistry
        for prepare in (False, True):
            with self.subTest(prepared=prepare):
                budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                                    max_operations=16)
                store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
                registry = TraceBatchRegistry(budget=budget, store=store,
                                              broker_id="a" * 32, process_generation="b" * 32,
                                              max_jobs=1, max_slots_per_job=8)
                registry.register(fixture_plan(), registration_id="c" * 32, generation=1, batch_size=2,
                                  sequence_length=3, hidden_width=4, layer_count=2,
                                  max_event_bytes=64)
                job = registry.lookup(7, registration_id="c" * 32, generation=1)
                if prepare:
                    job.prepare()
                job.cancel()
                job.confirm_sources_disposed()
                self.assertTrue(job.dispose(timeout=5))
                registry.retire(job)
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_cannot_return_between_final_seal_check_and_commit(self):
        from looking_glass.trace_batch_owner import TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        job.stage(events)
        events = None
        job.confirm_sources_disposed()
        checked, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
        original_is_set = job._cancel.is_set
        count = 0
        def blocked_check():
            nonlocal count
            value = original_is_set()
            count += 1
            if count == 6:
                checked.set()
                self.assertTrue(resume.wait(3))
            return value
        failures = []
        def seal():
            try:
                job.seal()
            except BaseException as error:
                failures.append(error)
        def cancel():
            job.cancel()
            cancelled.set()
        with patch.object(job._cancel, "is_set", new=blocked_check):
            seal_thread = threading.Thread(target=seal)
            seal_thread.start()
            self.assertTrue(checked.wait(3))
            cancel_thread = threading.Thread(target=cancel)
            cancel_thread.start()
            self.assertFalse(cancelled.wait(0.05))
            resume.set()
            seal_thread.join(3)
            cancel_thread.join(3)
        self.assertFalse(seal_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(cancelled.is_set())
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)

    def test_failed_source_admission_drains_metadata_without_phantom_source(self):
        from looking_glass.trace_batch_owner import TraceBatchRegistry
        from looking_glass.host_budget import HostBudgetExceeded
        from looking_glass.trace_transport import TraceDelivery
        budget = HostBudget(max_bytes=65536, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        with self.assertRaises(HostBudgetExceeded):
            job.prepare()
        original_cancel = TraceDelivery.cancel
        def cancel_after_transition(child):
            result = original_cancel(child)
            deadline = time.monotonic() + 2
            while child.state == "new" and time.monotonic() < deadline:
                time.sleep(0.001)
            return result
        with patch.object(TraceDelivery, "cancel", new=cancel_after_transition):
            job.confirm_sources_disposed()
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_rejected_stage_does_not_retain_source_via_public_error(self):
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id="a" * 32,
                                      process_generation="b" * 32, max_jobs=1,
                                      max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        events = capture(plan)
        refs = tuple(weakref.ref(event.value) for event in events)
        retained = None
        try:
            job.stage(events)
        except BatchStateError as error:
            retained = error
        events = None
        gc.collect()
        self.assertIsNotNone(retained)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_complete_batch_is_private_until_sealed_and_drains_after_disposal(self):
        try:
            module = importlib.import_module("looking_glass.trace_batch_owner")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "batch custody owner is not implemented")
        budget = HostBudget(max_bytes=1_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=1024, max_items=8)
        registry = module.TraceBatchRegistry(budget=budget, store=store,
                                            broker_id="a" * 32,
                                            process_generation="b" * 32,
                                            max_jobs=2, max_slots_per_job=8)
        plan = fixture_plan()
        registry.register(plan, registration_id="c" * 32, generation=1, batch_size=2,
                          sequence_length=3, hidden_width=4, layer_count=2,
                          max_event_bytes=64)
        job = registry.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        events = capture(plan)
        refs = tuple(weakref.ref(event.value) for event in events)
        job.stage(tuple(reversed(events)))
        with self.assertRaises(module.BatchStateError):
            job.manifest()
        events = None
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        job.confirm_sources_disposed()
        job.seal()
        manifest = job.manifest()
        self.assertEqual([(item.event.sequence, item.event.batch_positions,
                           item.event.layer_index, item.event.module_kind)
                          for item in manifest.events],
                         [(12, (0,), 0, "attn"), (13, (1,), 0, "attn"),
                          (14, (0,), 1, "mlp"), (15, (1,), 1, "mlp")])
        self.assertTrue(job.dispose(timeout=5))
        registry.retire(job)
        self.assertEqual(registry.active_jobs(), ())
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(len(budget.snapshot().requests), 0)


if __name__ == "__main__":
    unittest.main()
