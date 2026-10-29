"""CPU-only custody and routing tests for the trace-delivery adapter."""
from dataclasses import replace
from concurrent.futures import Future
import dis
import gc
import hashlib
import inspect
import struct
import sys
import threading
import time
import types
import unittest
from unittest import mock
import weakref

import torch

from looking_glass.capture_dispatcher import (
    CaptureDispatcher, CaptureRequest, CapturedSlice, SampleSelection,
)
from looking_glass.host_budget import (
    HostBudget, HostBudgetExceeded, HostClaim, HostRequestCancelled,
)
from looking_glass.trace_store import HostTraceStore
from looking_glass.trace_transport import (
    DeliveryConstructionError, DeliveryError, TraceDelivery,
)


def event(*, sample_id="sample", row=1):
    return CapturedSlice(
        run_id="run", job_id="lg-" + "0" * 32 + "-7", request_id="request", sample_id=sample_id,
        batch_position=row, token_positions=(0,), forward_index=0,
        generation_step=None, sequence=0, module_path="block.0", layer_index=0,
        module_kind="block", output_path=(), invocation=0, source_shape=(2, 1, 2),
        source_dtype="torch.float32", source_device="cpu", saved_shape=(1, 2),
        saved_dtype="torch.float32", saved_device="cpu",
        value=torch.tensor([[1.0, 2.0]]),
    )


META = 65536


def make_delivery(*, limit=131072):
    budget = HostBudget(max_bytes=limit, max_items=12, max_requests=4, max_operations=4)
    store = HostTraceStore(max_tensor_bytes=64, max_items=4)
    delivery = TraceDelivery(
        budget, store, broker_id="0" * 32, job_id=7, team_id="team",
        run_id="run", request_id="request", sample_id="sample", batch_position=1,
        max_payload_bytes=64, metadata_reserve_bytes=META,
    )
    return budget, store, delivery


class TraceTransportTests(unittest.TestCase):
    def test_interrupted_constructor_reclaims_only_its_own_metadata_generation(self):
        """Interrupt request adoption and metadata handle assignment."""
        instructions = list(dis.get_instructions(TraceDelivery.__init__))
        metadata_stores = [op.offset for op in instructions
                           if op.opname == "STORE_ATTR" and op.argval == "_metadata_claim"]
        self.assertEqual(len(metadata_stores), 2)
        offsets = [
            next(op.offset for op in instructions
                 if op.opname == "STORE_ATTR" and op.argval == "_request"),
            metadata_stores[-1],
        ]
        for target in offsets:
            with self.subTest(offset=target):
                budget = HostBudget(max_bytes=200000, max_items=8, max_requests=3,
                                    max_operations=3)
                store = HostTraceStore(max_tensor_bytes=64, max_items=4)
                unrelated = budget.open_request("unrelated")
                unrelated_claim = unrelated.reserve("other", "data", 9)

                def interrupt(frame, action, _arg):
                    if frame.f_code is TraceDelivery.__init__.__code__:
                        if action == "call":
                            frame.f_trace_opcodes = True
                        if action == "opcode" and frame.f_lasti == target:
                            sys.settrace(None)
                            raise KeyboardInterrupt("simulated handoff interruption")
                    return interrupt

                try:
                    sys.settrace(interrupt)
                    with self.assertRaises(BaseException):
                        TraceDelivery(
                            budget, store, broker_id="0" * 32, job_id=7,
                            team_id="team", run_id="run", request_id="request",
                            sample_id="sample", batch_position=0,
                            max_payload_bytes=64, metadata_reserve_bytes=META,
                        )
                finally:
                    sys.settrace(None)
                snap = budget.snapshot()
                self.assertEqual(snap.used_bytes, 9)
                self.assertEqual([item.request_id for item in snap.requests], ["unrelated"])
                unrelated_claim.release()
                unrelated.close()

    def test_constructor_failure_preserves_reused_id_from_another_generation(self):
        budget = HostBudget(max_bytes=200000, max_items=8, max_requests=3, max_operations=3)
        store = HostTraceStore(max_tensor_bytes=64, max_items=4)
        existing = budget.open_request("delivery-" + "a" * 32)
        claim = existing.reserve("other", "data", 9)
        with mock.patch("looking_glass.trace_transport.uuid.uuid4",
                        return_value=types.SimpleNamespace(hex="a" * 32)):
            with self.assertRaises(ValueError):
                TraceDelivery(
                    budget, store, broker_id="0" * 32, job_id=7, team_id="team",
                    run_id="run", request_id="request", sample_id="sample",
                    batch_position=0, max_payload_bytes=64, metadata_reserve_bytes=META,
                )
        self.assertEqual(claim.info().nbytes, 9)
        self.assertEqual(budget.snapshot().used_bytes, 9)
        claim.release()
        existing.close()

    def test_constructor_failed_cleanup_leaves_guard_for_supervisor(self):
        budget = HostBudget(max_bytes=200000, max_items=8, max_requests=3, max_operations=3)
        store = HostTraceStore(max_tensor_bytes=64, max_items=4)
        with mock.patch.object(store, "request", side_effect=RuntimeError("store failure")):
            with mock.patch.object(HostClaim, "release", side_effect=MemoryError("cleanup")):
                with self.assertRaises(DeliveryConstructionError) as captured:
                    TraceDelivery(
                        budget, store, broker_id="0" * 32, job_id=7,
                        team_id="team", run_id="run", request_id="request",
                        sample_id="sample", batch_position=0, max_payload_bytes=64,
                        metadata_reserve_bytes=META,
                    )
        guards = budget.active_guards()
        self.assertEqual(len(guards), 1)
        self.assertIs(captured.exception.guard, guards[0])
        generation = guards[0].generation
        claims = [claim for claim in budget.active_claims()
                  if claim.info().request_generation == generation]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].info().stage, "metadata")
        claims[0].release()
        self.assertTrue(guards[0].cancel_and_close_empty())
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_constructor_tombstones_generation_before_scanning_claims(self):
        budget = HostBudget(max_bytes=200000, max_items=8, max_requests=3, max_operations=3)
        store = HostTraceStore(max_tensor_bytes=64, max_items=4)
        original = budget.active_claims
        checked = []

        def inspect_during_recovery():
            guarded_request = budget.active_guards()[0].adopt()
            with self.assertRaises(HostRequestCancelled):
                guarded_request.reserve("other", "late", 1)
            checked.append(True)
            return original()

        with mock.patch.object(store, "request", side_effect=RuntimeError("store failure")):
            with mock.patch.object(budget, "active_claims", new=inspect_during_recovery):
                with self.assertRaises(RuntimeError):
                    TraceDelivery(
                        budget, store, broker_id="0" * 32, job_id=7, team_id="team",
                        run_id="run", request_id="request", sample_id="sample",
                        batch_position=0, max_payload_bytes=64, metadata_reserve_bytes=META,
                    )
        self.assertEqual(checked, [True])
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_constructor_does_not_dispose_foreign_generation_claims(self):
        budget = HostBudget(max_bytes=200000, max_items=8, max_requests=3, max_operations=3)
        store = HostTraceStore(max_tensor_bytes=64, max_items=4)
        foreign_claims = []

        def reserve_foreign_then_fail(_request_id):
            request = budget.active_guards()[0].adopt()
            foreign_claims.append(request.reserve("external", "payload", 13))
            raise RuntimeError("store failure")

        with mock.patch.object(store, "request", new=reserve_foreign_then_fail):
            with self.assertRaises(DeliveryConstructionError) as captured:
                TraceDelivery(
                    budget, store, broker_id="0" * 32, job_id=7, team_id="team",
                    run_id="run", request_id="request", sample_id="sample",
                    batch_position=0, max_payload_bytes=64, metadata_reserve_bytes=META,
                )
        self.assertEqual(foreign_claims[0].info().nbytes, 13)
        self.assertEqual(budget.snapshot().used_bytes, META + 13)
        guard = captured.exception.guard
        for claim in budget.active_claims():
            if claim.info().request_generation == guard.generation:
                claim.release()  # Fixture has no real payload; supervisor disposes it.
        self.assertTrue(guard.cancel_and_close_empty())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_dispatcher_capture_round_trips_through_owned_delivery(self):
        """A real hook capture preserves identity, coordinates and bytes."""
        # Only the delivery source is reserved; fixture input/output and
        # dispatcher workspace are outside this adapter's ledger.
        budget = HostBudget(max_bytes=131072, max_items=12, max_requests=4, max_operations=4)
        store = HostTraceStore(max_tensor_bytes=64, max_items=4)
        delivery = TraceDelivery(
            budget, store, broker_id="0" * 32, job_id=7, team_id="team",
            run_id="run", request_id="request", sample_id="sample", batch_position=0,
            max_payload_bytes=64, metadata_reserve_bytes=META,
        )
        # This envelope is reserved before the dispatcher creates the slice.
        delivery.reserve_source(16)
        request = CaptureRequest(
            run_id="run", job_id="lg-" + "0" * 32 + "-7", request_id="request",
            samples=(SampleSelection(0, "sample", (2,)),), forward_index=0,
            generation_step=None, max_events=1, max_trace_bytes=16,
        )
        dispatcher = CaptureDispatcher(
            (request,), batch_size=1, sequence_length=3, max_events=1,
            max_trace_bytes=16, sequence_starts={"run": 4},
        )
        model = torch.nn.Identity()
        hook = model.register_forward_hook(
            lambda _module, _args, output: dispatcher.record(
                "block.0", layer_index=0, module_kind="block", output=output,
                output_path=()))
        try:
            _output, events = dispatcher.run(
                lambda: model(torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)))
        finally:
            hook.remove()
        self.assertEqual(len(events), 1)
        delivery.stage(events[0])
        manifest = delivery.publish()
        self.assertEqual(manifest.event.identity.job_id, request.job_id)
        self.assertEqual(manifest.event.batch_positions, (0,))
        self.assertEqual(manifest.event.token_positions, (2,))
        self.assertEqual(manifest.event.sequence, 4)
        self.assertEqual(manifest.event.module_path, "block.0")
        self.assertEqual(manifest.event.invocation, 0)
        digests = []
        delivery.send(lambda data: digests.append(hashlib.sha256(data).digest()))
        expected = struct.pack("<4f", 8.0, 9.0, 10.0, 11.0)
        self.assertEqual(digests, [hashlib.sha256(expected).digest()])
        delivery.ack_received(delivery.identity)
        delivery.ack_release(delivery.identity)
        events = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(store.usage().claimed_items, 0)

    def test_owned_sample_receipt_and_release_are_distinct(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        self.assertEqual(budget.snapshot().used_bytes, META + 8 + 8)
        self.assertEqual(store.usage().ready_bytes, 8)
        manifest = delivery.publish()
        self.assertEqual(manifest.event.identity.sample_id, "sample")
        self.assertEqual(manifest.event.batch_positions, (1,))
        hashes = []
        delivery.send(lambda data: hashes.append(hashlib.sha256(data).hexdigest()))
        self.assertEqual(hashes, [hashlib.sha256(bytes.fromhex("0000803f00000040")).hexdigest()])
        self.assertEqual(store.usage().claimed_bytes, 0)
        self.assertEqual(budget.snapshot().used_bytes, META + 8 + 8)
        with self.assertRaises(DeliveryError):
            delivery.ack_received(replace(delivery.identity, team_id="other"))
        delivery.ack_received(delivery.identity)
        self.assertEqual(budget.snapshot().used_bytes, META + 8 + 8)
        delivery.ack_release(delivery.identity)
        self.assertFalse(delivery.drain(0.05))
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        self.assertFalse(delivery.drain(0.01))
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_foreign_row_is_rejected_before_copy(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        foreign = event(sample_id="another", row=0)
        with mock.patch("looking_glass.trace_store._clone_tensor",
                        side_effect=AssertionError("must not clone")) as clone:
            with self.assertRaises(DeliveryError):
                delivery.stage(foreign)
            clone.assert_not_called()
        self.assertEqual(store.usage().claimed_items, 0)
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        delivery.cancel()
        foreign = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_foreign_broker_lifetime_is_rejected_before_copy(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        foreign = replace(event(), job_id="lg-" + "1" * 32 + "-7")
        with mock.patch("looking_glass.trace_store._clone_tensor") as clone:
            with self.assertRaises(DeliveryError):
                delivery.stage(foreign)
            clone.assert_not_called()
        self.assertEqual(store.usage().claimed_items, 0)
        foreign = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_store_copy_is_reserved_before_allocating(self):
        budget, store, delivery = make_delivery(limit=META + 15)
        delivery.reserve_source(8)
        source = event()
        with mock.patch("looking_glass.trace_store._clone_tensor",
                        side_effect=AssertionError("must not clone")):
            with self.assertRaises((DeliveryError, HostBudgetExceeded)):
                delivery.stage(source)
        self.assertEqual(store.usage().claimed_items, 0)
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        delivery.cancel()
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_cancel_during_copy_keeps_both_claims_until_copy_stops(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        entered, resume = threading.Event(), threading.Event()
        original = __import__("looking_glass.trace_store", fromlist=["_clone_tensor"])._clone_tensor
        errors = []

        def blocked_copy(value):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError("test copy timed out")
            return original(value)

        def stage():
            try:
                delivery.stage(event())
            except DeliveryError as error:
                errors.append(error)

        with mock.patch("looking_glass.trace_store._clone_tensor", side_effect=blocked_copy):
            thread = threading.Thread(target=stage)
            thread.start()
            self.assertTrue(entered.wait(2))
            delivery.cancel()
            self.assertFalse(delivery.drain(0.01))
            snap = budget.snapshot()
            self.assertEqual(snap.used_bytes, META + 16)
            self.assertEqual(len(snap.operations), 1)
            self.assertEqual(store.usage().claimed_bytes, 8)
            resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertFalse(delivery.drain(0.05))
        self.assertEqual(store.usage().claimed_bytes, 0)
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_concurrent_stage_rejection_cannot_finish_another_copy(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        entered, resume = threading.Event(), threading.Event()
        original = __import__("looking_glass.trace_store", fromlist=["_clone_tensor"])._clone_tensor
        errors = []

        def blocked_copy(value):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError("test copy timed out")
            return original(value)

        def first_stage():
            try:
                delivery.stage(event())
            except DeliveryError as error:
                errors.append(error)

        with mock.patch("looking_glass.trace_store._clone_tensor", side_effect=blocked_copy):
            thread = threading.Thread(target=first_stage)
            thread.start()
            self.assertTrue(entered.wait(2))
            with self.assertRaises(DeliveryError):
                delivery.stage(event())
            self.assertEqual(budget.snapshot().used_bytes, META + 16)
            self.assertEqual(len(budget.snapshot().operations), 1)
            self.assertEqual(store.usage().claimed_bytes, 8)
            resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        delivery.cancel()
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_rejects_a_small_view_with_larger_backing_storage(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = replace(event(), value=torch.arange(32.0)[:2].reshape(1, 2))
        with self.assertRaises(DeliveryError):
            delivery.stage(source)
        self.assertEqual(store.usage().claimed_items, 0)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_direct_capture_rejects_source_saved_dtype_mismatch(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = replace(event(), source_dtype="torch.float64")
        with mock.patch("looking_glass.trace_store._clone_tensor",
                        side_effect=AssertionError("must not clone")) as clone:
            with self.assertRaises(DeliveryError):
                delivery.stage(source)
            clone.assert_not_called()
        self.assertEqual(store.usage().claimed_items, 0)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_store_contention_cannot_block_cancel_or_bounded_drain(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        entered, resume = threading.Event(), threading.Event()

        def hold_store():
            with store._lock:
                entered.set()
                resume.wait(3)

        thread = threading.Thread(target=hold_store)
        thread.start()
        self.assertTrue(entered.wait(1))
        cancel_thread = threading.Thread(target=delivery.cancel)
        cancel_thread.start()
        try:
            cancel_thread.join(0.2)
            self.assertFalse(cancel_thread.is_alive(), "cancel blocked on store contention")
            self.assertFalse(delivery.drain(0.03))
            self.assertEqual(budget.snapshot().used_bytes, META + 16)
        finally:
            resume.set()
            thread.join(3)
            cancel_thread.join(3)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_metadata_envelope_must_cover_bounded_schema(self):
        budget = HostBudget(max_bytes=100000, max_items=4, max_requests=2, max_operations=2)
        store = HostTraceStore(max_tensor_bytes=64, max_items=2)
        with self.assertRaises(ValueError):
            TraceDelivery(
                budget, store, broker_id="0" * 32, job_id=7, team_id="team",
                run_id="run", request_id="request", sample_id="sample",
                batch_position=1, max_payload_bytes=64, metadata_reserve_bytes=1,
            )
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_cancel_while_store_reservation_is_blocked(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        held, resume = threading.Event(), threading.Event()
        errors = []

        def hold_store():
            with store._lock:
                held.set()
                resume.wait(3)

        def stage():
            try:
                delivery.stage(event())
            except DeliveryError as exc:
                errors.append(exc)

        holder = threading.Thread(target=hold_store)
        holder.start()
        self.assertTrue(held.wait(1))
        worker = threading.Thread(target=stage)
        worker.start()
        try:
            deadline = time.monotonic() + 1
            while delivery.state != "staging" and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(delivery.state, "staging")
            delivery.cancel()
            self.assertFalse(delivery.drain(0.03))
            self.assertEqual(len(budget.snapshot().operations), 1)
            self.assertEqual(budget.snapshot().used_bytes, META + 16)
        finally:
            resume.set()
            holder.join(3)
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors)
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(store.usage().claimed_items, 0)

    def test_cancel_while_store_handoff_is_blocked(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()
        held, resume = threading.Event(), threading.Event()
        errors, received = [], []

        def hold_store():
            with store._lock:
                held.set()
                resume.wait(3)

        def send():
            try:
                delivery.send(lambda data: received.append(len(data)))
            except DeliveryError as exc:
                errors.append(exc)

        holder = threading.Thread(target=hold_store)
        holder.start()
        self.assertTrue(held.wait(1))
        worker = threading.Thread(target=send)
        worker.start()
        try:
            deadline = time.monotonic() + 1
            while delivery.state != "sending" and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(delivery.state, "sending")
            delivery.cancel()
            self.assertFalse(delivery.drain(0.03))
            self.assertEqual(len(budget.snapshot().operations), 1)
            self.assertEqual(budget.snapshot().used_bytes, META + 24)
        finally:
            resume.set()
            holder.join(3)
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors)
        self.assertEqual(received, [])
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(store.usage().claimed_items, 0)

    def test_cleanup_finalizer_does_not_hold_the_lifecycle_lock(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        entered, resume = threading.Event(), threading.Event()
        weakrefs = []
        original = __import__("looking_glass.trace_store", fromlist=["_clone_tensor"])._clone_tensor

        def tracked(value):
            copy = original(value)
            weakrefs.append(weakref.ref(copy, lambda _: (entered.set(), resume.wait(3))))
            return copy

        with mock.patch("looking_glass.trace_store._clone_tensor", new=tracked):
            delivery.stage(source)
        delivery.cancel()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(delivery.drain(0.03))
            self.assertEqual(budget.snapshot().used_bytes, META + 16)
        finally:
            resume.set()
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_serializer_oom_does_not_leave_an_uncharged_tensor_in_traceback(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        refs = []
        original = __import__("looking_glass.trace_store", fromlist=["_clone_tensor"])._clone_tensor

        def tracked(value):
            copy = original(value)
            refs.append(weakref.ref(copy))
            return copy

        def oom(_payload):
            raise MemoryError("simulated encoder allocation failure")

        with mock.patch("looking_glass.trace_store._clone_tensor", new=tracked):
            delivery.stage(source)
        delivery.publish()
        with mock.patch("looking_glass.trace_transport._encode_payload", new=oom):
            with self.assertRaises(DeliveryError) as captured:
                delivery.send(lambda _data: None)
        self.assertFalse(delivery.drain(0.05))
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertIsNone(captured.exception.__cause__)
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_deferred_sink_quarantines_all_inflight_claims(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()
        with self.assertRaises(DeliveryError):
            delivery.send(lambda _data: Future())
        self.assertEqual(delivery.state, "quarantined")
        self.assertFalse(delivery.drain(0.01))
        self.assertEqual(budget.snapshot().used_bytes, META + 24)
        self.assertEqual(len(budget.snapshot().operations), 1)
        source = None
        delivery.confirm_source_disposed()
        self.assertFalse(delivery.close(0.01))

    def test_drain_rejects_invalid_and_oversized_timeouts(self):
        _, _, delivery = make_delivery()
        for timeout in (float("nan"), float("inf"), -1, 10**500, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                delivery.drain(timeout)
        self.assertTrue(delivery.close(1))

    def test_drain_is_bounded_when_ledger_finalization_blocks(self):
        _, _, delivery = make_delivery()
        entered, resume = threading.Event(), threading.Event()
        original = delivery._metadata_claim.release
        results = []

        def blocked_release():
            entered.set()
            resume.wait(3)
            return original()

        with mock.patch.object(delivery._metadata_claim, "release", new=blocked_release):
            delivery.cancel()
            thread = threading.Thread(target=lambda: results.append(delivery.drain(0.03)))
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                thread.join(0.2)
                self.assertFalse(thread.is_alive(), "drain blocked in ledger release")
                self.assertEqual(results, [False])
            finally:
                resume.set()
                thread.join(3)
            self.assertTrue(delivery.drain(1))

    def test_drain_timeout_includes_lifecycle_lock_acquisition(self):
        _, _, delivery = make_delivery()
        entered, resume = threading.Event(), threading.Event()

        def holder():
            with delivery._condition:
                entered.set()
                resume.wait(3)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(entered.wait(1))
        try:
            started = time.monotonic()
            self.assertFalse(delivery.drain(0.03))
            self.assertLess(time.monotonic() - started, 0.3)
        finally:
            resume.set()
            thread.join(3)
        self.assertTrue(delivery.close(1))

    def test_each_ack_identity_component_is_bound_to_the_delivery(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()
        delivery.send(lambda _data: None)
        ident = delivery.identity
        alternatives = {
            "broker_id": "1" * 32, "job_id": 8, "team_id": "foreign",
            "run_id": "foreign", "request_id": "foreign", "sample_id": "foreign",
            "batch_position": 0, "delivery_id": "f" * 32,
        }
        for field, value in alternatives.items():
            with self.subTest(field=field), self.assertRaises(DeliveryError):
                delivery.ack_received(replace(ident, **{field: value}))
        for forged in (replace(ident, job_id=7.0), replace(ident, batch_position=True)):
            with self.assertRaises(DeliveryError):
                delivery.ack_received(forged)
        self.assertEqual(delivery.state, "sent")
        delivery.ack_received(ident)
        self.assertTrue(delivery.ack_release(ident))
        self.assertFalse(delivery.ack_release(ident))
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_failed_ledger_cancellation_quarantines_instead_of_claiming_cleanup(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        with mock.patch.object(delivery._request, "cancel",
                               side_effect=RuntimeError("ledger failure")):
            delivery.cancel()
            deadline = time.monotonic() + 1
            while delivery.state != "quarantined" and time.monotonic() < deadline:
                time.sleep(0.001)
        self.assertEqual(delivery.state, "quarantined")
        self.assertFalse(delivery.drain(0.01))
        self.assertEqual(budget.snapshot().used_bytes, META + 16)
        self.assertEqual(store.usage().claimed_bytes, 8)
        source = None
        delivery.confirm_source_disposed()
        self.assertFalse(delivery.drain(0.01))

    def test_cancel_does_not_wait_for_the_shared_ledger(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        held, resume = threading.Event(), threading.Event()

        def holder():
            with budget._lock:
                held.set()
                resume.wait(3)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(held.wait(1))
        cancellation = threading.Thread(target=delivery.cancel)
        cancellation.start()
        try:
            cancellation.join(0.2)
            self.assertFalse(cancellation.is_alive(), "cancellation waited on ledger")
            with self.assertRaises(DeliveryError):
                delivery.publish()
            self.assertFalse(delivery.drain(0.03))
        finally:
            resume.set()
            thread.join(3)
            cancellation.join(3)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_cancel_signal_does_not_wait_for_inflight_ledger_finish(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        copy_entered, copy_resume = threading.Event(), threading.Event()
        ledger_held, ledger_resume, finish_entered = (threading.Event() for _ in range(3))
        original_clone = __import__("looking_glass.trace_store", fromlist=["_clone_tensor"])._clone_tensor
        errors = []

        def blocked_copy(value):
            copy_entered.set()
            copy_resume.wait(3)
            return original_clone(value)

        def stage():
            try:
                delivery.stage(event())
            except DeliveryError as exc:
                errors.append(exc)

        with mock.patch("looking_glass.trace_store._clone_tensor", new=blocked_copy):
            worker = threading.Thread(target=stage)
            worker.start()
            self.assertTrue(copy_entered.wait(1))
            operation = delivery._operation
            original_finish = operation.finish

            def announced_finish():
                finish_entered.set()
                return original_finish()

            def hold_ledger():
                with budget._lock:
                    ledger_held.set()
                    ledger_resume.wait(3)

            holder = threading.Thread(target=hold_ledger)
            holder.start()
            self.assertTrue(ledger_held.wait(1))
            cancellation = None
            try:
                with mock.patch.object(operation, "finish", new=announced_finish):
                    copy_resume.set()
                    self.assertTrue(finish_entered.wait(1))
                    cancellation = threading.Thread(target=delivery.cancel)
                    cancellation.start()
                    cancellation.join(0.2)
                    self.assertFalse(cancellation.is_alive(), "cancel waited on a finishing operation")
                    self.assertFalse(delivery.drain(0.03))
            finally:
                ledger_resume.set()
                holder.join(3)
                worker.join(3)
                if cancellation is not None:
                    cancellation.join(3)
        self.assertTrue(errors)
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_cancel_thread_creation_failure_dominates_concurrent_state(self):
        budget, _, delivery = make_delivery()
        reserved, resume = threading.Event(), threading.Event()
        errors = []
        original = delivery._request.reserve

        def blocked_reserve(owner, stage, nbytes, **kwargs):
            claim = original(owner, stage, nbytes, **kwargs)
            if stage == "source":
                reserved.set()
                resume.wait(3)
            return claim

        def reserve():
            try:
                delivery.reserve_source(8)
            except DeliveryError as exc:
                errors.append(exc)

        with mock.patch.object(delivery._request, "reserve", new=blocked_reserve):
            worker = threading.Thread(target=reserve)
            worker.start()
            self.assertTrue(reserved.wait(1))
            try:
                with mock.patch("threading.Thread.start", side_effect=MemoryError("no thread")):
                    delivery.cancel()
            finally:
                resume.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors)
        self.assertEqual(delivery.state, "quarantined")
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        self.assertFalse(delivery.drain(0.01))

    def test_interrupted_take_handoff_is_quarantined(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()
        lines, first = inspect.getsourcelines(TraceDelivery._send_impl)
        target = next(first + i for i, line in enumerate(lines)
                      if "self._entry = None" in line)

        def interrupt(frame, action, _arg):
            if (frame.f_code is TraceDelivery._send_impl.__code__ and action == "line"
                    and frame.f_lineno == target):
                sys.settrace(None)
                raise KeyboardInterrupt("simulated interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliveryError):
                delivery.send(lambda _data: None)
        finally:
            sys.settrace(None)
        self.assertEqual(delivery.state, "quarantined")
        self.assertEqual(len(budget.snapshot().operations), 1)
        self.assertEqual(budget.snapshot().used_bytes, META + 24)
        source = None
        delivery.confirm_source_disposed()
        self.assertFalse(delivery.drain(0.01))

    def test_interrupted_source_reservation_cannot_be_retried(self):
        budget, _, delivery = make_delivery()
        lines, first = inspect.getsourcelines(TraceDelivery.reserve_source)
        target = next(first + i for i, line in enumerate(lines)
                      if 'self._state = "reserved"' in line)

        def interrupt(frame, action, _arg):
            if (frame.f_code is TraceDelivery.reserve_source.__code__
                    and action == "line" and frame.f_lineno == target):
                sys.settrace(None)
                raise KeyboardInterrupt("simulated interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliveryError):
                delivery.reserve_source(8)
        finally:
            sys.settrace(None)
        self.assertEqual(delivery.state, "quarantined")
        with self.assertRaises(DeliveryError):
            delivery.reserve_source(8)
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        self.assertFalse(delivery.drain(0.01))

    def test_interrupted_stage_after_copy_keeps_claims_quarantined(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        lines, first = inspect.getsourcelines(TraceDelivery._stage_impl)
        target = next(first + i + 1 for i, line in enumerate(lines)
                      if "This frame must not hold" in line)

        def interrupt(frame, action, _arg):
            if (frame.f_code is TraceDelivery._stage_impl.__code__
                    and action == "line" and frame.f_lineno == target):
                sys.settrace(None)
                raise KeyboardInterrupt("simulated interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliveryError):
                delivery.stage(source)
        finally:
            sys.settrace(None)
        self.assertEqual(delivery.state, "quarantined")
        self.assertEqual(len(budget.snapshot().operations), 1)
        self.assertEqual(budget.snapshot().used_bytes, META + 16)
        source = None
        delivery.confirm_source_disposed()
        self.assertFalse(delivery.drain(0.01))

    def test_interrupted_publication_is_quarantined(self):
        _, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        lines, first = inspect.getsourcelines(TraceDelivery.publish)
        target = next(first + i for i, line in enumerate(lines)
                      if "return manifest" in line)

        def interrupt(frame, action, _arg):
            if (frame.f_code is TraceDelivery.publish.__code__ and action == "line"
                    and frame.f_lineno == target):
                sys.settrace(None)
                raise KeyboardInterrupt("simulated interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliveryError):
                delivery.publish()
        finally:
            sys.settrace(None)
        self.assertEqual(delivery.state, "quarantined")
        with self.assertRaises(DeliveryError):
            delivery.send(lambda _data: None)
        source = None
        delivery.confirm_source_disposed()
        self.assertFalse(delivery.drain(0.01))

    def test_cancel_ordering_around_publication_commit(self):
        for before_check in (True, False):
            with self.subTest(before_check=before_check):
                budget, _, delivery = make_delivery()
                delivery.reserve_source(8)
                source = event()
                delivery.stage(source)
                lines, first = inspect.getsourcelines(TraceDelivery.publish)
                needle = ("if self._is_cancelled():" if before_check
                          else 'self._state = "ready"')
                target = next(first + i for i, line in enumerate(lines) if needle in line)

                def cancel_at_boundary(frame, action, _arg):
                    if (frame.f_code is TraceDelivery.publish.__code__
                            and action == "line" and frame.f_lineno == target):
                        sys.settrace(None)
                        delivery.cancel()
                    return cancel_at_boundary

                try:
                    sys.settrace(cancel_at_boundary)
                    if before_check:
                        with self.assertRaises(DeliveryError):
                            delivery.publish()
                    else:
                        self.assertEqual(delivery.publish().identity, delivery.identity)
                finally:
                    sys.settrace(None)
                with self.assertRaises(DeliveryError):
                    delivery.send(lambda _data: None)
                source = None
                delivery.confirm_source_disposed()
                self.assertTrue(delivery.drain(1))
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_ordering_around_sink_commit(self):
        for before_check in (True, False):
            with self.subTest(before_check=before_check):
                budget, _, delivery = make_delivery()
                delivery.reserve_source(8)
                source = event()
                delivery.stage(source)
                delivery.publish()
                lines, first = inspect.getsourcelines(TraceDelivery._send_impl)
                checks = [first + i for i, line in enumerate(lines)
                          if "if self._is_cancelled():" in line]
                target = (checks[-1] if before_check else
                          next(first + i for i, line in enumerate(lines)
                               if 'self._state = "delivering"' in line))
                received = []

                def cancel_at_boundary(frame, action, _arg):
                    if (frame.f_code is TraceDelivery._send_impl.__code__
                            and action == "line" and frame.f_lineno == target):
                        sys.settrace(None)
                        delivery.cancel()
                    return cancel_at_boundary

                try:
                    sys.settrace(cancel_at_boundary)
                    with self.assertRaises(DeliveryError):
                        delivery.send(lambda data: received.append(len(data)))
                finally:
                    sys.settrace(None)
                self.assertEqual(received, [] if before_check else [8])
                with self.assertRaises(DeliveryError):
                    delivery.ack_received(delivery.identity)
                source = None
                delivery.confirm_source_disposed()
                self.assertTrue(delivery.drain(1))
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_failed_source_release_quarantines_without_publishing(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        with mock.patch.object(delivery._source_claim, "release",
                               side_effect=RuntimeError("ledger failure")):
            source = None
            delivery.confirm_source_disposed()
        self.assertEqual(delivery.state, "quarantined")
        with self.assertRaises(DeliveryError):
            delivery.publish()
        self.assertFalse(delivery.drain(0.01))
        self.assertEqual(budget.snapshot().used_bytes, META + 16)

    def test_close_is_idempotent_and_closed_stage_is_side_effect_free(self):
        budget, _, delivery = make_delivery()
        self.assertTrue(delivery.close(1))
        self.assertTrue(delivery.close(1))
        with self.assertRaises(DeliveryError):
            delivery.stage(event())
        self.assertEqual(len(budget.snapshot().requests), 0)

    def test_cancel_during_send_waits_for_sink_to_return(self):
        budget, store, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()
        entered, resume = threading.Event(), threading.Event()
        errors = []

        def sink(data):
            entered.set()
            if not resume.wait(3):
                raise RuntimeError("test sink timed out")

        def send():
            try:
                delivery.send(sink)
            except DeliveryError as error:
                errors.append(error)

        thread = threading.Thread(target=send)
        thread.start()
        self.assertTrue(entered.wait(2))
        delivery.cancel()
        self.assertFalse(delivery.drain(0.01))
        snap = budget.snapshot()
        self.assertEqual(snap.used_bytes, META + 24)
        self.assertEqual(len(snap.operations), 1)
        resume.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertFalse(delivery.drain(0.05))
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))

    def test_partial_send_failure_cannot_be_acknowledged(self):
        budget, _, delivery = make_delivery()
        delivery.reserve_source(8)
        source = event()
        delivery.stage(source)
        delivery.publish()

        def broken(_data):
            raise OSError("partial write")

        with self.assertRaises(DeliveryError) as captured:
            delivery.send(broken)
        self.assertIsNone(captured.exception.__cause__)
        with self.assertRaises(DeliveryError):
            delivery.ack_received(delivery.identity)
        self.assertFalse(delivery.drain(0.05))
        self.assertEqual(budget.snapshot().used_bytes, META + 8)
        source = None
        delivery.confirm_source_disposed()
        self.assertTrue(delivery.drain(1))


if __name__ == "__main__":
    unittest.main()
