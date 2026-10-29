"""CPU-only tests for bounded, recoverable TraceDelivery adoption."""
import dis
import sys
import threading
import time
import tracemalloc
import types
import unittest
from unittest import mock

import torch

from looking_glass.capture_dispatcher import CapturedSlice
from looking_glass.host_budget import HostBudget, HostClaim
from looking_glass.trace_store import HostTraceStore
from looking_glass.trace_transport import TraceDelivery
from looking_glass.trace_delivery_guard import (
    DeliverySlotError, DeliverySlotFull, DeliverySlotRegistry,
)


def resources():
    return (HostBudget(max_bytes=200000, max_items=8, max_requests=4, max_operations=4),
            HostTraceStore(max_tensor_bytes=64, max_items=4))


def options():
    return dict(broker_id="0" * 32, job_id=7, team_id="team", run_id="run",
                request_id="request", sample_id="sample", batch_position=0,
                max_payload_bytes=64, metadata_reserve_bytes=65536)


class DeliverySlotTests(unittest.TestCase):
    def test_registry_limits_identities_and_rejects_forged_capabilities(self):
        for invalid in (0, True, 65537, 1.0):
            with self.assertRaises(ValueError):
                DeliverySlotRegistry(max_slots=invalid)
        registry = DeliverySlotRegistry(max_slots=2)
        for slot_id, generation in (("", 1), ("x" * 129, 1), ("ok", True),
                                    ("ok", 0), ("ok", 2**63)):
            with self.assertRaises(ValueError):
                registry.register(slot_id, generation)
        registry.register("valid", 1)
        legitimate = registry.lookup("valid", 1)
        forged = type(legitimate)(registry, "valid", 1)
        for action in (forged.adopt, forged.cancel, forged.retire,
                       forged.owned_delivery):
            with self.assertRaises(DeliverySlotError):
                action()
        self.assertEqual(registry.active_slots(), (("valid", 1, "reserved"),))
        self.assertTrue(legitimate.retire())

    def test_oversized_slot_identity_is_rejected_without_a_large_copy(self):
        registry = DeliverySlotRegistry(max_slots=1)
        oversized = " " + "x" * 2_000_000 + " "
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
            with self.assertRaises(ValueError):
                registry.register(oversized, 1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not tracing:
                tracemalloc.stop()
        self.assertLess(peak - baseline, 100_000)
        self.assertEqual(registry.active_slots(), ())

    def test_oversized_delivery_identity_is_rejected_without_a_large_copy(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        params = options()
        params["team_id"] = " " + "x" * 2_000_000 + " "
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
            with self.assertRaises(DeliverySlotError):
                slot.construct(budget, store, **params)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not tracing:
                tracemalloc.stop()
        self.assertLess(peak - baseline, 100_000)
        self.assertEqual(slot.state, "failed_clean")
        self.assertTrue(slot.retire())
        self.assertFalse(budget.active_claims())

    def test_bounded_slots_and_stale_generation_cannot_alias_reuse(self):
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        with self.assertRaises(DeliverySlotFull):
            registry.register("other", 2)
        self.assertTrue(slot.retire())
        with self.assertRaises(DeliverySlotError):
            registry.register("slot", 1)
        registry.register("slot", 2)
        replacement = registry.lookup("slot", 2)
        with self.assertRaises(DeliverySlotError):
            slot.retire()
        with self.assertRaises(DeliverySlotError):
            registry.lookup("slot", 1)
        self.assertTrue(replacement.retire())

    def test_same_slot_key_in_distinct_registries_is_not_job_identity(self):
        # Registry/process epoch and expected job binding belong to the
        # coordinator: matching slot keys alone cannot authorize routing.
        budget, store = resources()
        first_registry = DeliverySlotRegistry(max_slots=1)
        second_registry = DeliverySlotRegistry(max_slots=1)
        first_registry.register("same-slot", 1)
        second_registry.register("same-slot", 1)
        first = first_registry.lookup("same-slot", 1)
        second = second_registry.lookup("same-slot", 1)
        other_options = options()
        other_options.update(broker_id="1" * 32, job_id=8, team_id="other")
        first.construct(budget, store, **options())
        second.construct(budget, store, **other_options)
        first_child = first.adopt()
        second_child = second.adopt()
        self.assertIsNot(first_child, second_child)
        self.assertNotEqual(first_child.identity.broker_id, second_child.identity.broker_id)
        self.assertNotEqual(first_child.identity.job_id, second_child.identity.job_id)
        self.assertNotEqual(first_child.identity.team_id, second_child.identity.team_id)
        self.assertTrue(first_child.close(1))
        self.assertTrue(second_child.close(1))
        self.assertTrue(first.retire())
        self.assertTrue(second.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_adoption_return_interruption_does_not_lose_exact_delivery(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=2)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        slot.construct(budget, store, **options())
        original = slot.adopt()

        def caller():
            observed = slot.adopt()
            return observed

        target = next(op.offset for op in dis.get_instructions(caller)
                      if op.opname == "STORE_FAST" and op.argval == "observed")

        def interrupt(frame, action, _arg):
            if frame.f_code is caller.__code__:
                if action == "call":
                    frame.f_trace_opcodes = True
                if action == "opcode" and frame.f_lasti == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt("simulated caller interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                caller()
        finally:
            sys.settrace(None)
        self.assertIs(registry.lookup("slot", 1).adopt(), original)
        self.assertEqual(budget.snapshot().used_bytes, 65536)
        self.assertTrue(original.close(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interruption_after_slot_insertion_is_discoverable(self):
        registry = DeliverySlotRegistry(max_slots=1)
        instructions = list(dis.get_instructions(registry.register))
        insertion = next(i for i, op in enumerate(instructions)
                         if op.opname == "STORE_SUBSCR")
        target = instructions[insertion + 1].offset

        def interrupt(frame, action, _arg):
            if frame.f_code is type(registry).register.__code__:
                if action == "call":
                    frame.f_trace_opcodes = True
                if action == "opcode" and frame.f_lasti == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt("lost registration return")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(KeyboardInterrupt):
                registry.register("known", 9)
        finally:
            sys.settrace(None)
        self.assertEqual(registry.active_slots(), (("known", 9, "reserved"),))
        self.assertTrue(registry.lookup("known", 9).retire())
        with self.assertRaises(DeliverySlotError):
            registry.register("new", 9)
        registry.register("new", 10)
        self.assertTrue(registry.lookup("new", 10).retire())

    def test_interruption_after_constructor_keeps_exact_child_charged(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("known", 1)
        slot = registry.lookup("known", 1)
        instructions = list(dis.get_instructions(type(slot).construct))
        init_load = next(i for i, op in enumerate(instructions)
                         if op.opname == "LOAD_ATTR" and op.argval == "__init__")
        constructor_call = next(i for i in range(init_load, len(instructions))
                                if instructions[i].opname == "CALL")
        target = instructions[constructor_call + 1].offset

        def interrupt(frame, action, _arg):
            if frame.f_code is type(slot).construct.__code__:
                if action == "call":
                    frame.f_trace_opcodes = True
                if action == "opcode" and frame.f_lasti == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt("lost constructor return")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliverySlotError):
                slot.construct(budget, store, **options())
        finally:
            sys.settrace(None)
        self.assertEqual(slot.state, "quarantined")
        child = registry.lookup("known", 1).owned_delivery()
        self.assertIs(type(child), TraceDelivery)
        self.assertEqual(budget.snapshot().used_bytes, 65536)
        with self.assertRaises(DeliverySlotError):
            slot.adopt()
        with self.assertRaises(DeliverySlotError):
            slot.retire()
        # This test knows construction finished and can safely dispose it.
        self.assertTrue(child.close(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_uninitialized_allocation_can_retire(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        target = next(op.offset for op in dis.get_instructions(type(slot).construct)
                      if op.opname == "STORE_ATTR" and op.argval == "_delivery")

        def interrupt(frame, action, _arg):
            if frame.f_code is type(slot).construct.__code__:
                if action == "call":
                    frame.f_trace_opcodes = True
                if action == "opcode" and frame.f_lasti == target:
                    sys.settrace(None)
                    raise MemoryError("allocation handoff")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaises(DeliverySlotError):
                slot.construct(budget, store, **options())
        finally:
            sys.settrace(None)
        self.assertEqual(slot.state, "failed_clean")
        self.assertFalse(budget.active_requests())
        self.assertTrue(slot.retire())
        registry.register("replacement", 2)
        self.assertTrue(registry.lookup("replacement", 2).retire())

    def test_concurrent_constructor_rejection_does_not_touch_active_child(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        entered, resume = threading.Event(), threading.Event()
        original = TraceDelivery.__init__
        errors = []

        def delayed(instance, *args, **kwargs):
            entered.set()
            resume.wait(3)
            original(instance, *args, **kwargs)

        def construct():
            try:
                slot.construct(budget, store, **options())
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(TraceDelivery, "__init__", new=delayed):
            thread = threading.Thread(target=construct)
            thread.start()
            self.assertTrue(entered.wait(1))
            with self.assertRaises(DeliverySlotError):
                slot.construct(budget, store, **options())
            self.assertEqual(slot.state, "constructing")
            with self.assertRaises(DeliverySlotError):
                slot.retire()
            resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        child = slot.adopt()
        self.assertTrue(child.close(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_unfinished_child_cannot_be_retired(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        slot.construct(budget, store, **options())
        child = slot.adopt()
        self.assertFalse(slot.retire())
        with self.assertRaises(DeliverySlotFull):
            registry.register("other", 2)
        self.assertTrue(slot.cancel())
        self.assertFalse(slot.cancel())
        self.assertTrue(child.drain(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_blocked_child_cancellation_does_not_hold_registry_or_release_slot(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        slot.construct(budget, store, **options())
        child = slot.adopt()
        completed = threading.Event()
        failures = []

        def cancel():
            try:
                slot.cancel()
            except BaseException as exc:
                failures.append(exc)
            finally:
                completed.set()

        child._cancel_signal_lock.acquire()
        try:
            thread = threading.Thread(target=cancel)
            thread.start()
            deadline = time.monotonic() + 2
            while slot.state != "cancelling" and time.monotonic() < deadline:
                completed.wait(.005)
            self.assertEqual(slot.state, "cancelling")
            self.assertFalse(completed.is_set())
            self.assertEqual(registry.active_slots(), (("slot", 1, "cancelling"),))
            with self.assertRaises(DeliverySlotError):
                slot.adopt()
            with self.assertRaises(DeliverySlotError):
                slot.retire()
            self.assertEqual(budget.snapshot().used_bytes, 65536)
        finally:
            child._cancel_signal_lock.release()
        thread.join(3)
        self.assertTrue(completed.is_set())
        self.assertFalse(failures)
        self.assertTrue(child.drain(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_during_store_copy_retains_delivery_and_claims(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        slot.construct(budget, store, **options())
        child = slot.adopt()
        child.reserve_source(8)
        source = CapturedSlice(
            run_id="run", job_id="lg-" + "0" * 32 + "-7", request_id="request",
            sample_id="sample", batch_position=0, token_positions=(0,),
            forward_index=0, generation_step=None, sequence=0,
            module_path="block.0", layer_index=0, module_kind="block",
            output_path=(), invocation=0, source_shape=(1, 1, 2),
            source_dtype="torch.float32", source_device="cpu", saved_shape=(1, 2),
            saved_dtype="torch.float32", saved_device="cpu",
            value=torch.tensor([[1.0, 2.0]]),
        )
        entered, resume = threading.Event(), threading.Event()
        failures = []
        from looking_glass.trace_store import _clone_tensor

        def slow_clone(value):
            entered.set()
            resume.wait(3)
            return _clone_tensor(value)

        def stage():
            try:
                child.stage(source)
            except BaseException as exc:
                failures.append(exc)

        with mock.patch("looking_glass.trace_store._clone_tensor", new=slow_clone):
            thread = threading.Thread(target=stage)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertTrue(slot.cancel())
            self.assertFalse(slot.retire())
            self.assertIs(slot.owned_delivery(), child)
            self.assertEqual(budget.snapshot().used_bytes, 65536 + 8 + 8)
            self.assertEqual(store.usage().claimed_bytes, 8)
            resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        failures.clear()
        source = None
        child.confirm_source_disposed()
        self.assertTrue(child.drain(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancellation_during_construction_keeps_the_slot_until_drain(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        entered, resume = threading.Event(), threading.Event()
        original = TraceDelivery.__init__
        errors = []

        def delayed(instance, *args, **kwargs):
            entered.set()
            resume.wait(3)
            original(instance, *args, **kwargs)

        def construct():
            try:
                slot.construct(budget, store, **options())
            except DeliverySlotError as exc:
                errors.append(exc)

        with mock.patch.object(TraceDelivery, "__init__", new=delayed):
            worker = threading.Thread(target=construct)
            worker.start()
            self.assertTrue(entered.wait(1))
            self.assertTrue(slot.cancel())
            with self.assertRaises(DeliverySlotError):
                slot.adopt()
            with self.assertRaises(DeliverySlotError):
                slot.retire()
            resume.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors)
        delivery = slot.owned_delivery()
        self.assertTrue(delivery.drain(1))
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_failed_constructor_retains_exact_guard_and_quarantines(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        with mock.patch.object(store, "request", side_effect=RuntimeError("store")):
            with mock.patch.object(HostClaim, "release", side_effect=MemoryError("cleanup")):
                with self.assertRaises(DeliverySlotError):
                    slot.construct(budget, store, **options())
        self.assertEqual(slot.state, "quarantined")
        self.assertIs(slot.failure_guard, budget.active_guards()[0])
        with self.assertRaises(DeliverySlotError):
            slot.adopt()
        with self.assertRaises(DeliverySlotError):
            slot.retire()
        # The registry never releases claims; only a trusted supervisor can
        # reconcile this payload-free constructor failure.
        for claim in budget.active_claims():
            claim.release()
        self.assertTrue(slot.failure_guard.cancel_and_close_empty())

    def test_proven_clean_construction_rejections_can_retire(self):
        for denied, bad_config in ((True, False), (False, True)):
            with self.subTest(denied=denied, bad_config=bad_config):
                budget = HostBudget(max_bytes=0 if denied else 200000, max_items=8,
                                    max_requests=4, max_operations=4)
                store = HostTraceStore(max_tensor_bytes=64, max_items=4)
                registry = DeliverySlotRegistry(max_slots=1)
                registry.register("slot", 1)
                slot = registry.lookup("slot", 1)
                params = options()
                if bad_config:
                    params["broker_id"] = "bad"
                with self.assertRaises(DeliverySlotError):
                    slot.construct(budget, store, **params)
                self.assertEqual(slot.state, "failed_clean")
                self.assertEqual(budget.snapshot().used_bytes, 0)
                self.assertFalse(budget.active_requests())
                self.assertFalse(budget.active_claims())
                with self.assertRaises(DeliverySlotError):
                    slot.adopt()
                self.assertTrue(slot.retire())
                registry.register("new", 2)
                self.assertTrue(registry.lookup("new", 2).retire())

    def test_clean_failure_never_releases_a_colliding_foreign_generation(self):
        budget, store = resources()
        other = budget.open_request("delivery-" + "a" * 32)
        claim = other.reserve("other", "data", 9)
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        with mock.patch("looking_glass.trace_transport.uuid.uuid4",
                        return_value=types.SimpleNamespace(hex="a" * 32)):
            with self.assertRaises(DeliverySlotError):
                slot.construct(budget, store, **options())
        self.assertEqual(slot.state, "failed_clean")
        self.assertEqual(claim.info().nbytes, 9)
        self.assertTrue(slot.retire())
        self.assertEqual(budget.snapshot().used_bytes, 9)
        claim.release()
        other.close()

    def test_construction_evidence_interruptions_fail_closed(self):
        instructions = list(dis.get_instructions(TraceDelivery.__init__))
        markers = [i for i, op in enumerate(instructions)
                   if op.opname == "STORE_ATTR" and op.argval == "_construction_clean"]
        self.assertEqual(len(markers), 3)
        open_load = next(i for i, op in enumerate(instructions)
                         if op.opname == "LOAD_METHOD" and op.argval == "open")
        open_call = next(i for i in range(open_load, len(instructions))
                         if instructions[i].opname == "CALL")
        metadata_assignments = [i for i, op in enumerate(instructions)
                                if op.opname == "STORE_ATTR" and op.argval == "_metadata_claim"]
        cases = (
            ("before first evidence", markers[0], False, "quarantined"),
            ("after initial clean", markers[0] + 1, False, "failed_clean"),
            ("before dirty", markers[1], False, "failed_clean"),
            ("after dirty", markers[1] + 1, False, "quarantined"),
            ("after guard open", open_call + 1, False, "failed_clean"),
            ("after claim before assignment", metadata_assignments[-1], False, "failed_clean"),
            ("after guarded cleanup", markers[2] + 1, True, "failed_clean"),
        )
        for label, index, denied, expected in cases:
            with self.subTest(label=label):
                budget = HostBudget(max_bytes=0 if denied else 200000, max_items=8,
                                    max_requests=4, max_operations=4)
                store = HostTraceStore(max_tensor_bytes=64, max_items=4)
                registry = DeliverySlotRegistry(max_slots=1)
                registry.register("slot", 1)
                slot = registry.lookup("slot", 1)
                target = instructions[index].offset
                observed = []

                def interrupt(frame, action, _arg):
                    if frame.f_code is TraceDelivery.__init__.__code__:
                        if action == "call":
                            frame.f_trace_opcodes = True
                        if action == "opcode" and frame.f_lasti == target:
                            observed.append(True)
                            sys.settrace(None)
                            raise KeyboardInterrupt("constructor boundary")
                    return interrupt

                try:
                    sys.settrace(interrupt)
                    with self.assertRaises(DeliverySlotError):
                        slot.construct(budget, store, **options())
                finally:
                    sys.settrace(None)
                self.assertTrue(observed)
                self.assertEqual(slot.state, expected)
                self.assertFalse(budget.active_requests())
                self.assertFalse(budget.active_claims())
                self.assertEqual(budget.snapshot().used_bytes, 0)
                if expected == "failed_clean":
                    self.assertTrue(slot.retire())
                else:
                    with self.assertRaises(DeliverySlotError):
                        slot.retire()

    def test_interrupted_cancel_does_not_lose_dominant_quarantine(self):
        budget, store = resources()
        registry = DeliverySlotRegistry(max_slots=1)
        registry.register("slot", 1)
        slot = registry.lookup("slot", 1)
        entered, resume = threading.Event(), threading.Event()
        original = TraceDelivery.__init__
        failures = []

        def delayed(instance, *args, **kwargs):
            entered.set()
            resume.wait(3)
            original(instance, *args, **kwargs)

        def construct():
            try:
                slot.construct(budget, store, **options())
            except DeliverySlotError as exc:
                failures.append(exc)

        instructions = list(dis.get_instructions(type(slot).cancel))
        signal = next(i for i, op in enumerate(instructions)
                      if op.opname == "STORE_ATTR" and op.argval == "_cancel_requested")
        target = instructions[signal + 1].offset

        def interrupt(frame, action, _arg):
            if frame.f_code is type(slot).cancel.__code__:
                if action == "call":
                    frame.f_trace_opcodes = True
                if action == "opcode" and frame.f_lasti == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt("cancel boundary")
            return interrupt

        with mock.patch.object(TraceDelivery, "__init__", new=delayed):
            thread = threading.Thread(target=construct)
            thread.start()
            self.assertTrue(entered.wait(1))
            try:
                sys.settrace(interrupt)
                with self.assertRaises(DeliverySlotError):
                    slot.cancel()
            finally:
                sys.settrace(None)
                resume.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertEqual(slot.state, "quarantined")
        with self.assertRaises(DeliverySlotError):
            slot.adopt()
        child = slot.owned_delivery()
        self.assertTrue(child.close(1))
        self.assertEqual(budget.snapshot().used_bytes, 0)


if __name__ == "__main__":
    unittest.main()
