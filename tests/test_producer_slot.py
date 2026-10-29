import concurrent.futures
import dis
import importlib
import inspect
import sys
import threading
import unittest
from unittest.mock import patch

from looking_glass.host_budget import HostBudget, HostClaim, HostClaimGuard
from looking_glass.trace_producer import (ClaimLimit, ProducerCancelled,
                                         ProducerEnvelope, ProducerProfile)


def slot_module():
    try:
        return importlib.import_module("looking_glass.producer_slot")
    except ModuleNotFoundError as exc:
        raise AssertionError("producer adoption slot is unavailable") from exc


def ledger(max_bytes=30, max_items=7):
    return HostBudget(max_bytes=max_bytes, max_items=max_items, max_requests=4,
                      max_operations=4)


def profile():
    return ProducerProfile(ClaimLimit(8, 2), ClaimLimit(4, 1),
                           ClaimLimit(12, 3), ClaimLimit(6, 1))


def dispose(child):
    for stage in ("input_copies", "output", "trace_bundle", "workspace"):
        child.confirm_disposed(stage)
    child.close()


def interrupt_at(method, line_text, opname, call):
    source, first = inspect.getsourcelines(method)
    lines = [first + i for i, line in enumerate(source) if line_text in line]
    if len(lines) != 1:
        raise AssertionError(f"ambiguous source line {line_text!r}: {lines}")
    offsets = [i.offset for i in dis.get_instructions(method)
               if i.positions.lineno == lines[0] and i.opname == opname]
    if len(offsets) != 1:
        raise AssertionError(f"ambiguous opcode {opname}: {offsets}")

    def tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        if frame.f_code is method.__code__ and event == "opcode" and frame.f_lasti == offsets[0]:
            sys.settrace(None)
            raise KeyboardInterrupt("injected interruption")
        return tracer

    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(None)


class ProducerSlotTests(unittest.TestCase):
    def test_registry_is_available(self):
        self.assertTrue(hasattr(slot_module(), "ProducerSlotRegistry"))

    def test_caller_known_identity_supports_lost_adoption_assignment(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=2)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger()
        self.assertIsNone(slot.construct(budget, "request", profile()))

        def caller():
            envelope = slot.adopt()
            return envelope

        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(caller, "envelope = slot.adopt()", "STORE_FAST", caller)
        envelope = registry.lookup("job", 1).adopt()
        self.assertIs(envelope, slot.adopt())
        self.assertEqual(budget.snapshot().used_bytes, 30)
        with self.assertRaises(slots.ProducerSlotError):
            slot.retire()
        dispose(envelope)
        self.assertTrue(slot.retire())
        self.assertEqual(registry.active_slots(), ())
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_slot_registration_is_bounded_and_rejects_reused_generation(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        old = registry.lookup("job", 1)
        with self.assertRaises(slots.ProducerSlotFull):
            registry.register("other", 2)
        self.assertTrue(old.cancel())
        self.assertTrue(old.retire())
        with self.assertRaises(slots.ProducerSlotError):
            registry.register("job", 1)
        registry.register("job", 2)
        with self.assertRaises(slots.ProducerSlotError):
            old.adopt()
        with self.assertRaises(slots.ProducerSlotError):
            registry.lookup("job", 1)
        registry.lookup("job", 2).cancel()
        registry.lookup("job", 2).retire()

    def test_interrupted_envelope_close_can_be_retried_before_retirement(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger()
        slot.construct(budget, "request", profile())
        envelope = slot.adopt()
        for stage in ("input_copies", "output", "trace_bundle", "workspace"):
            envelope.confirm_disposed(stage)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(ProducerEnvelope.close, "closed = self._request.close()",
                         "STORE_FAST", envelope.close)
        with self.assertRaises(slots.ProducerSlotError):
            slot.retire()
        self.assertFalse(envelope.close())
        self.assertTrue(slot.retire())

    def test_failed_admission_cleans_own_claims_before_retirement(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger(max_bytes=29)
        with self.assertRaises(slots.ProducerSlotError):
            slot.construct(budget, "request", profile())
        self.assertEqual(slot.state, "failed_clean")
        self.assertEqual((budget.snapshot().used_bytes, budget.snapshot().requests), (0, ()))
        self.assertTrue(slot.retire())

    def test_cancellation_while_reserving_never_releases_live_claims(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger()
        entered, resume = threading.Event(), threading.Event()
        original = HostClaimGuard.reserve

        def reserve_then_wait(guard):
            result = original(guard)
            if guard._stage == "input_copies":
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test did not resume")
            return result

        with patch.object(HostClaimGuard, "reserve", reserve_then_wait):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(slot.construct, budget, "request", profile())
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertTrue(slot.cancel())
                    with self.assertRaises(slots.ProducerSlotError):
                        slot.adopt()
                    with self.assertRaises(slots.ProducerSlotError):
                        slot.guards()
                    with self.assertRaises(slots.ProducerSlotError):
                        slot.retire()
                    self.assertEqual(budget.snapshot().used_bytes, 8)
                finally:
                    resume.set()
                with self.assertRaises(slots.ProducerSlotError):
                    future.result(timeout=5)
        self.assertEqual(slot.state, "cancelled")
        self.assertEqual(budget.snapshot().used_bytes, 30)
        child = slot.owned_envelope()
        with self.assertRaises(ProducerCancelled):
            child.run(lambda: None)
        dispose(child)
        self.assertTrue(slot.retire())

    def test_failed_cleanup_quarantines_and_keeps_claims_discoverable(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger(max_bytes=29)
        with patch.object(HostClaim, "release", side_effect=MemoryError("cleanup failed")):
            with self.assertRaises(slots.ProducerSlotError):
                slot.construct(budget, "request", profile())
        self.assertEqual(slot.state, "quarantined")
        self.assertGreater(budget.snapshot().used_bytes, 0)
        self.assertTrue(budget.active_claim_guards())
        with self.assertRaises(slots.ProducerSlotError):
            slot.retire()

    def test_foreign_claim_stays_charged_when_admission_fails(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger(max_items=8)
        entered, resume = threading.Event(), threading.Event()
        original = HostClaimGuard.reserve

        def reserve_then_wait(guard):
            result = original(guard)
            if guard._stage == "input_copies":
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test did not resume")
            return result

        with patch.object(HostClaimGuard, "reserve", reserve_then_wait):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(slot.construct, budget, "request", profile())
                try:
                    self.assertTrue(entered.wait(5))
                    request, = budget.active_requests()
                    foreign = request.reserve("external", "payload", 4)
                    payload = bytearray(4)
                finally:
                    resume.set()
                with self.assertRaises(slots.ProducerSlotError):
                    future.result(timeout=5)
        self.assertEqual(len(payload), 4)
        self.assertEqual(slot.state, "quarantined")
        self.assertEqual(budget.snapshot().used_bytes, 4)
        self.assertEqual(foreign.info().owners, ("external",))
        del payload
        foreign.release()
        request_guard, _ = slot.guards()
        request_guard.cancel_and_close_empty()

    def test_transferred_claim_stays_charged_when_admission_fails(self):
        slots = slot_module()
        registry = slots.ProducerSlotRegistry(max_slots=1)
        registry.register("job", 1)
        slot = registry.lookup("job", 1)
        budget = ledger(max_bytes=29)
        entered, resume = threading.Event(), threading.Event()
        original = HostClaimGuard.reserve

        def reserve_then_wait(guard):
            result = original(guard)
            if guard._stage == "input_copies":
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test did not resume")
            return result

        with patch.object(HostClaimGuard, "reserve", reserve_then_wait):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(slot.construct, budget, "request", profile())
                try:
                    self.assertTrue(entered.wait(5))
                    claim, = budget.active_claims()
                    transferred = claim.transfer("external")
                finally:
                    resume.set()
                with self.assertRaises(slots.ProducerSlotError):
                    future.result(timeout=5)
        self.assertEqual(slot.state, "quarantined")
        self.assertEqual(budget.snapshot().used_bytes, 8)
        self.assertEqual(transferred.info().owners[-1], "external")
        transferred.release()
        request_guard, _ = slot.guards()
        request_guard.cancel_and_close_empty()


if __name__ == "__main__":
    unittest.main()
