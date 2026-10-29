import concurrent.futures
import gc
import inspect
import sys
import threading
import unittest

import weakref
from unittest.mock import patch

if __package__:
    from .opcode_support import offsets_at_line, run_opcode_trace
else:
    from opcode_support import offsets_at_line, run_opcode_trace

from looking_glass.host_budget import (HostBudget, HostBudgetExceeded, HostBusy,
                                       HostRequest, HostRequestCancelled)
from looking_glass.trace_producer import (
    ClaimLimit, ProducerEnvelope, ProducerProfile, ProducerCancelled,
    ProducerQuarantined,
)


def profile():
    return ProducerProfile(input_copies=ClaimLimit(8, 2), output=ClaimLimit(4, 1),
                           trace_bundle=ClaimLimit(12, 3), workspace=ClaimLimit(6, 1))


def budget(max_bytes=30, max_items=7):
    return HostBudget(max_bytes=max_bytes, max_items=max_items, max_requests=4,
                      max_operations=4)


def dispose_all(envelope):
    for stage in ("input_copies", "output", "trace_bundle", "workspace"):
        envelope.confirm_disposed(stage)
    envelope.close()


def interrupt_opcode(method, line_text, opname, call, *, occurrence=None):
    """Interrupt one specific bytecode boundary, including call-to-store gaps."""
    source, first = inspect.getsourcelines(method)
    lines = [first + index for index, line in enumerate(source) if line_text in line]
    if occurrence is not None:
        lines = [lines[occurrence]]
    if len(lines) != 1:
        raise AssertionError(f"expected one source line containing {line_text!r}")
    matches = offsets_at_line(method, lines[0], opname)
    if len(matches) != 1:
        raise AssertionError(f"expected one {opname} at {line_text!r}: {matches}")

    def tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        if (frame.f_code is method.__code__ and event == "opcode"
                and frame.f_lasti == matches[0]):
            sys.settrace(None)
            raise KeyboardInterrupt("injected interruption")
        return tracer

    run_opcode_trace(tracer, call)


class ProducerEnvelopeTests(unittest.TestCase):
    def test_profile_rejects_invalid_limits_and_non_cpu_device(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ClaimLimit(value, 1)
        for value in (0, -1, True):
            with self.subTest(items=value), self.assertRaises(ValueError):
                ClaimLimit(1, value)
        with self.assertRaises(ValueError):
            ProducerProfile(ClaimLimit(1, 1), ClaimLimit(1, 1), ClaimLimit(1, 1),
                            ClaimLimit(1, 1), device="cuda")

    def test_admitted_profile_is_read_only(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        with self.assertRaises(AttributeError):
            envelope.profile = ProducerProfile(*(ClaimLimit(0, 1) for _ in range(4)))
        self.assertEqual(envelope.profile.trace_bundle.nbytes, 12)
        dispose_all(envelope)

    def test_all_claims_are_reserved_before_forward_and_stay_until_disposal(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        observed = []

        def run():
            state = ledger.snapshot()
            observed.append((state.used_bytes, state.used_items, len(state.operations)))
            return {"logits": [1], "events": [2, 3]}

        result = envelope.run(run)
        self.assertEqual(result, {"logits": [1], "events": [2, 3]})
        self.assertEqual(observed, [(30, 7, 1)])
        self.assertEqual((ledger.snapshot().used_bytes, ledger.snapshot().used_items), (30, 7))
        with self.assertRaises(HostBusy):
            envelope.close()
        envelope.confirm_disposed("workspace")
        self.assertEqual(ledger.snapshot().used_bytes, 24)
        envelope.confirm_disposed("input_copies")
        envelope.confirm_disposed("output")
        envelope.confirm_disposed("trace_bundle")
        envelope.close()
        self.assertEqual((ledger.snapshot().used_bytes, len(ledger.snapshot().requests)), (0, 0))

    def test_failed_admission_rolls_back_all_claims_without_forward(self):
        ledger = budget(max_bytes=29)
        with self.assertRaises(HostBudgetExceeded):
            ProducerEnvelope.acquire(ledger, "job-1", profile())
        state = ledger.snapshot()
        self.assertEqual((state.used_bytes, state.used_items, len(state.requests)), (0, 0, 0))

    def test_admission_uses_public_request_guard(self):
        ledger = budget()
        with patch.object(HostBudget, "open_request", side_effect=AssertionError("unguarded")):
            envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        self.assertEqual(len(ledger.active_guards()), 1)
        dispose_all(envelope)

    def test_interrupt_after_adoption_return_recovers_only_new_request(self):
        ledger = budget()
        existing = ledger.open_request("existing")
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.acquire.__func__,
                             "request = guard.adopt()", "STORE_FAST",
                             lambda: ProducerEnvelope.acquire(ledger, "new", profile()),
                             occurrence=0)
        state = ledger.snapshot()
        self.assertEqual([(r.request_id, r.cancelled) for r in state.requests],
                         [("existing", False)])
        existing.close()

    def test_interrupt_after_guard_open_recovers_only_new_request(self):
        ledger = budget()
        existing = ledger.open_request("existing")
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.acquire.__func__, "guard.open()", "POP_TOP",
                             lambda: ProducerEnvelope.acquire(ledger, "new", profile()))
        self.assertEqual([(r.request_id, r.cancelled) for r in ledger.snapshot().requests],
                         [("existing", False)])
        existing.close()

    def test_existing_request_is_never_cleaned_up_by_failed_acquisition(self):
        ledger = budget()
        existing = ledger.open_request("existing")
        with self.assertRaises(ValueError):
            ProducerEnvelope.acquire(ledger, "existing", profile())
        self.assertFalse(ledger.snapshot().requests[0].cancelled)
        claim = existing.reserve("other", "source", 1)
        claim.release()
        existing.close()

    def test_interrupted_claim_acquisitions_roll_back_every_registered_claim(self):
        original = HostRequest.reserve
        for target in ("input_copies", "output", "trace_bundle", "workspace"):
            with self.subTest(target=target):
                ledger = budget()

                def reserve_then_interrupt(request, owner, stage, nbytes, *, items=1):
                    claim = original(request, owner, stage, nbytes, items=items)
                    if stage == target:
                        raise KeyboardInterrupt("after reservation")
                    return claim

                with patch.object(HostRequest, "reserve", reserve_then_interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        ProducerEnvelope.acquire(ledger, "new", profile())
                state = ledger.snapshot()
                self.assertEqual((state.used_bytes, state.used_items, state.requests),
                                 (0, 0, ()))

    def test_failed_admission_preserves_another_owners_claim(self):
        ledger = budget(max_bytes=30, max_items=8)
        entered, resume = threading.Event(), threading.Event()
        original = HostRequest.reserve

        def reserve_then_pause(request, owner, stage, nbytes, *, items=1):
            claim = original(request, owner, stage, nbytes, items=items)
            if stage == "input_copies":
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test did not release admission")
            return claim

        with patch.object(HostRequest, "reserve", reserve_then_pause):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(ProducerEnvelope.acquire, ledger, "job-1", profile())
                self.assertTrue(entered.wait(5))
                request, = ledger.active_requests()
                foreign = request.reserve("external", "payload", 4)
                payload = bytearray(4)  # A separately owned, live allocation.
                resume.set()
                with self.assertRaises(HostBudgetExceeded):
                    future.result(timeout=5)

        self.assertEqual(len(payload), 4)
        self.assertEqual((ledger.snapshot().used_bytes, ledger.snapshot().used_items), (4, 1))
        self.assertEqual(foreign.info().owners, ("external",))
        del payload
        foreign.release()
        guard, = ledger.active_guards()
        guard.cancel_and_close_empty()

    def test_failed_admission_preserves_a_transferred_producer_claim(self):
        ledger = budget(max_bytes=29)
        entered, resume = threading.Event(), threading.Event()
        original = HostRequest.reserve

        def reserve_then_pause(request, owner, stage, nbytes, *, items=1):
            claim = original(request, owner, stage, nbytes, items=items)
            if stage == "input_copies":
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test did not release admission")
            return claim

        with patch.object(HostRequest, "reserve", reserve_then_pause):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(ProducerEnvelope.acquire, ledger, "job-1", profile())
                self.assertTrue(entered.wait(5))
                source, = ledger.active_claims()
                transferred = source.transfer("external")
                resume.set()
                with self.assertRaises(HostBudgetExceeded):
                    future.result(timeout=5)

        self.assertEqual((ledger.snapshot().used_bytes, ledger.snapshot().used_items), (8, 2))
        self.assertEqual(transferred.info().owners[-1], "external")
        transferred.release()
        guard, = ledger.active_guards()
        guard.cancel_and_close_empty()

    def test_overlapping_delivery_claims_are_charged_separately(self):
        ledger = budget(max_bytes=36, max_items=8)
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        delivery = ledger.open_request("delivery")
        source = delivery.reserve("delivery", "source", 6)
        self.assertEqual(ledger.snapshot().used_bytes, 36)
        with self.assertRaises(HostBudgetExceeded):
            delivery.reserve("delivery", "store-copy", 1)
        envelope.run(lambda: None)
        # Only an owner that has disposed all bundle aliases may make this attestation.
        envelope.confirm_disposed("trace_bundle")
        self.assertEqual(ledger.snapshot().used_bytes, 24)
        source.release()
        delivery.close()
        for stage in ("input_copies", "output", "workspace"):
            envelope.confirm_disposed(stage)
        envelope.close()

    def test_cancel_before_start_does_not_release_claims(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        self.assertTrue(envelope.cancel())
        self.assertFalse(envelope.cancel())
        called = []
        with self.assertRaises(ProducerCancelled):
            envelope.run(lambda: called.append(True))
        self.assertEqual(called, [])
        self.assertEqual(ledger.snapshot().used_bytes, 30)
        dispose_all(envelope)

    def test_interrupted_cancel_retry_tombstones_the_ledger(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.cancel, "self._request.cancel()", "LOAD_FAST",
                             envelope.cancel)
        envelope.cancel()
        self.assertTrue(ledger.snapshot().requests[0].cancelled)
        request, = ledger.active_requests()
        with self.assertRaises(HostRequestCancelled):
            request.reserve("other", "new", 0)
        dispose_all(envelope)

    def test_interrupt_after_ledger_cancel_is_already_tombstoned(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.cancel, "self._cancelled = True", "LOAD_CONST",
                             envelope.cancel)
        self.assertTrue(ledger.snapshot().requests[0].cancelled)
        request, = ledger.active_requests()
        with self.assertRaises(HostRequestCancelled):
            request.reserve("other", "new", 0)
        self.assertTrue(envelope.cancel())
        dispose_all(envelope)

    def test_cancel_during_forward_keeps_claims_pinned_and_discards_result(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        entered, resume = threading.Event(), threading.Event()

        def blocked():
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release forward")
            return object()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(envelope.run, blocked)
            try:
                self.assertTrue(entered.wait(5))
                envelope.cancel()
                with self.assertRaises(HostBusy):
                    envelope.confirm_disposed("trace_bundle")
                other = ledger.open_request("other")
                with self.assertRaises(HostBudgetExceeded):
                    other.reserve("other", "buffer", 1)
                other.close()
                self.assertEqual(ledger.snapshot().used_bytes, 30)
            finally:
                resume.set()
            with self.assertRaises(ProducerCancelled):
                future.result(timeout=5)
        self.assertEqual(ledger.snapshot().used_bytes, 30)
        dispose_all(envelope)

    def test_failure_keeps_claims_until_explicit_disposal(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())

        def fail():
            raise MemoryError("synthetic allocation failure")

        with self.assertRaises(MemoryError):
            envelope.run(fail)
        self.assertEqual(ledger.snapshot().used_bytes, 30)
        with self.assertRaises(RuntimeError):
            envelope.run(lambda: None)
        dispose_all(envelope)

    def test_failed_callback_traceback_does_not_retain_its_local_payload(self):
        class Payload:
            pass

        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        observed = []

        def fail():
            payload = Payload()
            observed.append(weakref.ref(payload))
            raise MemoryError("synthetic failure")

        with self.assertRaises(MemoryError) as caught:
            envelope.run(fail)
        gc.collect()
        self.assertIsNone(observed[0]())
        self.assertEqual(ledger.snapshot().used_bytes, 30)
        del caught
        dispose_all(envelope)

    def test_interrupt_before_returned_work_is_assigned_keeps_operation_pinned(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        pending = concurrent.futures.Future()
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.run, "result, failure = _invoke_sync(forward)",
                             "UNPACK_SEQUENCE",
                             lambda: envelope.run(lambda: pending))
        self.assertEqual(len(ledger.snapshot().operations), 1)
        with self.assertRaises(HostBusy):
            envelope.confirm_disposed("trace_bundle")
        self.assertEqual(ledger.snapshot().used_bytes, 30)

    def test_interruption_immediately_before_callback_quarantines_and_tombstones(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        called = []
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.run, "result, failure = _invoke_sync(forward)",
                             "LOAD_GLOBAL", lambda: envelope.run(lambda: called.append(True)))
        self.assertEqual(called, [])
        self.assertTrue(ledger.snapshot().requests[0].cancelled)
        self.assertEqual(len(ledger.snapshot().operations), 1)
        with self.assertRaises(HostBusy):
            envelope.confirm_disposed("trace_bundle")

    def test_deferred_work_is_quarantined_with_claims_pinned(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        pending = concurrent.futures.Future()
        with self.assertRaises(ProducerQuarantined):
            envelope.run(lambda: pending)
        self.assertEqual(len(ledger.snapshot().operations), 1)
        self.assertEqual(ledger.snapshot().used_bytes, 30)
        with self.assertRaises(HostBusy):
            envelope.confirm_disposed("trace_bundle")
        with self.assertRaises(HostBusy):
            envelope.close()

    def test_interrupted_quarantine_still_tombstones_ledger(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        pending = concurrent.futures.Future()
        with self.assertRaises(KeyboardInterrupt):
            interrupt_opcode(ProducerEnvelope.run, "self._cancelled = True", "LOAD_CONST",
                             lambda: envelope.run(lambda: pending), occurrence=0)
        self.assertTrue(ledger.snapshot().requests[0].cancelled)
        request, = ledger.active_requests()
        with self.assertRaises(HostRequestCancelled):
            request.reserve("other", "new", 0)
        self.assertEqual(len(ledger.snapshot().operations), 1)
        self.assertEqual(ledger.snapshot().used_bytes, 30)

    def test_external_unpin_cannot_release_envelope_owned_deferred_payload(self):
        class Payload:
            pass

        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        payload = Payload()
        payload_ref = weakref.ref(payload)
        pending = concurrent.futures.Future()
        pending.set_result(payload)
        with self.assertRaises(ProducerQuarantined):
            envelope.run(lambda: pending)
        del payload, pending
        operation, = ledger.active_operations()
        operation.finish()  # An external recovery step does not dispose the result.
        gc.collect()
        self.assertIsNotNone(payload_ref())
        with self.assertRaises(HostBusy):
            envelope.confirm_disposed("trace_bundle")
        with self.assertRaises(HostBusy):
            envelope.close()
        self.assertEqual(ledger.snapshot().used_bytes, 30)

    def test_two_concurrent_calls_cannot_execute_same_envelope(self):
        ledger = budget()
        envelope = ProducerEnvelope.acquire(ledger, "job-1", profile())
        entered, resume = threading.Event(), threading.Event()

        def blocked():
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release forward")
            return 7

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(envelope.run, blocked)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):
                    envelope.run(lambda: 9)
                self.assertEqual(len(ledger.snapshot().operations), 1)
            finally:
                resume.set()
            self.assertEqual(future.result(timeout=5), 7)
        dispose_all(envelope)


if __name__ == "__main__":
    unittest.main()
