import concurrent.futures
import dis
import gc
import inspect
import sys
import threading
import unittest
import weakref

from looking_glass.host_budget import (HostBudget, HostBusy, HostBudgetExceeded,
                                       HostRequestCancelled, InvalidHostHandle)


def ledger():
    return HostBudget(max_bytes=10, max_items=3, max_requests=2, max_operations=2)


def interrupt_at(method, line_text, opname, call):
    lines, first = inspect.getsourcelines(method)
    matching = [first + i for i, line in enumerate(lines) if line_text in line]
    if len(matching) != 1:
        raise AssertionError(f"ambiguous line {line_text!r}: {matching}")
    instructions = [inst.offset for inst in dis.get_instructions(method)
                    if inst.positions.lineno == matching[0] and inst.opname == opname]
    if len(instructions) != 1:
        raise AssertionError(f"ambiguous opcode {opname}: {instructions}")

    def tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        if (frame.f_code is method.__code__ and event == "opcode"
                and frame.f_lasti == instructions[0]):
            sys.settrace(None)
            raise KeyboardInterrupt("injected interruption")
        return tracer

    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(None)


class HostAdoptionTests(unittest.TestCase):
    def test_budget_offers_an_adoption_guard(self):
        self.assertTrue(hasattr(ledger(), "request_guard"))

    def test_guard_creation_does_not_register_and_adoption_retains_exact_handle(self):
        budget = ledger()
        guard = budget.request_guard("one")
        self.assertEqual(budget.snapshot().requests, ())
        self.assertIsNone(guard.open())
        first = guard.adopt()
        self.assertIs(first, guard.adopt())
        self.assertEqual(budget.active_guards(), (guard,))
        self.assertEqual(guard.generation, budget.snapshot().requests[0].generation)
        self.assertTrue(guard.cancel_and_close_empty())
        self.assertEqual(budget.active_guards(), ())

    def test_guard_return_assignment_interruption_opens_nothing(self):
        budget = ledger()

        def create():
            guard = budget.request_guard("one")
            return guard

        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(create, "guard = budget.request_guard", "STORE_FAST", create)
        self.assertEqual(budget.snapshot().requests, ())

    def test_interruption_before_registration_can_abandon_without_affecting_new_guard(self):
        budget = ledger()
        guard = budget.request_guard("one")
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(type(guard).open, "budget._requests[self._request_id] = record",
                         "STORE_SUBSCR", guard.open)
        self.assertEqual(budget.snapshot().requests, ())
        with self.assertRaises(InvalidHostHandle):
            guard.open()
        replacement = budget.request_guard("one")
        replacement.open()
        self.assertFalse(guard.cancel_and_close_empty())
        self.assertFalse(budget.snapshot().requests[0].cancelled)
        replacement.cancel_and_close_empty()

    def test_open_return_interruption_remains_recoverable_by_guard(self):
        budget = ledger()
        guard = budget.request_guard("one")
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(type(guard).open, "return None  # opened", "NOP", guard.open)
        request = guard.adopt()
        self.assertIs(budget.active_requests()[0], request)
        self.assertIs(budget.active_guards()[0], guard)
        self.assertTrue(guard.cancel_and_close_empty())

    def test_adoption_return_interruption_retains_ledger_guard(self):
        budget = ledger()
        guard = budget.request_guard("one")
        guard.open()

        def consume():
            request = guard.adopt()
            return request

        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(consume, "request = guard.adopt()", "STORE_FAST", consume)
        self.assertIs(guard.adopt(), budget.active_requests()[0])
        self.assertIs(budget.active_guards()[0], guard)
        guard.cancel_and_close_empty()

    def test_failed_competing_guard_never_cancels_existing_generation(self):
        budget = ledger()
        original = budget.open_request("same")
        other = budget.request_guard("same")
        with self.assertRaises(ValueError):
            other.open()
        self.assertFalse(other.cancel_and_close_empty())
        self.assertFalse(budget.snapshot().requests[0].cancelled)
        claim = original.reserve("owner", "payload", 1)
        claim.release()
        original.close()

    def test_stale_guard_cannot_cancel_or_close_a_reused_id(self):
        budget = ledger()
        guard = budget.request_guard("same")
        guard.open()
        old = guard.adopt()
        previous_generation = guard.generation
        old.close()
        new = budget.open_request("same")
        self.assertNotEqual(previous_generation, budget.snapshot().requests[0].generation)
        self.assertFalse(guard.cancel_and_close_empty())
        self.assertFalse(budget.snapshot().requests[0].cancelled)
        with self.assertRaises(InvalidHostHandle):
            guard.adopt()
        new.close()

    def test_closed_request_handle_is_never_reissued_as_new_generation(self):
        budget = ledger()
        guard = budget.request_guard("same")
        guard.open()
        guard.adopt().close()
        with self.assertRaises(InvalidHostHandle):
            guard.open()
        self.assertEqual(budget.snapshot().requests, ())

    def test_interrupted_retirement_does_not_touch_a_new_generation(self):
        budget = ledger()
        guard = budget.request_guard("same")
        guard.open()
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at(type(guard).cancel_and_close_empty,
                         "closed = self._handle.close()", "STORE_FAST",
                         guard.cancel_and_close_empty)
        replacement = budget.open_request("same")
        self.assertFalse(guard.cancel_and_close_empty())
        self.assertFalse(budget.snapshot().requests[0].cancelled)
        replacement.close()

    def test_cleanup_tombstones_but_never_releases_live_claims(self):
        budget = ledger()
        guard = budget.request_guard("one")
        guard.open()
        request = guard.adopt()
        claim = request.reserve("owner", "payload", 10)
        operation = request.begin_operation("copy", [claim])
        with self.assertRaises(HostBusy):
            guard.cancel_and_close_empty()
        self.assertTrue(budget.snapshot().requests[0].cancelled)
        self.assertEqual(budget.snapshot().used_bytes, 10)
        with self.assertRaises(HostRequestCancelled):
            request.reserve("owner", "other", 0)
        with self.assertRaises(HostBusy):
            claim.release()
        self.assertFalse(operation.finish())
        with self.assertRaises(HostBusy):
            guard.cancel_and_close_empty()
        claim.release()
        self.assertTrue(guard.cancel_and_close_empty())

    def test_guard_metadata_is_ledger_bounded_and_retires_on_direct_close(self):
        budget = ledger()
        first = budget.request_guard("one")
        second = budget.request_guard("two")
        third = budget.request_guard("three")
        first.open()
        second.open()
        with self.assertRaises(HostBudgetExceeded):
            third.open()
        self.assertEqual(len(budget.active_guards()), 2)
        ref = weakref.ref(first)
        first.adopt().close()
        del first
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(budget.active_guards(), (second,))
        third.open()
        second.cancel_and_close_empty()
        third.cancel_and_close_empty()

    def test_guard_remains_discoverable_when_caller_drops_it(self):
        budget = ledger()
        guard = budget.request_guard("one")
        guard.open()
        ref = weakref.ref(guard)
        del guard
        gc.collect()
        recovered, = budget.active_guards()
        self.assertIs(recovered, ref())
        recovered.cancel_and_close_empty()

    def test_cancel_racing_reserve_never_frees_a_claim(self):
        budget = ledger()
        guard = budget.request_guard("one")
        guard.open()
        request = guard.adopt()
        barrier = threading.Barrier(2)

        def reserve():
            barrier.wait(timeout=5)
            try:
                return request.reserve("owner", "payload", 10)
            except (HostRequestCancelled, InvalidHostHandle):
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reserve)
            barrier.wait(timeout=5)
            try:
                guard.cancel_and_close_empty()
            except HostBusy:
                pass
            claim = future.result(timeout=5)
        if claim is not None:
            self.assertEqual(budget.snapshot().used_bytes, 10)
            claim.release()
            guard.cancel_and_close_empty()
        self.assertEqual((budget.snapshot().used_bytes, budget.snapshot().requests), (0, ()))


if __name__ == "__main__":
    unittest.main()
