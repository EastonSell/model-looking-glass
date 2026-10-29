import importlib
import gc
import inspect
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


def host_module():
    try:
        return importlib.import_module("looking_glass.host_budget")
    except ModuleNotFoundError as exc:
        raise AssertionError("host custody ledger is unavailable") from exc


def interrupt_at_line(method, text, call):
    source, start = inspect.getsourcelines(method)
    matches = [start + index for index, line in enumerate(source) if text in line]
    if len(matches) != 1:
        raise AssertionError(f"expected one interruption point for {text!r}, got {matches}")
    line_number = matches[0]

    def tracer(frame, event, arg):
        if frame.f_code is method.__code__ and event == "line" and frame.f_lineno == line_number:
            sys.settrace(None)
            raise KeyboardInterrupt("injected interruption")
        return tracer

    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(None)


class HostBudgetTests(unittest.TestCase):
    def test_reservations_bound_simultaneous_buffers_until_explicit_release(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=10, max_items=2, max_requests=1, max_operations=2)
        request = budget.open_request("r1")
        source = request.reserve("worker", "source", 6)
        clone = request.reserve("store", "clone", 4)
        with self.assertRaises(host.HostBudgetExceeded):
            request.reserve("transport", "serialization", 1)
        self.assertEqual(budget.snapshot().used_bytes, 10)
        self.assertEqual(budget.snapshot().used_items, 2)
        self.assertTrue(source.release())
        self.assertFalse(source.release())
        self.assertEqual(budget.snapshot().used_bytes, 4)
        clone.release()
        request.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_transfer_invalidates_previous_owner_without_releasing_quota(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=5, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r1")
        source = request.reserve("worker", "logits", 5)
        delivered = source.transfer("broker")
        with self.assertRaises(host.InvalidHostHandle):
            source.release()
        self.assertEqual(budget.snapshot().used_bytes, 5)
        self.assertEqual(delivered.info().owners, ("worker", "broker"))
        delivered.release()
        with self.assertRaises(host.InvalidHostHandle):
            source.release()
        request.close()

    def test_cancellation_keeps_inflight_copy_charged_until_completion_and_disposal(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=10, max_items=2, max_requests=1, max_operations=2)
        request = budget.open_request("r1")
        source = request.reserve("worker", "source", 5)
        clone = request.reserve("store", "clone", 5)
        copying = request.begin_operation("clone", [source, clone])
        request.cancel()
        with self.assertRaises(host.HostBusy):
            clone.release()
        with self.assertRaises(host.HostRequestCancelled):
            request.reserve("store", "new-clone", 1)
        self.assertEqual(budget.snapshot().used_bytes, 10)
        self.assertFalse(copying.finish())
        source.release()
        clone.release()
        request.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_public_limits_cannot_be_changed_after_configuration(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        for name in ("max_bytes", "max_items", "max_requests", "max_operations", "max_transfers"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(budget, name, 1000)
        request = budget.open_request("r")
        with self.assertRaises(host.HostBudgetExceeded):
            request.reserve("worker", "source", 2)
        request.close()

    def test_zero_byte_claims_still_consume_items_and_request_capacity(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=0, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("worker", "empty", 0)
        with self.assertRaises(host.HostBudgetExceeded):
            request.reserve("worker", "another", 0)
        with self.assertRaises(host.HostBudgetExceeded):
            budget.open_request("another")
        with self.assertRaises(ValueError):
            request.reserve("worker", "invalid", 0, items=0)
        claim.release()
        request.close()

    def test_shared_source_stays_pinned_until_all_operations_finish(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=6, max_items=3, max_requests=1, max_operations=2)
        request = budget.open_request("r")
        source = request.reserve("worker", "source", 2)
        left = request.reserve("left", "clone", 2)
        right = request.reserve("right", "clone", 2)
        a = request.begin_operation("copy-left", [source, left])
        b = request.begin_operation("copy-right", [source, right])
        self.assertEqual(source.info().active_operations, 2)
        self.assertTrue(a.finish())
        with self.assertRaises(host.HostBusy):
            source.release()
        with self.assertRaises(host.HostBusy):
            source.transfer("another")
        self.assertTrue(b.finish())
        with self.assertRaises(host.InvalidHostHandle):
            b.finish()
        for claim in (source, left, right):
            claim.release()
        request.close()

    def test_stale_foreign_duplicate_and_forged_handles_fail_without_changing_state(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=2, max_items=2, max_requests=2, max_operations=1)
        first = budget.open_request("first")
        second = budget.open_request("second")
        claim = first.reserve("worker", "source", 1)
        fake = host.HostClaim(budget, claim.info().claim_id)
        with self.assertRaises(host.InvalidHostHandle):
            fake.release()
        with self.assertRaises(host.InvalidHostHandle):
            second.begin_operation("foreign", [claim])
        with self.assertRaises(ValueError):
            first.begin_operation("duplicate", [claim, claim])
        self.assertEqual(budget.snapshot().used_bytes, 1)
        moved = claim.transfer("broker")
        with self.assertRaises(host.InvalidHostHandle):
            claim.info()
        moved.release()
        first.close()
        second.close()

    def test_cancelled_claim_can_transfer_only_for_cleanup(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("worker", "source", 1)
        self.assertTrue(request.cancel())
        self.assertFalse(request.cancel())
        with self.assertRaises(host.HostRequestCancelled):
            claim.transfer("client")
        with self.assertRaises(host.HostRequestCancelled):
            request.begin_operation("later", [claim])
        cleanup = claim.transfer("cleanup", cleanup=True)
        self.assertEqual(cleanup.info().cleanup_transfers, (True,))
        with self.assertRaises(host.HostBusy):
            request.close()
        cleanup.release()
        request.close()
        replacement = budget.open_request("r")
        with self.assertRaises(host.InvalidHostHandle):
            request.cancel()
        replacement.close()

    def test_bounded_transfer_history_rejects_without_losing_current_owner(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1,
                                 max_operations=1, max_transfers=1)
        request = budget.open_request("r")
        claim = request.reserve("one", "source", 1).transfer("two")
        with self.assertRaises(host.HostBudgetExceeded):
            claim.transfer("three")
        self.assertEqual(claim.info().owners, ("one", "two"))
        request.cancel()
        claim = claim.transfer("cleanup", cleanup=True)
        self.assertEqual(claim.info().owners, ("one", "two", "cleanup"))
        with self.assertRaises(host.HostBudgetExceeded):
            claim.transfer("another-cleanup", cleanup=True)
        claim.release()
        request.close()

    def test_request_generations_distinguish_reused_ids_in_provenance(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        first = budget.open_request("same")
        first_generation = budget.snapshot().requests[0].generation
        first.close()
        second = budget.open_request("same")
        claim = second.reserve("owner", "source", 1)
        operation = second.begin_operation("copy", [claim])
        snapshot = budget.snapshot()
        generation = snapshot.requests[0].generation
        self.assertNotEqual(first_generation, generation)
        self.assertEqual(snapshot.claims[0].request_generation, generation)
        self.assertEqual(snapshot.operations[0].request_generation, generation)
        operation.finish()
        claim.release()
        second.close()

    def test_cancel_does_not_allocate_a_replacement_record(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        with patch.object(host, "replace", side_effect=MemoryError("no space")):
            self.assertTrue(request.cancel())
        with self.assertRaises(host.HostRequestCancelled):
            request.reserve("owner", "source", 1)
        request.close()

    def test_operation_rejects_oversized_claim_list_before_copying_it(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("owner", "source", 1)
        with self.assertRaises(ValueError):
            request.begin_operation("invalid", [claim, claim])
        self.assertEqual(len(budget.snapshot().operations), 0)
        claim.release()
        request.close()

    def test_forgotten_handles_remain_charged_and_available_for_recovery(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("owner", "source", 1)
        operation = request.begin_operation("copy", [claim])
        claim_ref, operation_ref = weakref.ref(claim), weakref.ref(operation)
        del claim, operation
        gc.collect()
        self.assertEqual(budget.snapshot().used_bytes, 1)
        recovered_operation, = budget.active_operations()
        recovered_claim, = budget.active_claims()
        self.assertIs(recovered_operation, operation_ref())
        self.assertIs(recovered_claim, claim_ref())
        recovered_operation.finish()
        recovered_claim.release()
        request.close()

    def test_concurrent_reservations_never_exceed_aggregate_limits(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=10, max_items=3, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        barrier = threading.Barrier(8)

        def reserve():
            try:
                claim = request.reserve("worker", "source", 3)
            except host.HostBudgetExceeded:
                claim = None
            barrier.wait(timeout=5)
            if claim is not None:
                claim.release()
                return True
            return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            accepted = list(executor.map(lambda _: reserve(), range(8)))
        self.assertEqual(sum(accepted), 3)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        request.close()

    def test_interrupted_registration_and_transfer_roll_back(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=2, max_items=2, max_requests=1, max_operations=1)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostBudget.open_request, "return handle", lambda: budget.open_request("r"))
        self.assertEqual(budget.snapshot().requests, ())
        request = budget.open_request("r")
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostRequest.reserve, "return handle", lambda: request.reserve("a", "source", 1))
        self.assertEqual(budget.snapshot().used_bytes, 0)
        claim = request.reserve("a", "source", 1)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostRequest.begin_operation, "return handle", lambda: request.begin_operation("copy", [claim]))
        self.assertEqual(budget.snapshot().operations, ())
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostClaim.transfer, "return handle", lambda: claim.transfer("b"))
        self.assertEqual(claim.info().owners, ("a",))
        claim.release()
        request.close()

    def test_interrupted_release_and_close_are_retryable(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("owner", "source", 1)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostClaim.release, "del budget._claims", claim.release)
        self.assertEqual(budget.snapshot().used_bytes, 1)
        self.assertTrue(claim.release())
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostRequest.close, "del budget._requests", request.close)
        self.assertEqual(len(budget.snapshot().requests), 1)
        self.assertTrue(request.close())

    def test_interrupt_after_removal_preserves_idempotent_release_and_close(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("owner", "source", 1)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostClaim.release, "if self._claim_id not in budget._claims", claim.release)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        self.assertFalse(claim.release())
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostRequest.close, "if self._request_id not in budget._requests", request.close)
        replacement = budget.open_request("r")
        self.assertFalse(request.close())
        self.assertEqual(len(budget.snapshot().requests), 1)
        replacement.close()

    def test_interrupted_operation_finish_keeps_claims_pinned_for_retry(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        claim = request.reserve("owner", "source", 1)
        operation = request.begin_operation("copy", [claim])
        with self.assertRaises(KeyboardInterrupt):
            interrupt_at_line(host.HostOperation.finish, "return live", operation.finish)
        self.assertEqual(len(budget.snapshot().operations), 1)
        with self.assertRaises(host.HostBusy):
            claim.release()
        self.assertTrue(operation.finish())
        claim.release()
        request.close()

    def test_invalid_and_over_limit_requests_leave_the_ledger_unchanged(self):
        host = host_module()
        budget = host.HostBudget(max_bytes=1, max_items=1, max_requests=1, max_operations=1)
        request = budget.open_request("r")
        before = budget.snapshot()
        for nbytes in (-1, True, 1.5):
            with self.subTest(nbytes=nbytes), self.assertRaises(ValueError):
                request.reserve("owner", "source", nbytes)
        with self.assertRaises(host.HostBudgetExceeded):
            request.reserve("owner", "source", 2)
        self.assertEqual(budget.snapshot(), before)
        request.close()


if __name__ == "__main__":
    unittest.main()
