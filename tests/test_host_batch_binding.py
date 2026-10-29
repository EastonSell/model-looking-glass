"""CPU tests for a claim's durable one-batch ownership binding."""
import threading
import inspect
import sys
import unittest
from unittest.mock import patch

from looking_glass.host_budget import HostBudget, HostBusy, HostClaim, HostRequestCancelled, InvalidHostHandle


class HostBatchBindingTests(unittest.TestCase):
    def setUp(self):
        self.budget = HostBudget(max_bytes=1024, max_items=8, max_requests=3, max_operations=4)
        self.request = self.budget.open_request('group')
        self.claim = self.request.reserve('coordinator', 'batch-envelope', 256, items=2)

    def test_binding_survives_operation_transfer_and_cancellation(self):
        self.assertTrue(self.claim.bind_once('group:one'))
        self.assertFalse(self.claim.bind_once('group:one'))
        self.assertEqual(self.claim.info().binding, 'group:one')
        operation = self.request.begin_operation('work', (self.claim,))
        self.assertTrue(operation.finish())
        moved = self.claim.transfer('delivery')
        self.assertEqual(moved.info().binding, 'group:one')
        with self.assertRaises(InvalidHostHandle):
            self.claim.bind_once('group:one')
        with self.assertRaises(ValueError):
            moved.bind_once('group:two')
        self.request.cancel()
        self.assertEqual(moved.info().binding, 'group:one')
        with self.assertRaises(HostRequestCancelled):
            moved.bind_once('group:one')

    def test_cannot_first_bind_an_in_use_claim(self):
        operation = self.request.begin_operation('existing-use', (self.claim,))
        with self.assertRaises(HostBusy):
            self.claim.bind_once('group:one')
        self.assertIsNone(self.claim.info().binding)
        self.assertTrue(operation.finish())
        self.assertTrue(self.claim.bind_once('group:one'))

    def test_two_distinct_groups_cannot_bind_the_same_claim(self):
        barrier = threading.Barrier(3)
        results = []

        def bind(group):
            barrier.wait()
            try:
                results.append((group, self.claim.bind_once(group)))
            except ValueError:
                results.append((group, 'rejected'))

        threads = [threading.Thread(target=bind, args=(group,)) for group in ('group:one', 'group:two')]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sum(result is True for _, result in results), 1)
        self.assertEqual(sum(result == 'rejected' for _, result in results), 1)
        self.assertIn(self.claim.info().binding, ('group:one', 'group:two'))

    def test_failed_record_replacement_does_not_bind(self):
        with patch('looking_glass.host_budget.replace', side_effect=MemoryError('injected')):
            with self.assertRaises(MemoryError):
                self.claim.bind_once('group:one')
        self.assertIsNone(self.claim.info().binding)
        self.assertTrue(self.claim.bind_once('group:two'))

    def test_interrupt_at_return_keeps_lock_available_and_binding_recoverable(self):
        lines, first = inspect.getsourcelines(HostClaim.bind_once)
        return_line = first + next(i for i, line in enumerate(lines) if line.strip() == 'return bound')

        def trace(frame, event, arg):
            if frame.f_code is HostClaim.bind_once.__code__ and event == 'line' and frame.f_lineno == return_line:
                sys.settrace(None)
                raise KeyboardInterrupt('return interrupted')
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.claim.bind_once('group:one')
        finally:
            sys.settrace(None)
        finished = threading.Event()
        seen = []

        def inspect_claim():
            seen.append(self.claim.info().binding)
            finished.set()

        reader = threading.Thread(target=inspect_claim, daemon=True)
        reader.start()
        self.assertTrue(finished.wait(2), 'ledger lock was not released')
        reader.join(2)
        self.assertEqual(seen, ['group:one'])
        self.assertFalse(self.claim.bind_once('group:one'))


if __name__ == '__main__':
    unittest.main()
