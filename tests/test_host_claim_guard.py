"""Recovery of a host claim when its original handoff is interrupted."""

from concurrent.futures import ThreadPoolExecutor
import dis
import inspect
import sys
from threading import Barrier, Event, Thread
import unittest

from looking_glass.host_budget import (
    HostBudget, HostBudgetExceeded, HostBusy, HostClaimGuard, HostRequest,
    HostRequestCancelled, InvalidHostHandle,
)


class HostClaimGuardTests(unittest.TestCase):
    def setUp(self):
        self.budget = HostBudget(max_bytes=100, max_items=4, max_requests=3, max_operations=3)
        self.request_guard = self.budget.request_guard('job-1')
        self.request_guard.open()
        self.request = self.request_guard.adopt()

    def test_guard_registers_one_recoverable_claim_and_does_not_release_it(self):
        guard = self.request.claim_guard('owner', 'snapshot', 25, items=2)
        self.assertEqual(self.budget.active_claim_guards(), ())
        self.assertIsNone(guard.reserve())
        claim = guard.adopt()
        self.assertIs(guard.adopt(), claim)
        self.assertIsNone(guard.reserve())
        self.assertEqual(self.budget.active_claim_guards(), (guard,))
        self.assertEqual((claim.info().nbytes, claim.info().items), (25, 2))
        self.request.cancel()
        self.assertIs(guard.adopt(), claim)
        with self.assertRaises(HostBusy):
            self.request.close()
        self.assertTrue(claim.release())
        self.assertEqual(self.budget.active_claim_guards(), ())
        with self.assertRaises(InvalidHostHandle):
            guard.adopt()
        self.assertTrue(self.request.close())

    def test_return_event_interrupt_leaves_same_guard_adoptable_and_lock_free(self):
        guard = self.request.claim_guard('owner', 'snapshot', 25)
        previous = sys.gettrace()
        def trace(frame, event, arg):
            if event == 'return' and frame.f_code is type(guard).reserve.__code__:
                raise KeyboardInterrupt
            return trace
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                guard.reserve()
        finally:
            sys.settrace(previous)
        seen = []
        thread = Thread(target=lambda: seen.append(guard.adopt()), daemon=True)
        thread.start()
        thread.join(1)
        self.assertFalse(thread.is_alive(), 'ledger lock was stranded')
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(self.budget.active_claims()), 1)
        self.assertIs(guard.adopt(), seen[0])
        self.assertEqual(seen[0].info().request_generation, self.request_guard.generation)

    def test_adoption_return_interrupt_preserves_claim_for_another_thread(self):
        guard = self.request.claim_guard('owner', 'snapshot', 25)
        guard.reserve()
        previous = sys.gettrace()
        def trace(frame, event, arg):
            if event == 'return' and frame.f_code is type(guard).adopt.__code__:
                raise KeyboardInterrupt
            return trace
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                guard.adopt()
        finally:
            sys.settrace(previous)
        recovered = []
        def recover():
            self.request.cancel()
            recovered.append(guard.adopt())
        thread = Thread(target=recover, daemon=True)
        thread.start()
        thread.join(1)
        self.assertFalse(thread.is_alive(), 'ledger lock was stranded')
        self.assertEqual(recovered, list(self.budget.active_claims()))
        self.assertEqual(self.budget.snapshot().used_bytes, 25)

    def test_lock_exit_interrupt_is_detectable_and_claim_remains_charged(self):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 11):
            self.skipTest('opcode-level lock-exit probe is specific to CPython 3.11')
        guard = self.request.claim_guard('owner', 'snapshot', 25)
        instructions = list(dis.get_instructions(type(guard).reserve))
        exits = [instructions[i].offset for i in range(len(instructions) - 4)
                 if all(ins.opname == 'LOAD_CONST' and ins.argval is None
                        for ins in instructions[i:i + 3])
                 and instructions[i + 3].opname == 'PRECALL'
                 and instructions[i + 4].opname == 'CALL']
        self.assertTrue(exits, 'could not locate CPython normal with-exit sequence')
        exit_offset = exits[-1]
        interrupted, allow_test_cleanup = Event(), Event()
        def run():
            def trace(frame, event, arg):
                if frame.f_code is type(guard).reserve.__code__:
                    if event == 'call':
                        frame.f_trace_opcodes = True
                    elif event == 'opcode' and frame.f_lasti == exit_offset:
                        sys.settrace(None)
                        raise KeyboardInterrupt
                return trace
            try:
                sys.settrace(trace)
                guard.reserve()
            except KeyboardInterrupt:
                interrupted.set()
                allow_test_cleanup.wait(2)
            finally:
                sys.settrace(None)
                # Only this synthetic test knows it owns the RLock. Production
                # recovery uses a supervisor, not cross-thread lock release.
                if self.budget._lock._is_owned():
                    self.budget._lock.release()
        thread = Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(interrupted.wait(2), 'opcode interrupt did not reach the lock exit')
            acquired = self.budget._lock.acquire(timeout=0.1)
            if acquired:
                self.budget._lock.release()
        finally:
            allow_test_cleanup.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(acquired, 'lock-exit behavior changed; update this documented limitation')
        self.assertIsNotNone(guard.adopt())
        self.assertEqual(self.budget.snapshot().used_bytes, 25)

    def test_interrupt_at_each_reserve_stage_has_unambiguous_recovery(self):
        guard_type = type(self.request.claim_guard('unused', 'unused', 0))
        lines, first = inspect.getsourcelines(guard_type.reserve)
        fragments = ('self._handle = ', 'record = _ClaimRecord', 'budget._claims[claim_id] = ', 'return None')
        targets = [first + i for i, line in enumerate(lines)
                   if any(line.strip().startswith(fragment) for fragment in fragments)]
        self.assertGreaterEqual(len(targets), 4)
        for index, target in enumerate(targets):
            with self.subTest(line=target):
                budget = HostBudget(max_bytes=100, max_items=4, max_requests=2, max_operations=2)
                request = budget.open_request(f'job-{index}')
                guard = request.claim_guard('owner', 'snapshot', 25)
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if event == 'line' and frame.f_code is guard_type.reserve.__code__ and frame.f_lineno == target:
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        guard.reserve()
                finally:
                    sys.settrace(previous)
                claims = []
                def recover():
                    claims.extend(budget.active_claims())
                thread = Thread(target=recover, daemon=True)
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive(), 'ledger lock was stranded')
                self.assertLessEqual(len(claims), 1)
                if claims:
                    self.assertIs(guard.adopt(), claims[0])
                    self.assertEqual(budget.snapshot().used_bytes, 25)
                else:
                    self.assertEqual(budget.snapshot().used_bytes, 0)
                    with self.assertRaises(InvalidHostHandle):
                        guard.adopt()

    def test_transfer_retires_old_guard_without_releasing_live_storage(self):
        guard = self.request.claim_guard('owner', 'output', 30)
        guard.reserve()
        claim = guard.adopt()
        claim.bind_once('batch-purpose')
        transferred = claim.transfer('receiver')
        self.assertEqual(transferred.info().binding, 'batch-purpose')
        self.assertEqual(self.budget.active_claim_guards(), ())
        self.assertEqual(self.budget.snapshot().used_bytes, 30)
        with self.assertRaises(InvalidHostHandle):
            guard.adopt()
        with self.assertRaises(InvalidHostHandle):
            guard.reserve()
        self.assertTrue(transferred.release())

    def test_guard_cannot_attach_to_reused_request_generation(self):
        guard = self.request.claim_guard('owner', 'snapshot', 10)
        guard.reserve()
        guard.adopt().release()
        self.request.close()
        new_request = self.budget.open_request('job-1')
        new_guard = new_request.claim_guard('owner', 'snapshot', 10)
        new_guard.reserve()
        with self.assertRaises(InvalidHostHandle):
            guard.adopt()
        with self.assertRaises(InvalidHostHandle):
            guard.reserve()
        self.assertIsNot(guard, new_guard)

    def test_equal_metadata_in_separate_ledgers_is_not_a_capability(self):
        other = HostBudget(max_bytes=100, max_items=4, max_requests=3, max_operations=3)
        other_guard = other.request_guard('job-1')
        other_guard.open()
        other_request = other_guard.adopt()
        guards = [self.request.claim_guard('owner', 'snapshot', 10),
                  other_request.claim_guard('owner', 'snapshot', 10)]
        for guard in guards:
            guard.reserve()
        first, second = [guard.adopt() for guard in guards]
        self.assertEqual(first.info(), second.info())
        with self.assertRaises(InvalidHostHandle):
            self.request.begin_operation('copy', [second])
        with self.assertRaises(InvalidHostHandle):
            other_request.begin_operation('copy', [first])
        self.assertIs(self.budget.active_claim_guards()[0], guards[0])
        self.assertIs(other.active_claim_guards()[0], guards[1])
        self.assertTrue(first.release())
        self.assertEqual(other.snapshot().used_bytes, 10)
        self.assertIs(guards[1].adopt(), second)

    def test_concurrent_reserve_of_one_guard_cannot_duplicate_claim(self):
        guard = self.request.claim_guard('owner', 'snapshot', 40)
        barrier = Barrier(8)
        def reserve():
            barrier.wait()
            guard.reserve()
            return guard.adopt()
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(lambda _: reserve(), range(8)))
        self.assertTrue(all(claim is claims[0] for claim in claims))
        self.assertEqual(len(self.budget.active_claims()), 1)
        self.assertEqual(self.budget.snapshot().used_bytes, 40)

    def test_budget_and_cancel_failures_do_not_register_a_guard(self):
        too_big = self.request.claim_guard('owner', 'snapshot', 101)
        with self.assertRaises(HostBudgetExceeded):
            too_big.reserve()
        self.request.cancel()
        with self.assertRaises(HostRequestCancelled):
            self.request.claim_guard('owner', 'snapshot', 1).reserve()
        self.assertEqual(self.budget.active_claim_guards(), ())
        self.assertEqual(self.budget.snapshot().used_bytes, 0)

    def test_invalid_and_forged_inputs_never_register_a_claim(self):
        for args in (('', 's', 1, 1), ('o', '', 1, 1), ('o', 's', -1, 1),
                     ('o', 's', True, 1), ('o', 's', 1, 0), ('o', 's', 1, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.request.claim_guard(args[0], args[1], args[2], items=args[3])
        for request in (object(), HostRequest(None, 'forged'), HostRequest(self.budget, 'job-1')):
            with self.subTest(request=type(request)):
                if type(request) is HostRequest and request._budget is self.budget:
                    guard = HostClaimGuard(request, 'owner', 'stage', 1)
                    with self.assertRaises(InvalidHostHandle):
                        guard.reserve()
                else:
                    with self.assertRaises(ValueError):
                        HostClaimGuard(request, 'owner', 'stage', 1)
        self.assertEqual(self.budget.snapshot().used_bytes, 0)

    def test_stale_guard_cannot_adopt_after_release_and_new_claim(self):
        first = self.request.claim_guard('owner', 'stage', 10)
        first.reserve()
        old = first.adopt()
        old.release()
        second = self.request.claim_guard('owner', 'stage', 10)
        second.reserve()
        with self.assertRaises(InvalidHostHandle):
            first.adopt()
        self.assertIsNot(second.adopt(), old)
        self.assertEqual(self.budget.active_claim_guards(), (second,))


if __name__ == '__main__':
    unittest.main()
