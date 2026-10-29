import unittest
from queue import Full
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, Thread
from unittest.mock import patch

from looking_glass.broker import Broker
from looking_glass.fair_adapter import (
    AdapterClosed, ExperimentalFairAdapter, HANDOFF_PROTOCOL, HandoffFailure, FatalHandoffFailure,
    HandoffProtocolError,
)


def verify(credential):
    principals = {b'alice-key': 'alice', b'alice-alias-key': 'alice-alias', b'bob-key': 'bob'}
    if credential not in principals:
        raise PermissionError('credential rejected')
    return principals[credential]


class InMemoryHandoff:
    handoff_protocol = HANDOFF_PROTOCOL

    def __init__(self, teams=('A', 'B'), capacity=1):
        self._lock = Lock()
        self.teams = set(teams)
        self.capacity = capacity
        self.running = []

    def accept_next(self, claim):
        with self._lock:
            if len(self.running) >= self.capacity:
                return None
            request = claim(self.teams)
            if request is None:
                return None
            self.running.append(request)
            return request.ticket_id


class FairAdapterTests(unittest.TestCase):
    def make_adapter(self, *, target=None, **overrides):
        options = dict(max_waiters=4, max_waiters_per_team=3,
                       max_waiting_bytes=10, max_item_bytes=5)
        options.update(overrides)
        return ExperimentalFairAdapter(
            {'alice': 'A', 'alice-alias': 'A', 'bob': 'B'},
            verifier=verify, target=target or InMemoryHandoff(), **options,
        )

    def test_waiters_use_canonical_team_turns_and_handoff_owns_payload(self):
        target = InMemoryHandoff(capacity=3)
        adapter = ExperimentalFairAdapter(
            {'alice': 'A', 'alice-alias': 'A', 'bob': 'B'},
            verifier=verify, target=target, max_waiters=3,
            max_waiters_per_team=3, max_waiting_bytes=10, max_item_bytes=5,
        )
        first = adapter.submit(b'alice-key', b'a1')
        adapter.submit(b'alice-alias-key', b'a2')
        with self.assertRaises(Full):
            adapter.submit(b'alice-key', b'a3')
        adapter.submit(b'bob-key', b'b1')
        result = adapter.transfer_next()
        self.assertEqual(result.ticket, first)
        adapter.transfer_next()
        adapter.transfer_next()
        self.assertEqual([item.payload for item in target.running], [b'a1', b'b1', b'a2'])
        self.assertEqual(adapter.waiting_bytes, 0)

    def test_refuses_target_without_atomic_handoff_contract(self):
        with self.assertRaises(TypeError):
            self.make_adapter(target=object())

    def test_rejects_target_with_smaller_item_limit(self):
        target = InMemoryHandoff()
        target.max_item_bytes = 3
        with self.assertRaises(ValueError):
            self.make_adapter(target=target, max_item_bytes=5)

    def test_target_binding_receives_canonical_teams_and_limit_once(self):
        class BindingTarget(InMemoryHandoff):
            def __init__(self):
                super().__init__()
                self.binding = None

            def bind_waiting_queue(self, teams, max_item_bytes):
                if self.binding is not None:
                    raise RuntimeError('target already bound')
                self.binding = (teams, max_item_bytes)

        target = BindingTarget()
        self.make_adapter(target=target)
        self.assertEqual(target.binding, (frozenset({'A', 'B'}), 5))
        with self.assertRaises(RuntimeError):
            self.make_adapter(target=target)

    def test_invalid_adapter_does_not_bind_target(self):
        class BindingTarget(InMemoryHandoff):
            def __init__(self):
                super().__init__()
                self.bound = False

            def bind_waiting_queue(self, teams, max_item_bytes):
                self.bound = True

        target = BindingTarget()
        with self.assertRaises(ValueError):
            self.make_adapter(target=target, max_waiters=1)
        self.assertFalse(target.bound)

    def test_current_broker_is_rejected_as_an_atomic_target(self):
        class Cancelled(Exception):
            pass

        class IdleWorker:
            cancellation_exception = Cancelled
            poisoned = False

            def close(self):
                pass

        with Broker(IdleWorker(), max_pending=1, max_pending_per_team=1) as broker:
            with self.assertRaises(TypeError):
                self.make_adapter(target=broker)

    def test_requires_immutable_bytes_and_enforces_both_byte_limits(self):
        adapter = self.make_adapter(max_waiting_bytes=5, max_item_bytes=3)
        for invalid in (bytearray(b'a'), memoryview(b'a'), 'a'):
            with self.subTest(value=invalid), self.assertRaises(TypeError):
                adapter.submit(b'alice-key', invalid)
        with self.assertRaises(Full):
            adapter.submit(b'alice-key', b'four')
        adapter.submit(b'alice-key', b'123')
        with self.assertRaises(Full):
            adapter.submit(b'bob-key', b'456')
        self.assertEqual(adapter.waiting_bytes, 3)
        self.assertEqual(len(adapter), 1)

    def test_byte_limit_configuration_is_validated(self):
        for options in ({'max_waiting_bytes': 0}, {'max_item_bytes': True},
                        {'max_item_bytes': 1.5},
                        {'max_waiting_bytes': 2, 'max_item_bytes': 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_adapter(**options)

    def test_owner_only_cancellation_releases_bytes_and_capacity(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'12345')
        with self.assertRaises(PermissionError):
            adapter.cancel(b'bob-key', ticket)
        with self.assertRaises(PermissionError):
            adapter.cancel(b'wrong-key', ticket)
        self.assertEqual(adapter.waiting_bytes, 5)
        self.assertTrue(adapter.cancel(b'alice-key', ticket))
        self.assertFalse(adapter.cancel(b'alice-key', ticket))
        self.assertEqual(adapter.waiting_bytes, 0)
        self.assertEqual(len(adapter), 0)

    def test_unconfigured_verified_principal_is_rejected_before_byte_backpressure(self):
        adapter = ExperimentalFairAdapter(
            {'alice': 'A'}, verifier=lambda token: 'alice' if token == b'ok' else 'unknown',
            target=InMemoryHandoff(teams=('A',)), max_waiters=2,
            max_waiters_per_team=2, max_waiting_bytes=1, max_item_bytes=1,
        )
        adapter.submit(b'ok', b'x')
        with self.assertRaises(PermissionError):
            adapter.submit(b'unknown', b'x')

    def test_nonstring_verifier_result_cannot_impersonate_ticket_owner(self):
        class PretendOwner:
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        adapter = ExperimentalFairAdapter(
            {'alice': 'A'}, verifier=lambda token: 'alice' if token == b'ok' else PretendOwner(),
            target=InMemoryHandoff(teams=('A',)), max_waiters=1,
            max_waiters_per_team=1, max_waiting_bytes=4, max_item_bytes=4,
        )
        ticket = adapter.submit(b'ok', b'data')
        with self.assertRaises(PermissionError):
            adapter.cancel(b'bad', ticket)
        self.assertTrue(adapter.cancel(b'ok', ticket))

    def test_submit_rolls_back_if_its_own_bookkeeping_cannot_allocate(self):
        class FailingDict(dict):
            def __setitem__(self, key, value):
                raise MemoryError('deliberate bookkeeping failure')

        for field in ('_owners', '_sizes', '_tickets'):
            with self.subTest(field=field):
                adapter = ExperimentalFairAdapter(
                    {'alice': 'A'}, verifier=verify, target=InMemoryHandoff(teams=('A',)),
                    max_waiters=1, max_waiters_per_team=1, max_waiting_bytes=4, max_item_bytes=4,
                )
                setattr(adapter, field, FailingDict())
                with self.assertRaises(MemoryError):
                    adapter.submit(b'alice-key', b'data')
                self.assertEqual(adapter.waiting_bytes, 0)
                setattr(adapter, field, {})
                ticket = adapter.submit(b'alice-key', b'data')
                self.assertTrue(adapter.cancel(b'alice-key', ticket))

    def test_claim_does_not_allocate_a_new_ticket_after_removing_waiter(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'data')
        with patch('looking_glass.fair_adapter.WaitingTicket', side_effect=MemoryError):
            receipt = adapter.transfer_next()
        self.assertEqual(receipt.ticket, ticket)

    def test_receipt_allocation_failure_reports_claimed_ticket(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'data')
        with patch('looking_glass.fair_adapter.TransferReceipt', side_effect=MemoryError):
            with self.assertRaises(HandoffFailure) as caught:
                adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)

    def test_claim_and_cancel_do_not_do_byte_arithmetic_after_dequeue(self):
        class FailingSize(int):
            def __rsub__(self, other):
                raise MemoryError('deliberate accounting failure')

        for action in ('claim', 'cancel'):
            with self.subTest(action=action):
                adapter = self.make_adapter()
                ticket = adapter.submit(b'alice-key', b'data')
                adapter._sizes[ticket.sequence] = FailingSize(4)
                if action == 'claim':
                    self.assertEqual(adapter.transfer_next().ticket, ticket)
                else:
                    self.assertTrue(adapter.cancel(b'alice-key', ticket))
                self.assertEqual(len(adapter), 0)
                self.assertEqual(adapter.waiting_bytes, 0)

    def test_stale_ticket_from_another_adapter_cannot_cancel(self):
        first = self.make_adapter()
        second = self.make_adapter()
        old = first.submit(b'alice-key', b'old')
        current = second.submit(b'alice-key', b'new')
        self.assertFalse(second.cancel(b'alice-key', old))
        self.assertTrue(second.cancel(b'alice-key', current))

    def test_full_target_keeps_waiter_and_bytes_pending(self):
        target = InMemoryHandoff(capacity=1)
        adapter = self.make_adapter(target=target)
        adapter.submit(b'alice-key', b'a')
        second = adapter.submit(b'bob-key', b'bb')
        adapter.transfer_next()
        self.assertIsNone(adapter.transfer_next())
        self.assertEqual(adapter.waiting_bytes, 2)
        self.assertTrue(adapter.cancel(b'bob-key', second))

    def test_handoff_failure_reports_the_claimed_ticket(self):
        class Broken(InMemoryHandoff):
            def accept_next(self, claim):
                claim(self.teams)
                raise RuntimeError('failed after claim')

        adapter = self.make_adapter(target=Broken())
        ticket = adapter.submit(b'alice-key', b'data')
        with self.assertRaises(HandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)
        self.assertEqual(adapter.waiting_bytes, 0)
        self.assertFalse(adapter.cancel(b'alice-key', ticket))

    def test_fatal_handoff_failure_still_identifies_claimed_ticket(self):
        class Fatal(BaseException):
            pass

        class Broken(InMemoryHandoff):
            def accept_next(self, claim):
                claim(self.teams)
                raise Fatal()

        adapter = self.make_adapter(target=Broken())
        ticket = adapter.submit(b'alice-key', b'data')
        with self.assertRaises(FatalHandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)
        self.assertEqual(adapter.waiting_bytes, 0)

    def test_handoff_must_return_a_receipt_after_claim(self):
        class MissingReceipt(InMemoryHandoff):
            def accept_next(self, claim):
                claim(self.teams)
                return None

        adapter = self.make_adapter(target=MissingReceipt())
        ticket = adapter.submit(b'alice-key', b'data')
        with self.assertRaises(HandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, ticket)

    def test_callback_cannot_run_after_handoff_returns(self):
        class Deferred(InMemoryHandoff):
            def accept_next(self, claim):
                self.saved = claim
                return None

        target = Deferred()
        adapter = self.make_adapter(target=target)
        ticket = adapter.submit(b'alice-key', b'data')
        self.assertIsNone(adapter.transfer_next())
        with self.assertRaises(HandoffProtocolError):
            target.saved({'A'})
        self.assertTrue(adapter.cancel(b'alice-key', ticket))

    def test_callback_must_run_on_the_handoff_calling_thread(self):
        class WrongThread(InMemoryHandoff):
            def accept_next(self, claim):
                errors = []

                def call():
                    try:
                        claim(self.teams)
                    except BaseException as exc:
                        errors.append(exc)

                worker = Thread(target=call, daemon=True)
                worker.start()
                worker.join(timeout=1)
                if worker.is_alive():
                    raise RuntimeError('test timed out')
                if errors:
                    raise errors[0]
                return 'claimed'

        adapter = self.make_adapter(target=WrongThread())
        ticket = adapter.submit(b'alice-key', b'data')
        with self.assertRaises(HandoffProtocolError):
            adapter.transfer_next()
        self.assertTrue(adapter.cancel(b'alice-key', ticket))

    def test_callback_cannot_claim_two_requests(self):
        class DoubleClaim(InMemoryHandoff):
            def accept_next(self, claim):
                claim(self.teams)
                claim(self.teams)
                return 'claimed'

        adapter = self.make_adapter(target=DoubleClaim())
        first = adapter.submit(b'alice-key', b'a')
        second = adapter.submit(b'bob-key', b'b')
        with self.assertRaises(HandoffFailure) as caught:
            adapter.transfer_next()
        self.assertEqual(caught.exception.ticket, first)
        self.assertTrue(adapter.cancel(b'bob-key', second))

    def test_close_cancels_waiting_tickets_and_rejects_new_work(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'a')
        self.assertEqual(adapter.close(), (ticket,))
        self.assertEqual(adapter.close(), ())
        self.assertFalse(adapter.cancel(b'alice-key', ticket))
        self.assertIsNone(adapter.transfer_next())
        self.assertEqual(adapter.waiting_bytes, 0)
        with self.assertRaises(AdapterClosed):
            adapter.submit(b'alice-key', b'next')

    def test_close_failure_before_atomic_drain_keeps_waiters_retryable(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'data')
        with patch.object(adapter._queue, 'drain', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                adapter.close()
        self.assertEqual(len(adapter), 1)
        self.assertEqual(adapter.waiting_bytes, 4)
        self.assertEqual(adapter.close(), (ticket,))

    def test_interrupted_claim_closes_adapter_for_owner_reconciliation(self):
        adapter = self.make_adapter()
        ticket = adapter.submit(b'alice-key', b'data')
        original = adapter._queue.claim_next

        def remove_then_interrupt(eligible):
            original(eligible)
            raise KeyboardInterrupt()

        with patch.object(adapter._queue, 'claim_next', side_effect=remove_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                adapter.transfer_next()
        with self.assertRaises(AdapterClosed):
            adapter.submit(b'alice-key', b'new')
        with self.assertRaises(AdapterClosed):
            adapter.cancel(b'alice-key', ticket)
        with self.assertRaises(AdapterClosed):
            adapter.transfer_next()
        self.assertEqual(adapter.close(), (ticket,))
        self.assertFalse(adapter.cancel(b'alice-key', ticket))

    def test_cancel_and_transfer_race_has_one_winner(self):
        for _ in range(20):
            target = InMemoryHandoff()
            adapter = self.make_adapter(target=target)
            ticket = adapter.submit(b'alice-key', b'a')
            barrier = Barrier(2)

            def cancel():
                barrier.wait()
                return adapter.cancel(b'alice-key', ticket)

            def transfer():
                barrier.wait()
                return adapter.transfer_next()

            with ThreadPoolExecutor(max_workers=2) as workers:
                c = workers.submit(cancel)
                t = workers.submit(transfer)
                canceled, transferred = c.result(), t.result()
            self.assertEqual(canceled, transferred is None)
            if transferred is not None:
                self.assertEqual(transferred.ticket, ticket)
            self.assertEqual(adapter.waiting_bytes, 0)

    def test_close_wins_if_handoff_has_not_claimed_yet(self):
        class Delayed(InMemoryHandoff):
            def __init__(self):
                super().__init__()
                self.entered = Event()
                self.resume = Event()

            def accept_next(self, claim):
                self.entered.set()
                if not self.resume.wait(1):
                    raise RuntimeError('test timed out')
                return super().accept_next(claim)

        target = Delayed()
        adapter = self.make_adapter(target=target)
        ticket = adapter.submit(b'alice-key', b'a')
        results = []
        thread = Thread(target=lambda: results.append(adapter.transfer_next()), daemon=True)
        thread.start()
        self.assertTrue(target.entered.wait(1))
        self.assertEqual(adapter.close(), (ticket,))
        target.resume.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [None])
        self.assertEqual(target.running, [])


if __name__ == '__main__':
    unittest.main()
