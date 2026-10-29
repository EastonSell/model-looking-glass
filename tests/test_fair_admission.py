import unittest
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full
from threading import Barrier, Event, Thread

from looking_glass.fair_admission import FairAdmissionQueue


class FairAdmissionTests(unittest.TestCase):
    def test_other_team_keeps_one_waiter_slot_under_saturation(self):
        admission = FairAdmissionQueue(
            {'alice': 'analysis', 'bob': 'review'},
            max_waiters=3,
            max_waiters_per_team=3,
        )
        admission.submit('alice', 'first')
        admission.submit('alice', 'second')
        with self.assertRaises(Full):
            admission.submit('alice', 'third')
        admission.submit('bob', 'review')
        self.assertEqual(len(admission), 3)
        self.assertEqual([admission.claim_next().payload for _ in range(3)],
                         ['first', 'review', 'second'])

    def test_canceling_a_teams_last_waiter_restores_its_protected_slot(self):
        admission = FairAdmissionQueue({'a': 'A', 'b': 'B'}, 3, 3)
        admission.submit('a', 'a1')
        admission.submit('a', 'a2')
        b1 = admission.submit('b', 'b1')
        self.assertTrue(admission.cancel(b1))
        with self.assertRaises(Full):
            admission.submit('a', 'a3')
        admission.submit('b', 'b2')
        self.assertEqual([admission.claim_next().payload for _ in range(3)],
                         ['a1', 'b2', 'a2'])

    def test_aliases_share_a_quota_and_unknown_principals_are_rejected(self):
        mapping = {'alice': 'analysis', 'alice-alias': 'analysis', 'bob': 'review'}
        admission = FairAdmissionQueue(mapping, max_waiters=4, max_waiters_per_team=2)
        mapping['invented'] = 'analysis'
        admission.submit('alice', 'first')
        admission.submit('alice-alias', 'second')
        with self.assertRaises(Full):
            admission.submit('alice-alias', 'third')
        with self.assertRaises(PermissionError):
            admission.submit('invented', 'not trusted')
        with self.assertRaises(PermissionError):
            admission.submit('unknown', 'not trusted')
        admission.submit('bob', 'review')
        self.assertEqual([admission.claim_next().payload for _ in range(3)],
                         ['first', 'review', 'second'])

    def test_skipped_team_retains_its_position_when_eligible_again(self):
        admission = FairAdmissionQueue({'a': 'A', 'b': 'B', 'c': 'C'}, 6, 2)
        for principal, value in [('a', 'a1'), ('a', 'a2'), ('b', 'b1'),
                                 ('b', 'b2'), ('c', 'c1')]:
            admission.submit(principal, value)
        self.assertEqual(admission.claim_next({'B', 'C'}).payload, 'b1')
        self.assertEqual(admission.claim_next({'A', 'C'}).payload, 'a1')
        self.assertEqual([admission.claim_next().payload for _ in range(3)],
                         ['c1', 'b2', 'a2'])

    def test_no_eligible_team_does_not_remove_a_waiter(self):
        admission = FairAdmissionQueue({'a': 'A', 'b': 'B'}, 2, 1)
        admission.submit('a', 'a1')
        with self.assertRaises(Empty):
            admission.claim_next({'B'})
        self.assertEqual(len(admission), 1)
        self.assertEqual(admission.claim_next({'A'}).payload, 'a1')

    def test_cancel_releases_a_slot_and_never_reuses_a_ticket(self):
        admission = FairAdmissionQueue({'a': 'A'}, 2, 2)
        first = admission.submit('a', 'first')
        self.assertTrue(admission.cancel(first))
        self.assertFalse(admission.cancel(first))
        second = admission.submit('a', 'second')
        self.assertNotEqual(first, second)
        self.assertFalse(admission.cancel(first))
        self.assertEqual(admission.claim_next().payload, 'second')
        self.assertFalse(admission.cancel(second))

    def test_boolean_cancellation_cannot_target_integer_ticket(self):
        admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
        admission.submit('a', 'first')
        with self.assertRaises(ValueError):
            admission.cancel(True)
        self.assertEqual(admission.claim_next().payload, 'first')

    def test_invalid_configuration_is_rejected(self):
        bad = [({}, 1, 1), ({'a': 'A', 'b': 'B'}, 1, 1),
               ({'a': 'A'}, 0, 1), ({'a': 'A'}, 1, 0),
               ({'a': 'A'}, True, 1), ({'a': 'A'}, 1, True),
               ({'a': 'A'}, 1.5, 1), ({'a': ''}, 1, 1),
               ({'': 'A'}, 1, 1), ({1: 'A'}, 1, 1), ({'a': 1}, 1, 1)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                FairAdmissionQueue(*args)

    def test_constructor_validates_the_snapshot_it_will_use(self):
        class MisleadingMapping(dict):
            def items(self):
                return [('a', 'A')]

        with self.assertRaises(ValueError):
            FairAdmissionQueue(MisleadingMapping({'a': 5}), 1, 1)

    def test_failed_internal_ticket_index_update_rolls_back_admission(self):
        class FailingIndex(dict):
            def __setitem__(self, key, value):
                raise MemoryError('deliberate index allocation failure')

        admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
        admission._ticket_teams = FailingIndex()
        with self.assertRaises(MemoryError):
            admission.submit('a', 'first')
        self.assertEqual(len(admission), 0)
        admission._ticket_teams = {}
        admission.submit('a', 'second')
        self.assertEqual(admission.claim_next().payload, 'second')

    def test_failed_rotation_allocation_does_not_consume_a_waiter(self):
        class FailingRotation(deque):
            def append(self, value):
                raise MemoryError('deliberate rotation allocation failure')

        admission = FairAdmissionQueue({'a': 'A'}, 2, 2)
        admission.submit('a', 'first')
        admission.submit('a', 'second')
        admission._rotation = FailingRotation(admission._rotation)
        with self.assertRaises(MemoryError):
            admission.claim_next()
        self.assertEqual(len(admission), 2)
        admission._rotation = deque(admission._rotation)
        self.assertEqual([admission.claim_next().payload for _ in range(2)],
                         ['first', 'second'])

    def test_claim_survives_popitem_allocating_after_removal(self):
        class UnsafePopItem(OrderedDict):
            def popitem(self, last=True):
                super().popitem(last=last)
                raise MemoryError('tuple allocation after removal')

        admission = FairAdmissionQueue({'a': 'A'}, 2, 2)
        admission.submit('a', 'first')
        admission.submit('a', 'second')
        admission._teams['A'] = UnsafePopItem(admission._teams['A'])
        self.assertEqual([admission.claim_next().payload for _ in range(2)],
                         ['first', 'second'])

    def test_claim_and_cancel_keep_state_if_team_delete_fails(self):
        class FailingDelete(OrderedDict):
            def __delitem__(self, key):
                raise MemoryError('deliberate pre-delete failure')

            def pop(self, key, *args):
                raise MemoryError('deliberate pre-delete failure')

        for operation in ('claim', 'cancel'):
            with self.subTest(operation=operation):
                admission = FairAdmissionQueue({'a': 'A'}, 2, 2)
                ticket = admission.submit('a', 'first')
                admission.submit('a', 'second')
                admission._teams['A'] = FailingDelete(admission._teams['A'])
                with self.assertRaises(MemoryError):
                    admission.claim_next() if operation == 'claim' else admission.cancel(ticket)
                admission._teams['A'] = OrderedDict(admission._teams['A'])
                self.assertEqual([admission.claim_next().payload for _ in range(2)],
                                 ['first', 'second'])
                with self.assertRaises(Empty):
                    admission.claim_next()

    def test_cancel_size_allocation_failure_does_not_remove_a_waiter(self):
        class FailingCount(int):
            def __sub__(self, other):
                raise MemoryError('deliberate accounting failure')

        admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
        ticket = admission.submit('a', 'first')
        admission._size = FailingCount(1)
        with self.assertRaises(MemoryError):
            admission.cancel(ticket)
        admission._size = 1
        self.assertEqual(admission.claim_next().payload, 'first')

    def test_drain_returns_and_removes_every_waiter(self):
        admission = FairAdmissionQueue({'a': 'A', 'b': 'B'}, 3, 2)
        first = admission.submit('a', 'first')
        second = admission.submit('b', 'second')
        drained = admission.drain()
        self.assertEqual({item.ticket_id for item in drained}, {first, second})
        self.assertEqual(len(admission), 0)
        with self.assertRaises(Empty):
            admission.claim_next()

    def test_unknown_eligible_team_is_rejected_without_changing_queue(self):
        admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
        admission.submit('a', 'first')
        for eligible in ({'unknown'}, 'A', {1}):
            with self.subTest(eligible=eligible), self.assertRaises(ValueError):
                admission.claim_next(eligible)
        self.assertEqual(admission.claim_next().payload, 'first')

    def test_concurrent_alias_submissions_cannot_consume_another_team_slot(self):
        principals = {f'alice-{i}': 'A' for i in range(32)}
        principals['bob'] = 'B'
        admission = FairAdmissionQueue(principals, 8, 8)
        barrier = Barrier(32)

        def submit(i):
            barrier.wait()
            try:
                admission.submit(f'alice-{i}', i)
                return True
            except Full:
                return False

        with ThreadPoolExecutor(max_workers=32) as workers:
            accepted = list(workers.map(submit, range(32)))
        self.assertEqual(sum(accepted), 7)
        admission.submit('bob', 'review')
        self.assertEqual(len(admission), 8)
        self.assertEqual(admission.claim_next().team_id, 'A')
        self.assertEqual(admission.claim_next().team_id, 'B')

    def test_cancel_and_claim_race_has_exactly_one_winner(self):
        for _ in range(20):
            admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
            ticket = admission.submit('a', 'first')
            barrier = Barrier(2)

            def claim():
                barrier.wait()
                try:
                    return admission.claim_next().payload
                except Empty:
                    return None

            def cancel():
                barrier.wait()
                return admission.cancel(ticket)

            with ThreadPoolExecutor(max_workers=2) as workers:
                claimed = workers.submit(claim)
                canceled = workers.submit(cancel)
                self.assertIn((claimed.result(), canceled.result()),
                              [('first', False), (None, True)])
            self.assertEqual(len(admission), 0)

    def test_cancel_payload_finalizer_can_reenter_without_deadlock(self):
        admission = FairAdmissionQueue({'a': 'A'}, 1, 1)
        released = Event()

        class Payload:
            def __del__(self):
                len(admission)
                released.set()

        ticket = admission.submit('a', Payload())
        worker = Thread(target=admission.cancel, args=(ticket,), daemon=True)
        worker.start()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), 'payload finalizer deadlocked on queue lock')
        self.assertTrue(released.wait(timeout=1))


if __name__ == '__main__':
    unittest.main()
