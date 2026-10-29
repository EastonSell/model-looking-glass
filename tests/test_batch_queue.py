"""Independent-facing behavior of atomic fair cohort selection."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import inspect
from queue import Empty, Full
import random
import sys
from threading import Barrier, Event, Thread
import unittest

from looking_glass.fair_queue import FairTeamQueue


KEY = b'a' * 32
OTHER = b'b' * 32


class AtomicCohortTests(unittest.TestCase):
    def test_prefix_preserves_round_robin_and_per_team_fifo(self):
        q = FairTeamQueue(8, 3)
        for team, job, key in [('a', 'a1', KEY), ('a', 'a2', KEY),
                               ('b', 'b1', KEY), ('b', 'b2', OTHER), ('c', 'c1', KEY)]:
            q.put(team, job, object(), compatibility_key=key)
        self.assertEqual(q.peek_compatibility(), KEY)
        group = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2, 3), max_per_team=1)
        self.assertEqual([item.job_id for item in group], ['a1', 'b1', 'c1'])
        self.assertEqual([q.pop().job_id for _ in range(2)], ['a2', 'b2'])

    def test_incompatible_head_is_not_bypassed_and_unapproved_prefix_is_untouched(self):
        q = FairTeamQueue(4, 2)
        q.put('a', 'a1', 1, compatibility_key=KEY)
        q.put('b', 'b1', 2, compatibility_key=OTHER)
        q.put('c', 'c1', 3, compatibility_key=KEY)
        self.assertEqual(q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2, 3), max_per_team=1), ())
        self.assertEqual([q.pop().job_id for _ in range(3)], ['a1', 'b1', 'c1'])

    def test_longest_approved_prefix_and_next_dispatch_are_deterministic(self):
        q = FairTeamQueue(7, 3)
        for team, job in [('a', 'a1'), ('a', 'a2'), ('a', 'a3'), ('b', 'b1'), ('c', 'c1')]:
            q.put(team, job, job, compatibility_key=KEY)
        group = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2, 4), max_per_team=2)
        self.assertEqual([item.job_id for item in group], ['a1', 'b1', 'c1', 'a2'])
        self.assertEqual(q.pop().job_id, 'a3')

    def test_team_cap_stops_at_the_head_instead_of_bypassing_it(self):
        q = FairTeamQueue(5, 4)
        for team, job in [('a', 'a1'), ('a', 'a2'), ('a', 'a3'), ('b', 'b1')]:
            q.put(team, job, job, compatibility_key=KEY)
        group = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2, 3), max_per_team=1)
        self.assertEqual([item.job_id for item in group], ['a1', 'b1'])
        self.assertEqual([q.pop().job_id for _ in range(2)], ['a2', 'a3'])

    def test_nonbatchable_and_stale_head_are_not_claimed(self):
        q = FairTeamQueue(3, 2)
        q.put('a', 'a1', None)
        q.put('b', 'b1', None, compatibility_key=KEY)
        self.assertIsNone(q.peek_compatibility())
        self.assertEqual(q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1, 2), max_per_team=1), ())
        self.assertEqual(q.pop().job_id, 'a1')
        self.assertEqual(q.peek_compatibility(), KEY)
        self.assertEqual(q.pop_compatible(OTHER, claim_id='claim', allowed_sizes=(1, 2), max_per_team=1), ())
        self.assertEqual(q.pop().job_id, 'b1')
        with self.assertRaises(Empty):
            q.peek_compatibility()
        with self.assertRaises(Empty):
            q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1,), max_per_team=1)

    def test_rejects_untrusted_or_unbounded_compatibility_inputs(self):
        q = FairTeamQueue(4, 2)
        class BytesSubclass(bytes):
            pass
        for bad in (b'short', bytearray(KEY), BytesSubclass(KEY), 'a' * 32):
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                q.put('a', 'x', None, compatibility_key=bad)
        q.put('a', 'x', None, compatibility_key=KEY)
        for sizes in ((), [1, 2], (0,), (True,), (2, 1), (1, 1), (5,), (1, 2.0)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                q.pop_compatible(KEY, claim_id='claim', allowed_sizes=sizes, max_per_team=1)
        for value in (0, -1, True, 1.5, 5):
            with self.subTest(cap=value), self.assertRaises(ValueError):
                q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1,), max_per_team=value)
        with self.assertRaises(ValueError):
            q.pop_compatible(BytesSubclass(KEY), claim_id='claim', allowed_sizes=(1,), max_per_team=1)
        self.assertEqual(q.pop().job_id, 'x')

    def test_failed_shadow_allocation_does_not_change_queue(self):
        q = FairTeamQueue(4, 2)
        q.put('a', 'a1', 1, compatibility_key=KEY)
        q.put('b', 'b1', 2, compatibility_key=KEY)
        class FailingCopy(dict):
            def copy(self):
                raise MemoryError('shadow allocation failed')
        q._job_teams = FailingCopy(q._job_teams)
        with self.assertRaises(MemoryError):
            q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2,), max_per_team=1)
        q._job_teams = dict(q._job_teams)
        self.assertEqual(len(q), 2)
        self.assertEqual([q.pop().job_id for _ in range(2)], ['a1', 'b1'])

    def test_interrupt_during_commit_leaves_queue_unchanged(self):
        lines, first_line = inspect.getsourcelines(FairTeamQueue._pop_compatible_locked)
        locations = [first_line + i for i, line in enumerate(lines)
                     if line.strip().startswith(('self._teams = ', 'self._rotation = ',
                                                 'self._job_teams = ', 'self._size = ',
                                                 'self._claims = ', 'self._claimed_jobs = ',
                                                 'self._claimed_size = '))]
        self.assertGreaterEqual(len(locations), 7)
        for target in locations:
            with self.subTest(line=target):
                q = FairTeamQueue(4, 2)
                q.put('a', 'a1', 1, compatibility_key=KEY)
                q.put('b', 'b1', 2, compatibility_key=KEY)
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if event == 'line' and frame.f_code is FairTeamQueue._pop_compatible_locked.__code__ and frame.f_lineno == target:
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2,), max_per_team=1)
                finally:
                    sys.settrace(previous)
                self.assertEqual(len(q), 2)
                self.assertIsNone(q.get_claim('claim'))
                self.assertEqual([q.pop().job_id for _ in range(2)], ['a1', 'b1'])

    def test_cancel_and_claim_are_atomic(self):
        for _ in range(40):
            q = FairTeamQueue(2, 1)
            q.put('a', 'a1', 1, compatibility_key=KEY)
            q.put('b', 'b1', 2, compatibility_key=KEY)
            barrier = Barrier(2)
            def claim():
                barrier.wait()
                return tuple(item.job_id for item in q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1, 2), max_per_team=1))
            def cancel():
                barrier.wait()
                return q.cancel('b1')
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(claim)
                b = pool.submit(cancel)
                self.assertIn((a.result(), b.result()), [(('a1', 'b1'), False), (('a1',), True)])
            self.assertEqual(len(q), 0)

    def test_payload_is_not_consulted_and_finalizer_runs_after_unlock(self):
        q = FairTeamQueue(2, 1)
        released = Event()
        class Payload:
            def __getattribute__(self, name):
                if name not in ('__class__', '__del__'):
                    raise AssertionError('selection inspected a payload')
                return object.__getattribute__(self, name)
            def __del__(self):
                len(q)
                released.set()
        q.put('a', 'a1', Payload(), compatibility_key=KEY)
        def consume():
            result = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1,), max_per_team=1)
            self.assertTrue(q.acknowledge_claim('claim'))
            del result
        thread = Thread(target=consume, daemon=True)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive(), 'payload finalizer deadlocked on queue lock')
        self.assertTrue(released.wait(timeout=1))

    def test_random_cohorts_match_independent_round_robin_reference(self):
        rng = random.Random(29117)
        for _ in range(300):
            q = FairTeamQueue(12, 5)
            pending, rotation = {}, deque()
            jobs = list(range(rng.randint(1, 12)))
            for job in jobs:
                eligible = [team for team in 'abcd' if len(pending.get(team, ())) < 5]
                team = rng.choice(eligible)
                key = rng.choice((KEY, OTHER, None))
                q.put(team, job, job, compatibility_key=key)
                if team not in pending:
                    pending[team] = deque()
                    rotation.append(team)
                pending[team].append((job, key))
            cap = rng.randint(1, 3)
            sizes = tuple(sorted(rng.sample(range(1, 6), rng.randint(1, 5))))
            simulated = {team: deque(items) for team, items in pending.items()}
            turns = deque(rotation)
            taken, counts = [], {}
            while turns and len(taken) < sizes[-1]:
                team = turns[0]
                if counts.get(team, 0) == cap or simulated[team][0][1] != KEY:
                    break
                job, _ = simulated[team].popleft()
                taken.append((team, job))
                counts[team] = counts.get(team, 0) + 1
                turns.popleft()
                if simulated[team]:
                    turns.append(team)
            count = max((n for n in sizes if n <= len(taken)), default=0)
            expected = taken[:count]
            actual = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=sizes, max_per_team=cap)
            self.assertEqual([(item.team_id, item.job_id) for item in actual], expected)
            for team, job in expected:
                self.assertEqual(rotation.popleft(), team)
                self.assertEqual(pending[team].popleft()[0], job)
                if pending[team]:
                    rotation.append(team)
                else:
                    del pending[team]
            remainder = []
            while rotation:
                team = rotation.popleft()
                job, _ = pending[team].popleft()
                remainder.append((team, job))
                if pending[team]:
                    rotation.append(team)
            self.assertEqual([(item.team_id, item.job_id) for item in (q.pop() for _ in remainder)], remainder)

    def test_return_event_interrupt_retains_recoverable_claim(self):
        for method in (FairTeamQueue._pop_compatible_locked, FairTeamQueue.pop_compatible):
            lines, first = inspect.getsourcelines(method)
            return_line = next(first + i for i, line in enumerate(lines)
                               if line.strip() == 'return chosen')
            for event_kind in ('line', 'return'):
                with self.subTest(method=method.__name__, event=event_kind):
                    q = FairTeamQueue(3, 1)
                    q.put('a', 'a1', 1, compatibility_key=KEY)
                    q.put('b', 'b1', 2, compatibility_key=KEY)
                    q.put('c', 'c1', 3, compatibility_key=OTHER)
                    previous = sys.gettrace()
                    def trace(frame, event, arg):
                        if (frame.f_code is method.__code__ and event == event_kind
                                and (event == 'return' and arg is not None
                                     or event == 'line' and frame.f_lineno == return_line)):
                            raise KeyboardInterrupt
                        return trace
                    try:
                        sys.settrace(trace)
                        with self.assertRaises(KeyboardInterrupt):
                            q.pop_compatible(KEY, claim_id='broker-token', allowed_sizes=(2,), max_per_team=1)
                    finally:
                        sys.settrace(previous)
                    recovered_result = []
                    def recover():
                        recovered_result.append(q.get_claim('broker-token'))
                        recovered_result.append(q.cancel('c1'))
                    thread = Thread(target=recover, daemon=True)
                    thread.start()
                    thread.join(timeout=1)
                    self.assertFalse(thread.is_alive(), 'queue lock remained held after claim interrupt')
                    recovered, cancelled = recovered_result
                    self.assertTrue(cancelled)
                    self.assertEqual(len(q), 0)
                    self.assertEqual([item.job_id for item in recovered], ['a1', 'b1'])
                    self.assertEqual(q.get_claim('broker-token'), recovered)
                    self.assertTrue(q.acknowledge_claim('broker-token'))
                    self.assertIsNone(q.get_claim('broker-token'))
                    self.assertFalse(q.acknowledge_claim('broker-token'))

    def test_claim_ledger_is_bounded_and_duplicate_tokens_cannot_replace_claims(self):
        q = FairTeamQueue(2, 1)
        for team in 'ab':
            q.put(team, team + '1', None, compatibility_key=KEY)
        claim = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2,), max_per_team=1)
        for team in 'ab':
            q.put(team, team + '2', None, compatibility_key=KEY)
        with self.assertRaises(ValueError):
            q.put('a', 'a1', None, compatibility_key=KEY)
        with self.assertRaises(ValueError):
            q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(2,), max_per_team=1)
        with self.assertRaises(Full):
            q.pop_compatible(KEY, claim_id='claim2', allowed_sizes=(2,), max_per_team=1)
        self.assertEqual(q.get_claim('claim'), claim)
        self.assertEqual(len(q), 2)
        self.assertTrue(q.acknowledge_claim('claim'))
        second = q.pop_compatible(KEY, claim_id='claim2', allowed_sizes=(2,), max_per_team=1)
        self.assertEqual([item.job_id for item in second], ['a2', 'b2'])

    def test_invalid_claim_tokens_fail_without_consuming_work(self):
        q = FairTeamQueue(2, 1)
        q.put('a', 'a1', None, compatibility_key=KEY)
        class StringSubclass(str):
            pass
        for invalid in ('', 'x' * 129, StringSubclass('x'), b'bytes'):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                q.pop_compatible(KEY, claim_id=invalid, allowed_sizes=(1,), max_per_team=1)
            with self.assertRaises(ValueError):
                q.get_claim(invalid)
            with self.assertRaises(ValueError):
                q.acknowledge_claim(invalid)
        self.assertEqual(q.pop().job_id, 'a1')

    def test_interrupt_during_acknowledgment_restores_retained_claim(self):
        lines, first = inspect.getsourcelines(FairTeamQueue.acknowledge_claim)
        locations = [first + i for i, line in enumerate(lines)
                     if line.strip().startswith(('self._claims = ', 'self._claimed_jobs = ',
                                                 'self._claimed_size = '))]
        self.assertEqual(len(locations), 3)
        for target in locations:
            with self.subTest(line=target):
                q = FairTeamQueue(2, 1)
                q.put('a', 'a1', 1, compatibility_key=KEY)
                claimed = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1,), max_per_team=1)
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if (event == 'line' and frame.f_code is FairTeamQueue.acknowledge_claim.__code__
                            and frame.f_lineno == target):
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        q.acknowledge_claim('claim')
                finally:
                    sys.settrace(previous)
                recovered = []
                thread = Thread(target=lambda: recovered.append(q.get_claim('claim')), daemon=True)
                thread.start()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive(), 'queue lock remained held during acknowledgment rollback')
                self.assertEqual(recovered, [claimed])
                with self.assertRaises(ValueError):
                    q.put('a', 'a1', 2, compatibility_key=KEY)
                self.assertTrue(q.acknowledge_claim('claim'))

    def test_interrupt_after_acknowledgment_does_not_strand_lock(self):
        lines, first = inspect.getsourcelines(FairTeamQueue.acknowledge_claim)
        return_line = next(first + i for i, line in enumerate(lines)
                           if line.strip() == 'return acknowledged')
        for event_kind in ('line', 'return'):
            with self.subTest(event=event_kind):
                q = FairTeamQueue(2, 1)
                q.put('a', 'a1', 1, compatibility_key=KEY)
                owned = q.pop_compatible(KEY, claim_id='claim', allowed_sizes=(1,), max_per_team=1)
                q.put('b', 'b1', 2, compatibility_key=KEY)
                previous = sys.gettrace()
                def trace(frame, event, arg):
                    if (event == event_kind and frame.f_code is FairTeamQueue.acknowledge_claim.__code__
                            and (event == 'return' or frame.f_lineno == return_line)):
                        raise KeyboardInterrupt
                    return trace
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        q.acknowledge_claim('claim')
                finally:
                    sys.settrace(previous)
                observed = []
                def recover():
                    observed.append(q.get_claim('claim'))
                    observed.append(q.cancel('b1'))
                thread = Thread(target=recover, daemon=True)
                thread.start()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive(), 'queue lock remained held after acknowledgment interrupt')
                self.assertEqual(observed, [None, True])
                self.assertEqual([item.job_id for item in owned], ['a1'])


if __name__ == '__main__':
    unittest.main()
