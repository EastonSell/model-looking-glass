import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full
from threading import Barrier, Event, Thread

from looking_glass.fair_queue import FairTeamQueue


class FairTeamQueueTests(unittest.TestCase):
    def test_teams_take_turns_and_keep_submission_order(self):
        queue = FairTeamQueue(max_pending=8, max_pending_per_team=4)
        queue.put('a', 'a1', 1)
        queue.put('a', 'a2', 2)
        queue.put('b', 'b1', 3)
        queue.put('b', 'b2', 4)
        actual = [queue.pop().job_id for _ in range(4)]
        self.assertEqual(actual, ['a1', 'b1', 'a2', 'b2'])

    def test_global_capacity_rejects_submission_and_can_be_reused(self):
        queue = FairTeamQueue(max_pending=2, max_pending_per_team=2)
        queue.put('a', 'a1', 1)
        queue.put('b', 'b1', 2)
        with self.assertRaises(Full):
            queue.put('c', 'c1', 3)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.pop().job_id, 'a1')
        queue.put('c', 'c1', 3)
        self.assertEqual([queue.pop().job_id, queue.pop().job_id], ['b1', 'c1'])

    def test_per_team_capacity_rejects_one_team_but_allows_another(self):
        queue = FairTeamQueue(max_pending=4, max_pending_per_team=2)
        queue.put('a', 'a1', 1)
        queue.put('a', 'a2', 2)
        with self.assertRaises(Full):
            queue.put('a', 'a3', 3)
        queue.put('b', 'b1', 4)
        self.assertEqual([queue.pop().job_id for _ in range(3)], ['a1', 'b1', 'a2'])

    def test_empty_pop_raises_standard_empty(self):
        queue = FairTeamQueue(max_pending=1, max_pending_per_team=1)
        with self.assertRaises(Empty):
            queue.pop()

    def test_invalid_capacity_is_rejected(self):
        for value in (0, -1, True, 1.5, '2'):
            for args in ((value, 2), (2, value)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    FairTeamQueue(*args)

    def test_pending_job_ids_are_unique_across_teams(self):
        queue = FairTeamQueue(max_pending=3, max_pending_per_team=2)
        queue.put('a', 'job', 1)
        with self.assertRaises(ValueError):
            queue.put('b', 'job', 2)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.pop().payload, 1)
        queue.put('b', 'job', 3)
        self.assertEqual(queue.pop().payload, 3)

    def test_cancel_removes_pending_item_and_releases_capacity(self):
        queue = FairTeamQueue(max_pending=3, max_pending_per_team=2)
        queue.put('a', 'a1', 1)
        queue.put('a', 'a2', 2)
        queue.put('b', 'b1', 3)
        self.assertTrue(queue.cancel('a1'))
        self.assertFalse(queue.cancel('a1'))
        self.assertFalse(queue.cancel('missing'))
        queue.put('a', 'a1', 4)
        self.assertEqual(len(queue), 3)
        self.assertEqual([queue.pop().job_id for _ in range(3)], ['a2', 'b1', 'a1'])
        self.assertFalse(queue.cancel('a1'))

    def test_canceling_last_team_job_removes_its_place_in_rotation(self):
        queue = FairTeamQueue(max_pending=4, max_pending_per_team=2)
        queue.put('a', 'a1', 1)
        queue.put('b', 'b1', 2)
        queue.put('c', 'c1', 3)
        self.assertTrue(queue.cancel('b1'))
        queue.put('b', 'b2', 4)
        self.assertEqual([queue.pop().job_id for _ in range(3)], ['a1', 'c1', 'b2'])

    def test_arrivals_during_an_executing_job_cannot_overtake_waiting_team(self):
        queue = FairTeamQueue(max_pending=8, max_pending_per_team=4)
        queue.put('a', 'a1', 1)
        queue.put('a', 'a2', 2)
        queue.put('b', 'b1', 3)
        self.assertEqual(queue.pop().job_id, 'a1')
        queue.put('c', 'c1', 4)
        queue.put('a', 'a3', 5)
        queue.put('d', 'd1', 6)
        self.assertEqual([queue.pop().job_id for _ in range(5)], ['b1', 'a2', 'c1', 'd1', 'a3'])

    def test_reentry_after_last_dispatch_uses_arrival_order(self):
        queue = FairTeamQueue(max_pending=4, max_pending_per_team=2)
        queue.put('a', 'a1', 1)
        self.assertEqual(queue.pop().job_id, 'a1')
        queue.put('a', 'a2', 2)
        queue.put('b', 'b1', 3)
        self.assertEqual([queue.pop().job_id, queue.pop().job_id], ['a2', 'b1'])

    def test_new_team_gets_a_turn_even_if_another_keeps_submitting(self):
        queue = FairTeamQueue(max_pending=4, max_pending_per_team=2)
        queue.put('a', 'a0', None)
        queue.put('a', 'a1', None)
        queue.put('b', 'b0', None)
        self.assertEqual(queue.pop().job_id, 'a0')
        queue.put('a', 'a2', None)
        self.assertEqual(queue.pop().job_id, 'b0')

    def test_concurrent_submissions_obey_capacity_without_losing_jobs(self):
        for same_team, limit in ((False, 10), (True, 8)):
            with self.subTest(same_team=same_team):
                queue = FairTeamQueue(max_pending=10, max_pending_per_team=8 if same_team else 1)
                barrier = Barrier(32)

                def submit(i):
                    barrier.wait()
                    try:
                        queue.put('a' if same_team else str(i), str(i), i)
                        return i
                    except Full:
                        return None

                with ThreadPoolExecutor(max_workers=32) as workers:
                    admitted = list(workers.map(submit, range(32)))
                admitted = {value for value in admitted if value is not None}
                self.assertEqual(len(admitted), limit)
                self.assertEqual(len(queue), limit)
                self.assertEqual({queue.pop().payload for _ in range(limit)}, admitted)

    def test_simultaneous_duplicate_ids_admit_only_one(self):
        queue = FairTeamQueue(max_pending=20, max_pending_per_team=20)
        barrier = Barrier(10)

        def submit(i):
            barrier.wait()
            try:
                queue.put(i, 'same', i)
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=10) as workers:
            admitted = list(workers.map(submit, range(10)))
        self.assertEqual(sum(admitted), 1)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.pop().job_id, 'same')

    def test_simultaneous_cancel_and_pop_have_one_winner(self):
        for _ in range(20):
            queue = FairTeamQueue(max_pending=1, max_pending_per_team=1)
            queue.put('a', 'job', 42)
            barrier = Barrier(2)

            def pop():
                barrier.wait()
                try:
                    return queue.pop().payload
                except Empty:
                    return None

            def cancel():
                barrier.wait()
                return queue.cancel('job')

            with ThreadPoolExecutor(max_workers=2) as workers:
                pop_result = workers.submit(pop)
                cancel_result = workers.submit(cancel)
                self.assertIn((pop_result.result(), cancel_result.result()), [(42, False), (None, True)])
            self.assertEqual(len(queue), 0)

    def test_cancel_releases_payload_without_holding_queue_lock(self):
        queue = FairTeamQueue(max_pending=1, max_pending_per_team=1)
        released = Event()

        class ReentrantPayload:
            def __del__(self):
                len(queue)
                released.set()

        queue.put('team', 'job', ReentrantPayload())
        worker = Thread(target=queue.cancel, args=('job',), daemon=True)
        worker.start()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive(), 'payload finalizer deadlocked on queue lock')
        self.assertTrue(released.wait(timeout=1))


if __name__ == '__main__':
    unittest.main()
