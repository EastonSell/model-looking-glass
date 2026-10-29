import threading
import unittest
from dataclasses import dataclass
from time import perf_counter_ns

import torch
from torch import nn

from looking_glass.resident_worker import ForwardRequest, ResidentWorker

try:
    from looking_glass.broker import Broker, BrokerUnavailable, QueueFull, RequestState
except ModuleNotFoundError as exc:
    if exc.name not in ('looking_glass.broker', 'looking_glass.fair_queue'):
        raise
    Broker = None


@dataclass
class FakeRequest:
    request_id: str
    payload: bytes
    action: str = ''


class FakeCancelled(Exception):
    pass


class FakeOOM(Exception):
    pass


@dataclass(frozen=True)
class FakeResult:
    request_id: str
    payload: bytes
    owner_thread: int


class FakeWorker:
    """A deterministic CPU-only worker with a controllable forward call."""

    cancellation_exception = FakeCancelled

    def __init__(self):
        self.poisoned = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.order = []
        self.closed = False

    def snapshot_request(self, request):
        if not isinstance(request, FakeRequest) or not isinstance(request.payload, (bytes, bytearray)):
            raise ValueError('invalid fake request')
        return FakeRequest(request.request_id, bytes(request.payload), request.action or request.request_id)

    def execute(self, request, cancel_event=None):
        self.order.append(request.action)
        self.started.set()
        self.release.wait()
        if request.action == 'oom':
            self.poisoned = True
            raise FakeOOM('out of memory')
        if request.action == 'crash':
            raise SystemExit('owner failure')
        if request.action == 'cooperative' and cancel_event.is_set():
            raise FakeCancelled('cancelled')
        return FakeResult(request.request_id, request.payload, threading.get_ident())

    def close(self):
        self.closed = True


class BrokerTests(unittest.TestCase):
    def test_success_returns_result_and_monotonic_timing_from_owner(self):
        self.assertIsNotNone(Broker, 'broker not implemented')
        worker = FakeWorker()
        before = perf_counter_ns()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            outcome = broker.submit('alpha', FakeRequest('one', b'abc')).wait(2)
        after = perf_counter_ns()
        self.assertEqual(outcome.state, RequestState.SUCCEEDED)
        self.assertEqual((outcome.result.request_id, outcome.result.payload), ('one', b'abc'))
        self.assertNotEqual(outcome.result.owner_thread, threading.get_ident())
        self.assertLessEqual(outcome.submitted_ns, outcome.ready_ns)
        self.assertLessEqual(outcome.ready_ns, outcome.started_ns)
        self.assertLessEqual(outcome.started_ns, outcome.finished_ns)
        self.assertLessEqual(before, outcome.submitted_ns)
        self.assertLessEqual(outcome.finished_ns, after)

    def test_outcomes_distinguish_broker_sessions_after_job_id_reuse(self):
        with Broker(FakeWorker(), max_pending=1, max_pending_per_team=1) as first:
            a = first.submit('alpha', FakeRequest('one', b'x')).wait(2)
        with Broker(FakeWorker(), max_pending=1, max_pending_per_team=1) as second:
            b = second.submit('alpha', FakeRequest('one', b'x')).wait(2)
        self.assertEqual(a.job_id, b.job_id)
        self.assertEqual(a.broker_id, first.broker_id)
        self.assertEqual(b.broker_id, second.broker_id)
        self.assertNotEqual(a.broker_id, b.broker_id)

    def test_client_cannot_change_identity_and_cancel_another_request(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            first = broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            a = broker.submit('alpha', FakeRequest('a', b'a'))
            b = broker.submit('beta', FakeRequest('b', b'b'))
            try:
                for attr, value in (('job_id', b.job_id), ('team_id', 'beta'),
                                    ('request_id', 'b'), ('submitted_ns', 0)):
                    with self.assertRaises(AttributeError):
                        setattr(a, attr, value)
                self.assertTrue(a.cancel())
            finally:
                worker.release.set()
            self.assertEqual(first.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(a.wait(2).state, RequestState.CANCELLED)
            self.assertEqual((b.wait(2).result.request_id, b.wait().result.payload), ('b', b'b'))
            self.assertEqual(worker.order, ['running', 'b'])

    def test_pending_requests_rotate_fairly_and_snapshot_inputs(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=4, max_pending_per_team=2) as broker:
            running = broker.submit('alpha', FakeRequest('running', b'0'))
            self.assertTrue(worker.started.wait(2))
            payload = bytearray(b'original')
            a1 = broker.submit('alpha', FakeRequest('a1', payload))
            a2 = broker.submit('alpha', FakeRequest('a2', b'2'))
            b1 = broker.submit('beta', FakeRequest('b1', b'3'))
            b2 = broker.submit('beta', FakeRequest('b2', b'4'))
            payload[:] = b'mutated!'
            worker.release.set()
            for handle in (running, a1, a2, b1, b2):
                self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(a1.wait().result.payload, b'original')
            self.assertEqual(worker.order, ['running', 'a1', 'b1', 'a2', 'b2'])

    def test_queue_limits_cancel_and_invalid_request_release_capacity(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=1) as broker:
            broker.submit('alpha', FakeRequest('running', b'0'))
            self.assertTrue(worker.started.wait(2))
            queued = broker.submit('alpha', FakeRequest('queued', b'1'))
            try:
                with self.assertRaises(QueueFull):
                    broker.submit('alpha', FakeRequest('team-full', b'2'))
                with self.assertRaises(ValueError):
                    broker.submit('beta', object())
                other = broker.submit('beta', FakeRequest('other', b'3'))
                with self.assertRaises(QueueFull):
                    broker.submit('gamma', FakeRequest('global-full', b'4'))
                self.assertTrue(queued.cancel())
                self.assertFalse(queued.cancel())
                self.assertIsNone(queued.wait().started_ns)
                replacement = broker.submit('gamma', FakeRequest('replacement', b'5'))
            finally:
                worker.release.set()
            self.assertEqual(other.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(replacement.wait(2).state, RequestState.SUCCEEDED)

    def test_running_cancel_is_cooperative_and_discards_result(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            handle = broker.submit('alpha', FakeRequest('cooperative', b'x'))
            self.assertTrue(worker.started.wait(2))
            try:
                self.assertTrue(handle.cancel())
                self.assertFalse(handle.cancel())
                self.assertEqual(handle.state, RequestState.RUNNING)
                with self.assertRaises(TimeoutError):
                    handle.wait(0.01)
            finally:
                worker.release.set()
            outcome = handle.wait(2)
            self.assertEqual(outcome.state, RequestState.CANCELLED)
            self.assertIsNone(outcome.result)

    def test_slow_snapshot_does_not_block_another_jobs_cancellation(self):
        class SlowSnapshot(FakeWorker):
            def __init__(self):
                super().__init__()
                self.snapshot_entered = threading.Event()
                self.snapshot_release = threading.Event()

            def snapshot_request(self, request):
                if request.request_id == 'slow-snapshot':
                    self.snapshot_entered.set()
                    self.snapshot_release.wait()
                return super().snapshot_request(request)

        worker = SlowSnapshot()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        running = broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        submitted = []
        submitter = threading.Thread(target=lambda: submitted.append(
            broker.submit('beta', FakeRequest('slow-snapshot', b'y'))))
        submitter.start()
        self.assertTrue(worker.snapshot_entered.wait(2))
        cancel_done = threading.Event()
        canceller = threading.Thread(target=lambda: (running.cancel(), cancel_done.set()))
        canceller.start()
        try:
            self.assertTrue(cancel_done.wait(0.5), 'cancellation waited for an unrelated snapshot')
        finally:
            worker.snapshot_release.set()
            worker.release.set()
            submitter.join(2)
            canceller.join(2)
            broker.close(timeout=2)
        self.assertEqual(running.wait().state, RequestState.CANCELLED)

    def test_preparation_reserves_capacity_and_close_waits_for_it(self):
        class SlowSnapshot(FakeWorker):
            def __init__(self):
                super().__init__()
                self.snapshot_entered = threading.Event()
                self.snapshot_release = threading.Event()

            def snapshot_request(self, request):
                self.snapshot_entered.set()
                self.snapshot_release.wait()
                return super().snapshot_request(request)

        worker = SlowSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        errors = []

        def submit():
            try:
                broker.submit('alpha', FakeRequest('slow', b'x'))
            except Exception as exc:
                errors.append(type(exc))

        submitter = threading.Thread(target=submit)
        submitter.start()
        self.assertTrue(worker.snapshot_entered.wait(2))
        try:
            with self.assertRaises(QueueFull):
                broker.submit('beta', FakeRequest('overflow', b'y'))
            self.assertFalse(broker.close(wait=False))
            self.assertFalse(worker.closed)
        finally:
            worker.snapshot_release.set()
            submitter.join(2)
            self.assertTrue(broker.close(timeout=2))
        self.assertEqual(errors, [BrokerUnavailable])
        self.assertTrue(worker.closed)

    def test_team_fifo_uses_readiness_not_start_of_snapshot(self):
        class SlowFirst(FakeWorker):
            def __init__(self):
                super().__init__()
                self.snapshot_entered = threading.Event()
                self.snapshot_release = threading.Event()

            def snapshot_request(self, request):
                if request.request_id == 'first':
                    self.snapshot_entered.set()
                    self.snapshot_release.wait()
                return super().snapshot_request(request)

        worker = SlowFirst()
        worker.release.clear()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        running = broker.submit('other', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        first = []
        submitter = threading.Thread(target=lambda: first.append(
            broker.submit('alpha', FakeRequest('first', b'1'))))
        submitter.start()
        self.assertTrue(worker.snapshot_entered.wait(2))
        try:
            second = broker.submit('alpha', FakeRequest('second', b'2'))
            worker.snapshot_release.set()
            submitter.join(2)
            worker.release.set()
            for handle in (running, second, first[0]):
                self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(worker.order, ['running', 'second', 'first'])
        finally:
            worker.snapshot_release.set()
            worker.release.set()
            submitter.join(2)
            broker.close(timeout=2)

    def test_duplicate_external_ids_have_distinct_broker_ids_and_cancellation(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            first = broker.submit('alpha', FakeRequest('same', b'a'))
            self.assertTrue(worker.started.wait(2))
            second = broker.submit('alpha', FakeRequest('same', b'b'))
            try:
                self.assertNotEqual(first.job_id, second.job_id)
                self.assertTrue(second.cancel())
            finally:
                worker.release.set()
            self.assertEqual(first.wait(2).result.payload, b'a')
            self.assertEqual(second.wait().state, RequestState.CANCELLED)
            self.assertEqual(first.wait().request_id, second.wait().request_id)

    def test_oom_overrides_running_cancel_and_fails_pending(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            oom = broker.submit('alpha', FakeRequest('oom', b'x'))
            self.assertTrue(worker.started.wait(2))
            pending = broker.submit('beta', FakeRequest('waiting', b'y'))
            try:
                self.assertTrue(oom.cancel())
            finally:
                worker.release.set()
            outcome = oom.wait(2)
            self.assertEqual(outcome.state, RequestState.FAILED)
            self.assertEqual(outcome.error_type, 'FakeOOM')
            self.assertEqual(pending.wait(2).error_type, 'WorkerUnavailable')
            self.assertEqual(pending.state, RequestState.FAILED)
            self.assertIsNone(pending.wait().started_ns)
            with self.assertRaises(BrokerUnavailable):
                broker.submit('gamma', FakeRequest('after', b'z'))
        self.assertTrue(worker.closed)

    def test_poison_during_snapshot_fails_already_queued_work(self):
        class SnapshotOOM(FakeWorker):
            def snapshot_request(self, request):
                if request.request_id == 'bad-snapshot':
                    self.poisoned = True
                    raise FakeOOM('snapshot failed')
                return super().snapshot_request(request)

        worker = SnapshotOOM()
        worker.release.clear()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            running = broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            pending = broker.submit('beta', FakeRequest('waiting', b'y'))
            try:
                with self.assertRaises(FakeOOM):
                    broker.submit('gamma', FakeRequest('bad-snapshot', b'z'))
                self.assertEqual(pending.wait(0.2).state, RequestState.FAILED)
                self.assertEqual(pending.wait().error_type, 'WorkerUnavailable')
                with self.assertRaises(BrokerUnavailable):
                    broker.submit('gamma', FakeRequest('later', b'z'))
            finally:
                worker.release.set()
            self.assertEqual(running.wait(2).state, RequestState.SUCCEEDED)

    def test_overbroad_cancellation_exception_is_rejected(self):
        worker = FakeWorker()
        worker.cancellation_exception = Exception
        with self.assertRaises(TypeError):
            Broker(worker, max_pending=1, max_pending_per_team=1)

    def test_poisoned_worker_exception_is_never_reported_as_cancellation(self):
        class PoisonedCancellation(FakeWorker):
            def execute(self, request, cancel_event=None):
                self.poisoned = True
                raise FakeCancelled('worker is unhealthy')

        with Broker(PoisonedCancellation(), max_pending=1, max_pending_per_team=1) as broker:
            outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
            self.assertEqual(outcome.state, RequestState.FAILED)
            self.assertEqual(outcome.error_type, 'FakeCancelled')

    def test_mismatched_worker_result_id_fails_and_quarantines_worker(self):
        class WrongResult(FakeWorker):
            def execute(self, request, cancel_event=None):
                return FakeResult('another-job', b'private', threading.get_ident())

        with Broker(WrongResult(), max_pending=1, max_pending_per_team=1) as broker:
            outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
            self.assertEqual(outcome.state, RequestState.FAILED)
            self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
            self.assertIsNone(outcome.result)
            with self.assertRaises(BrokerUnavailable):
                broker.submit('alpha', FakeRequest('two', b'y'))

    def test_invalid_worker_result_property_fails_and_quarantines_worker(self):
        class InvalidResult:
            @property
            def request_id(self):
                raise RuntimeError('invalid result')

        class WrongResult(FakeWorker):
            def execute(self, request, cancel_event=None):
                return InvalidResult()

        with Broker(WrongResult(), max_pending=1, max_pending_per_team=1) as broker:
            outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
            self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
            with self.assertRaises(BrokerUnavailable):
                broker.submit('alpha', FakeRequest('two', b'y'))

    def test_stale_worker_result_fails_even_when_caller_reuses_request_id(self):
        class StaleResult(FakeWorker):
            previous = None

            def execute(self, request, cancel_event=None):
                if self.previous is None:
                    self.previous = request.request_id
                return FakeResult(self.previous, b'result', threading.get_ident())

        with Broker(StaleResult(), max_pending=1, max_pending_per_team=1) as broker:
            first = broker.submit('alpha', FakeRequest('same', b'a')).wait(2)
            self.assertEqual(first.state, RequestState.SUCCEEDED)
            second = broker.submit('beta', FakeRequest('same', b'b')).wait(2)
            self.assertEqual(second.state, RequestState.FAILED)
            self.assertEqual(second.error_type, 'InvalidWorkerResult')

    def test_unexpected_owner_error_does_not_strand_pending_jobs(self):
        class BadHealth(FakeWorker):
            @property
            def poisoned(self):
                if self.break_health:
                    raise RuntimeError('health check failed')
                return False

            @poisoned.setter
            def poisoned(self, value):
                self.break_health = value

            def execute(self, request, cancel_event=None):
                result = super().execute(request, cancel_event)
                self.break_health = True
                return result

        worker = BadHealth()
        worker.release.clear()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            running = broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            pending = broker.submit('beta', FakeRequest('waiting', b'y'))
            worker.release.set()
            self.assertEqual(running.wait(2).state, RequestState.FAILED)
            self.assertEqual(pending.wait(2).state, RequestState.FAILED)

    def test_owner_base_exception_fails_queued_work(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            crashed = broker.submit('alpha', FakeRequest('crash', b'x'))
            self.assertTrue(worker.started.wait(2))
            pending = broker.submit('beta', FakeRequest('waiting', b'y'))
            worker.release.set()
            self.assertEqual(crashed.wait(2).error_type, 'SystemExit')
            self.assertEqual(pending.wait(2).state, RequestState.FAILED)
            with self.assertRaises(BrokerUnavailable):
                broker.submit('gamma', FakeRequest('after', b'z'))

    def test_close_waits_for_running_work_without_accepting_new_jobs(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        self.assertTrue(broker.available)
        running = broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('pending', b'y'))
        try:
            self.assertFalse(broker.close(wait=False))
            self.assertFalse(broker.available)
            self.assertEqual(pending.wait(2).state, RequestState.CANCELLED)
            self.assertEqual(running.state, RequestState.RUNNING)
            with self.assertRaises(BrokerUnavailable):
                broker.submit('alpha', FakeRequest('later', b'z'))
        finally:
            worker.release.set()
            self.assertTrue(broker.close(timeout=2))
        self.assertEqual(running.wait().state, RequestState.CANCELLED)
        self.assertTrue(worker.closed)

    def test_real_cpu_worker_reuses_model_and_restores_caller_identity(self):
        class TokenModel(nn.Module):
            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1) + torch.tensor([0.0, 1.0])

        worker = ResidentWorker('tiny-cpu', TokenModel().eval(), max_sequence_length=4)
        with Broker(worker, max_pending=2, max_pending_per_team=1) as broker:
            first = broker.submit('alpha', ForwardRequest('same', {'input_ids': torch.tensor([[11]])}))
            second = broker.submit('beta', ForwardRequest('same', {'input_ids': torch.tensor([[22]])}))
            a, b = first.wait(2), second.wait(2)
            self.assertEqual(a.state, RequestState.SUCCEEDED)
            self.assertEqual(b.state, RequestState.SUCCEEDED)
            self.assertNotEqual(a.job_id, b.job_id)
            self.assertEqual((a.result.request_id, b.result.request_id), ('same', 'same'))
            self.assertTrue(torch.equal(a.result.logits, torch.tensor([[[11., 12.]]])))
            self.assertTrue(torch.equal(b.result.logits, torch.tensor([[[22., 23.]]])))


if __name__ == '__main__':
    unittest.main()
