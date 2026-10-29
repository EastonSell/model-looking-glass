import threading
import unittest
import weakref
import gc
import inspect
import sys
from dataclasses import dataclass
from time import perf_counter_ns
from unittest.mock import patch

import torch
from torch import nn

from looking_glass.resident_worker import ForwardRequest, ForwardResult, ResidentWorker

try:
    from looking_glass.broker import Broker, BrokerShutdownError, BrokerUnavailable, QueueFull, RequestState
except ModuleNotFoundError as exc:
    if exc.name not in ('looking_glass.broker', 'looking_glass.fair_queue'):
        raise
    Broker = None


def FakeRequest(request_id, payload, action=''):
    """Build a real forward request while carrying deterministic fake data."""
    action_bytes = (action or request_id).encode()
    return ForwardRequest(request_id, {'input_ids': torch.tensor(
        [[len(action_bytes), *action_bytes, *payload]], dtype=torch.int64)})


class FakeCancelled(Exception):
    pass


class FakeOOM(Exception):
    pass


class FakeWorker:
    """A deterministic CPU-only worker with a controllable forward call."""

    cancellation_exception = FakeCancelled

    def __init__(self):
        self.poisoned = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.order = []
        self.owner_thread = None
        self.closed = False

    def snapshot_request(self, request):
        if type(request) is not ForwardRequest:
            raise ValueError('invalid fake request')
        return ForwardRequest(request.request_id, {name: value.clone() for name, value in request.inputs.items()},
                              request.seed)

    def execute(self, request, cancel_event=None):
        tokens = request.inputs['input_ids'][0]
        action_length = int(tokens[0])
        action = bytes(tokens[1:1 + action_length].tolist()).decode()
        self.order.append(action)
        self.owner_thread = threading.get_ident()
        self.started.set()
        self.release.wait()
        if action == 'oom':
            self.poisoned = True
            raise FakeOOM('out of memory')
        if action == 'crash':
            raise SystemExit('owner failure')
        if action == 'cooperative' and cancel_event.is_set():
            raise FakeCancelled('cancelled')
        return ForwardResult(request.request_id, tokens[1 + action_length:].clone())

    def close(self):
        self.closed = True


class SnapshotOwner:
    """Simulate a claim tied to one tensor wrapper, not its storage aliases."""

    def __init__(self, released):
        self.released = released

    def __del__(self):
        self.released.set()


class ClaimTrackingWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.claims = {}
        self.aliases = {}

    def snapshot_request(self, request):
        snapshot = super().snapshot_request(request)
        if request.request_id.startswith('owned'):
            tensor = snapshot.inputs['input_ids']
            released = threading.Event()
            tensor.owner = SnapshotOwner(released)
            self.claims[request.request_id] = (weakref.ref(tensor.owner), released)
            self.aliases[request.request_id] = tensor.detach()
        return snapshot


def interrupt_line(function, text, operation):
    lines, first_line = inspect.getsourcelines(function)
    target = next(first_line + i for i, line in enumerate(lines) if line.strip() == text)

    def inject(frame, event, arg):
        if event == 'line' and frame.f_code is function.__code__ and frame.f_lineno == target:
            sys.settrace(None)
            raise KeyboardInterrupt(f'injected before {text}')
        return inject

    sys.settrace(inject)
    try:
        return operation()
    finally:
        sys.settrace(None)


class BrokerTests(unittest.TestCase):
    def test_success_returns_result_and_monotonic_timing_from_owner(self):
        self.assertIsNotNone(Broker, 'broker not implemented')
        worker = FakeWorker()
        before = perf_counter_ns()
        with Broker(worker, max_pending=2, max_pending_per_team=2) as broker:
            outcome = broker.submit('alpha', FakeRequest('one', b'abc')).wait(2)
        after = perf_counter_ns()
        self.assertEqual(outcome.state, RequestState.SUCCEEDED)
        self.assertEqual((outcome.result.request_id, bytes(outcome.result.logits.tolist())), ('one', b'abc'))
        self.assertNotEqual(worker.owner_thread, threading.get_ident())
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
            self.assertEqual((b.wait(2).result.request_id, bytes(b.wait().result.logits.tolist())), ('b', b'b'))
            self.assertEqual(worker.order, ['running', 'b'])

    def test_pending_requests_rotate_fairly_and_snapshot_inputs(self):
        worker = FakeWorker()
        worker.release.clear()
        with Broker(worker, max_pending=4, max_pending_per_team=2) as broker:
            running = broker.submit('alpha', FakeRequest('running', b'0'))
            self.assertTrue(worker.started.wait(2))
            request = FakeRequest('a1', b'original')
            a1 = broker.submit('alpha', request)
            a2 = broker.submit('alpha', FakeRequest('a2', b'2'))
            b1 = broker.submit('beta', FakeRequest('b1', b'3'))
            b2 = broker.submit('beta', FakeRequest('b2', b'4'))
            request.inputs['input_ids'][:] = 0
            worker.release.set()
            for handle in (running, a1, a2, b1, b2):
                self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
            self.assertEqual(bytes(a1.wait().result.logits.tolist()), b'original')
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

    def test_handle_allocation_failure_does_not_leak_preparation_slot(self):
        worker = FakeWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        try:
            with patch('looking_glass.broker.RequestHandle', side_effect=MemoryError('allocation')):
                with self.assertRaises(MemoryError):
                    broker.submit('alpha', FakeRequest('failed', b'x'))
            handle = broker.submit('alpha', FakeRequest('after', b'y'))
            self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)
        finally:
            broker.close(timeout=0.2)

    def test_thread_start_raising_after_start_retains_worker_and_stops_owner(self):
        worker = FakeWorker()
        original_start = threading.Thread.start
        captured = []

        def start_then_raise(thread):
            captured.append((thread, thread._target.__self__))
            original_start(thread)
            raise KeyboardInterrupt('injected after thread start')

        with patch('looking_glass.broker.Thread.start', new=start_then_raise):
            with self.assertRaises(KeyboardInterrupt) as caught:
                Broker(worker, max_pending=1, max_pending_per_team=1)
        self.assertIsNone(caught.exception.__context__)
        thread, owner = captured[0]
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(owner.quarantined)
        self.assertFalse(worker.closed)
        from looking_glass import broker as broker_module
        self.assertIn(owner, broker_module._owner_registry)

    def test_failed_worker_close_retains_worker_and_quarantines(self):
        class Claim:
            pass

        class BadClose(FakeWorker):
            def close(self):
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                raise RuntimeError('indeterminate teardown')

        worker = BadClose()
        worker_ref = weakref.ref(worker)
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        self.assertTrue(broker.quarantined)
        del broker, worker
        gc.collect()
        self.assertIsNotNone(worker_ref())
        self.assertIsNotNone(worker_ref().claim_ref())

    def test_readiness_timestamp_failure_does_not_publish_a_queue_item(self):
        worker = FakeWorker()
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            with patch('looking_glass.broker.perf_counter_ns', side_effect=[1, MemoryError('clock')]):
                with self.assertRaises(MemoryError):
                    broker.submit('alpha', FakeRequest('failed', b'x'))
            self.assertEqual(broker._pending_slots, 0)
            self.assertEqual(len(broker._queue), 0)
            handle = broker.submit('alpha', FakeRequest('after', b'y'))
            self.assertEqual(handle.wait(2).state, RequestState.SUCCEEDED)

    def test_queue_publication_failure_retains_unknown_accounting(self):
        worker = FakeWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        original_put = broker._queue.put

        def insert_then_fail(*args):
            original_put(*args)
            raise MemoryError('queue publication failed')

        broker._queue.put = insert_then_fail
        try:
            with self.assertRaises(BrokerUnavailable):
                broker.submit('alpha', FakeRequest('uncertain', b'x'))
            self.assertFalse(broker.available)
            self.assertEqual(broker._pending_slots, 1)
            self.assertFalse(worker.started.is_set())
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)
            self.assertEqual(broker._pending_slots, 1)
        finally:
            broker._queue.put = original_put

    def test_failed_snapshot_cleanup_marks_accounting_unknown(self):
        worker = FakeWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        original_release = broker._release_slot_locked

        def failed_release(_):
            raise MemoryError('cannot update reservation')

        broker._release_slot_locked = failed_release
        try:
            with patch.object(worker, 'snapshot_request', side_effect=ValueError('invalid snapshot')):
                with self.assertRaises(BrokerUnavailable):
                    broker.submit('alpha', FakeRequest('bad', b'x'))
            self.assertFalse(broker.available)
            self.assertEqual(broker._pending_slots, 1)
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)
        finally:
            broker._release_slot_locked = original_release

    def test_latched_fatal_cleanup_does_not_drain_on_worker_poison(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        try:
            broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            pending = broker.submit('beta', FakeRequest('pending', b'y'))
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            worker.poisoned = True
            worker.release.set()
            broker._thread.join(2)
            self.assertFalse(broker._thread.is_alive())
            self.assertEqual((len(broker._queue), broker._pending_slots), (1, 1))
            with self.assertRaises(BrokerUnavailable):
                pending.wait(0.1)
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=0.1)
            self.assertEqual((len(broker._queue), broker._pending_slots), (1, 1))
        finally:
            worker.release.set()

    def test_latched_fatal_cleanup_retains_slot_after_snapshot_poison(self):
        class SnapshotOOM(FakeWorker):
            def __init__(self):
                super().__init__()
                self.snapshot_started = threading.Event()
                self.snapshot_release = threading.Event()

            def snapshot_request(self, request):
                if request.request_id == 'bad-snapshot':
                    self.snapshot_started.set()
                    self.snapshot_release.wait(2)
                    self.poisoned = True
                    raise FakeOOM('snapshot failed')
                return super().snapshot_request(request)

        worker = SnapshotOOM()
        worker.release.clear()
        broker = Broker(worker, max_pending=3, max_pending_per_team=2)
        errors = []
        submitter = None
        try:
            broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            broker.submit('beta', FakeRequest('pending', b'y'))
            def prepare():
                try:
                    broker.submit('gamma', FakeRequest('bad-snapshot', b'z'))
                except BaseException as exc:
                    errors.append(exc)
            submitter = threading.Thread(target=prepare, daemon=True)
            submitter.start()
            self.assertTrue(worker.snapshot_started.wait(2))
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            worker.snapshot_release.set()
            submitter.join(2)
            self.assertFalse(submitter.is_alive())
            self.assertIsInstance(errors[0], BrokerUnavailable)
            worker.release.set()
            broker._thread.join(2)
            self.assertEqual((len(broker._queue), broker._pending_slots), (1, 2))
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=0.1)
        finally:
            worker.snapshot_release.set()
            worker.release.set()
            if submitter:
                submitter.join(2)

    def test_latched_fatal_cleanup_does_not_drain_after_owner_exception(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        try:
            broker.submit('alpha', FakeRequest('running', b'x'))
            self.assertTrue(worker.started.wait(2))
            broker.submit('beta', FakeRequest('pending', b'y'))
            original_finish = broker._finish_locked
            calls = []
            def fail_once(*args, **kwargs):
                calls.append(True)
                if len(calls) == 1:
                    raise MemoryError('injected publication failure')
                return original_finish(*args, **kwargs)
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            with patch.object(broker, '_finish_locked', side_effect=fail_once):
                worker.release.set()
                broker._thread.join(2)
            self.assertFalse(broker._thread.is_alive())
            self.assertEqual((len(broker._queue), broker._pending_slots), (1, 1))
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=0.1)
        finally:
            worker.release.set()

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
            self.assertEqual(bytes(first.wait(2).result.logits.tolist()), b'a')
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

    def test_non_forward_and_subclass_requests_are_rejected_before_snapshot(self):
        @dataclass(frozen=True)
        class TracedRequest(ForwardRequest):
            trace_policy: str = 'capture'

        worker = FakeWorker()
        snapshot = worker.snapshot_request
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            with patch.object(worker, 'snapshot_request', wraps=snapshot) as spy:
                for request in (object(), TracedRequest('trace', {'input_ids': torch.tensor([1])})):
                    with self.subTest(request_type=type(request).__name__):
                        with self.assertRaises(ValueError):
                            broker.submit('alpha', request)
                spy.assert_not_called()
            self.assertEqual(broker._pending_slots, 0)
            self.assertEqual(broker.submit('alpha', FakeRequest('ok', b'x')).wait(2).state,
                             RequestState.SUCCEEDED)

    def test_worker_cannot_return_a_trace_snapshot_or_subclass(self):
        @dataclass(frozen=True)
        class TracedRequest(ForwardRequest):
            trace_policy: str = 'capture'

        class TraceSnapshot(FakeWorker):
            def snapshot_request(self, request):
                return TracedRequest(request.request_id, request.inputs)

        worker = TraceSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('trace', b'x'))
        self.assertTrue(broker.quarantined)
        self.assertEqual(broker._pending_slots, 1)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        self.assertFalse(worker.closed)

    def test_worker_cannot_relabel_a_snapshot(self):
        class RelabelSnapshot(FakeWorker):
            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                return ForwardRequest('another-client-id', snapshot.inputs, snapshot.seed)

        worker = RelabelSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('caller-id', b'x'))
        self.assertFalse(worker.started.is_set())
        self.assertTrue(broker.quarantined)
        self.assertEqual(broker._pending_slots, 1)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_worker_snapshot_with_extra_owned_data_is_retained(self):
        class Claim:
            pass

        class BadSnapshot(FakeWorker):
            claim_ref = None

            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                return ForwardRequest(snapshot.request_id, {**snapshot.inputs, 'trace_claim': claim})

        worker = BadSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(worker.claim_ref())
        self.assertFalse(worker.closed)

    def test_interrupted_quarantine_notification_keeps_snapshot_and_blocks_work(self):
        class Claim:
            pass

        class BadSnapshot(FakeWorker):
            claim_ref = None

            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                return ForwardRequest(snapshot.request_id, {**snapshot.inputs, 'trace_claim': claim})

        worker = BadSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with patch.object(broker, '_mark_cleanup_failed_locked', side_effect=KeyboardInterrupt('notification')):
            with self.assertRaises(KeyboardInterrupt):
                broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertTrue(broker.quarantined)
        self.assertFalse(broker.available)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('later', b'y'))
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(worker.claim_ref())
        self.assertFalse(worker.closed)

    def test_interruption_before_retention_slot_write_retries_before_unwinding(self):
        class Claim:
            pass

        class BadSnapshot(FakeWorker):
            claim_ref = None

            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                return ForwardRequest(snapshot.request_id, {**snapshot.inputs, 'trace_claim': claim})

        worker = BadSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        original_store = broker._store_quarantine_value_locked
        calls = []

        def fail_once(value):
            calls.append(True)
            if len(calls) == 1:
                raise KeyboardInterrupt('before slot write')
            return original_store(value)

        with patch.object(broker, '_store_quarantine_value_locked', side_effect=fail_once):
            with self.assertRaises(KeyboardInterrupt):
                broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(worker.claim_ref())

    def test_unexpected_owner_failure_retains_result_through_traceback(self):
        class Claim:
            pass

        class ClaimWorker(FakeWorker):
            claim_ref = None
            alias = None

            def execute(self, request, cancel_event=None):
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                logits = torch.tensor([1])
                logits.claim = claim
                self.alias = logits.detach()
                self.started.set()
                self.release.wait(2)
                return ForwardResult(request.request_id, logits)

        worker = ClaimWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertTrue(worker.started.wait(2))
        original_finish = broker._finish_locked
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise MemoryError('before terminal publication')
            return original_finish(*args, **kwargs)

        with patch.object(broker, '_finish_locked', side_effect=fail_once):
            worker.release.set()
            broker._thread.join(2)
        self.assertFalse(broker._thread.is_alive())
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(worker.claim_ref())
        self.assertEqual(worker.alias.tolist(), [1])
        self.assertFalse(worker.closed)

    def test_concurrent_bad_snapshots_keep_each_claim_in_separate_slot(self):
        class Claim:
            pass

        class BadSnapshot(FakeWorker):
            def __init__(self):
                super().__init__()
                self.barrier = threading.Barrier(2)
                self.claim_refs = []

            def snapshot_request(self, request):
                snapshot = super().snapshot_request(request)
                claim = Claim()
                self.claim_refs.append(weakref.ref(claim))
                self.barrier.wait(2)
                return ForwardRequest(snapshot.request_id, {**snapshot.inputs, 'trace_claim': claim})

        worker = BadSnapshot()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        errors = []

        def submit(request_id):
            try:
                broker.submit('alpha', FakeRequest(request_id, b'x'))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=submit, args=(str(i),), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(exc, BrokerUnavailable) for exc in errors))
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertEqual(len(worker.claim_refs), 2)
        self.assertTrue(all(ref() is not None for ref in worker.claim_refs))

    def test_worker_cannot_publish_trace_or_non_cpu_tensor_results(self):
        @dataclass(frozen=True)
        class TracedResult(ForwardResult):
            claim: str = 'owned'

        class TraceResultWorker(FakeWorker):
            def __init__(self, result_factory):
                super().__init__()
                self.result_factory = result_factory

            def execute(self, request, cancel_event=None):
                return self.result_factory(request.request_id)

        def inference_tensor_result(request_id):
            with torch.inference_mode():
                logits = torch.ones(1)
            return ForwardResult(request_id, logits)

        cases = (
            (lambda request_id: TracedResult(request_id, torch.tensor([1])), True),
            (lambda request_id: ForwardResult(request_id, 'not a tensor'), True),
            (lambda request_id: ForwardResult(request_id, torch.empty(1, device='meta')), True),
            (lambda request_id: ForwardResult(request_id, torch.tensor([1.0], requires_grad=True)), True),
            (inference_tensor_result, True),
            (lambda request_id: ForwardResult(request_id, torch.Tensor._make_subclass(
                type('TensorSubclass', (torch.Tensor,), {}), torch.tensor([1]))), True),
        )
        for make_result, quarantine in cases:
            with self.subTest(factory=make_result):
                worker = TraceResultWorker(make_result)
                broker = Broker(worker, max_pending=1, max_pending_per_team=1)
                try:
                    outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
                    self.assertEqual(outcome.state, RequestState.FAILED)
                    self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
                    self.assertIsNone(outcome.result)
                    self.assertTrue(broker.failed)
                    self.assertEqual(broker.quarantined, quarantine)
                finally:
                    if quarantine:
                        with self.assertRaises(BrokerShutdownError):
                            broker.close(timeout=2)
                        self.assertFalse(worker.closed)
                    else:
                        broker.close(timeout=2)

    def test_unknown_result_and_worker_remain_retained_after_broker_is_dropped(self):
        @dataclass(frozen=True)
        class TracedResult(ForwardResult):
            claim: str = 'unknown'

        class TraceResultWorker(FakeWorker):
            result_ref = None

            def execute(self, request, cancel_event=None):
                result = TracedResult(request.request_id, torch.tensor([1]))
                self.result_ref = weakref.ref(result)
                return result

        worker = TraceResultWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        self.assertEqual(broker.submit('alpha', FakeRequest('one', b'x')).wait(2).error_type,
                         'InvalidWorkerResult')
        self.assertTrue(broker.quarantined)
        result_ref = worker.result_ref
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(result_ref())
        self.assertFalse(worker.closed)

    def test_wrong_id_plain_tensor_result_retains_owned_claim(self):
        class Claim:
            pass

        class WrongResult(FakeWorker):
            claim_ref = None
            alias = None

            def execute(self, request, cancel_event=None):
                claim = Claim()
                self.claim_ref = weakref.ref(claim)
                logits = torch.tensor([1])
                logits.claim = claim
                self.alias = logits.detach()
                return ForwardResult('wrong-id', logits)

        worker = WrongResult()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        self.assertEqual(broker.submit('alpha', FakeRequest('one', b'x')).wait(2).error_type,
                         'InvalidWorkerResult')
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        del broker
        gc.collect()
        self.assertIsNotNone(worker.claim_ref())
        self.assertEqual(worker.alias.tolist(), [1])
        self.assertFalse(worker.closed)

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
                return ForwardResult('another-job', torch.tensor([1]))

        broker = Broker(WrongResult(), max_pending=1, max_pending_per_team=1)
        outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
        self.assertEqual(outcome.state, RequestState.FAILED)
        self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
        self.assertIsNone(outcome.result)
        self.assertTrue(broker.failed)
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('two', b'y'))
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_invalid_worker_result_property_fails_and_quarantines_worker(self):
        class InvalidResult:
            @property
            def request_id(self):
                raise RuntimeError('invalid result')

        class WrongResult(FakeWorker):
            def execute(self, request, cancel_event=None):
                return InvalidResult()

        broker = Broker(WrongResult(), max_pending=1, max_pending_per_team=1)
        outcome = broker.submit('alpha', FakeRequest('one', b'x')).wait(2)
        self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('alpha', FakeRequest('two', b'y'))
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_stale_worker_result_fails_even_when_caller_reuses_request_id(self):
        class StaleResult(FakeWorker):
            previous = None

            def execute(self, request, cancel_event=None):
                if self.previous is None:
                    self.previous = request.request_id
                return ForwardResult(self.previous, torch.tensor([1]))

        broker = Broker(StaleResult(), max_pending=1, max_pending_per_team=1)
        first = broker.submit('alpha', FakeRequest('same', b'a')).wait(2)
        self.assertEqual(first.state, RequestState.SUCCEEDED)
        second = broker.submit('beta', FakeRequest('same', b'b')).wait(2)
        self.assertEqual(second.state, RequestState.FAILED)
        self.assertEqual(second.error_type, 'InvalidWorkerResult')
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

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

    def test_persistent_queue_oom_wakes_waiters_without_claiming_completion(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('pending', b'y'))
        original_pop = broker._queue.pop

        def persistent_oom():
            raise MemoryError('queue cannot pop')

        broker._queue.pop = persistent_oom
        try:
            worker.release.set()
            with self.assertRaises(BrokerUnavailable):
                pending.wait(2)
            self.assertFalse(broker.available)
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)
        finally:
            broker._queue.pop = original_pop
            try:
                broker.close(timeout=2)
            except BrokerShutdownError:
                pass

    def test_queue_oom_during_close_wakes_pending_waiters(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('pending', b'y'))
        original_pop = broker._queue.pop

        def persistent_oom():
            raise MemoryError('queue cannot pop')

        broker._queue.pop = persistent_oom
        try:
            with self.assertRaises(BrokerShutdownError):
                broker.close(wait=False)
            with self.assertRaises(BrokerUnavailable):
                pending.wait(2)
        finally:
            worker.release.set()
            broker._queue.pop = original_pop
            try:
                broker.close(timeout=2)
            except BrokerShutdownError:
                pass

    def test_terminal_publication_oom_keeps_pending_slot_and_input(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('pending', b'y'))
        original_finish = broker._finish_locked

        def broken_finish(handle, state, *args, **kwargs):
            if handle is pending:
                raise MemoryError('cannot publish outcome')
            return original_finish(handle, state, *args, **kwargs)

        broker._finish_locked = broken_finish
        try:
            with self.assertRaises(BrokerUnavailable):
                pending.cancel()
            with self.assertRaises(BrokerUnavailable):
                pending.wait(2)
            self.assertEqual(broker._pending_slots, 1)
            self.assertIsNotNone(pending._job)
        finally:
            broker._finish_locked = original_finish
            worker.release.set()
            try:
                broker.close(timeout=2)
            except BrokerShutdownError:
                pass

    def test_interrupted_pending_cancel_keeps_removed_snapshot_rooted(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('owned-one', b'y'))
        reference, released = worker.claims['owned-one']
        original_cancel = broker._queue.cancel

        def removed_then_interrupted(job_id):
            self.assertTrue(original_cancel(job_id))
            raise KeyboardInterrupt('after queue removal')

        broker._queue.cancel = removed_then_interrupted
        try:
            with self.assertRaises(KeyboardInterrupt) as caught:
                pending.cancel()
            self.assertIsNone(caught.exception.__context__)
            del pending
            gc.collect()
            self.assertTrue(broker.quarantined)
            self.assertIsNotNone(reference())
            self.assertFalse(released.is_set())
            self.assertEqual(len(broker._queue), 0)
        finally:
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)

    def test_interrupted_close_retains_all_removed_snapshots(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=2, max_pending_per_team=2)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        first = broker.submit('beta', FakeRequest('owned-one', b'y'))
        second = broker.submit('beta', FakeRequest('owned-two', b'z'))
        claims = list(worker.claims.values())
        original_pop = broker._queue.pop
        calls = 0

        def second_pop_interrupted():
            nonlocal calls
            item = original_pop()
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt('after second queue removal')
            return item

        broker._queue.pop = second_pop_interrupted
        try:
            with self.assertRaises(KeyboardInterrupt) as caught:
                broker.close(wait=False)
            self.assertIsNone(caught.exception.__context__)
            del first, second
            gc.collect()
            self.assertTrue(broker.quarantined)
            for reference, released in claims:
                self.assertIsNotNone(reference())
                self.assertFalse(released.is_set())
        finally:
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)

    def test_interrupted_dispatch_keeps_removed_snapshot_rooted(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('owned-one', b'y'))
        reference, released = worker.claims['owned-one']
        original_pop = broker._queue.pop

        def removed_then_interrupted():
            original_pop()
            raise KeyboardInterrupt('after dispatch removal')

        broker._queue.pop = removed_then_interrupted
        worker.release.set()
        broker._thread.join(2)
        with self.assertRaises(BrokerUnavailable):
            pending.wait(2)
        del pending
        gc.collect()
        self.assertTrue(broker.quarantined)
        self.assertIsNotNone(reference())
        self.assertFalse(released.is_set())
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_retirement_capacity_counts_removed_jobs_until_cleanup(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('waiting', b'y'))
        retire_started = threading.Event()
        retire_continue = threading.Event()
        original_retire = broker._retire_job

        def blocked_retire(job):
            retire_started.set()
            retire_continue.wait(2)
            return original_retire(job)

        broker._retire_job = blocked_retire
        canceller = threading.Thread(target=pending.cancel)
        canceller.start()
        try:
            self.assertTrue(retire_started.wait(2))
            with self.assertRaises(QueueFull):
                broker.submit('gamma', FakeRequest('over-cap', b'z'))
        finally:
            retire_continue.set()
            canceller.join(2)
            broker._retire_job = original_retire
        accepted = broker.submit('gamma', FakeRequest('accepted', b'z'))
        worker.release.set()
        self.assertEqual(accepted.wait(2).state, RequestState.SUCCEEDED)
        self.assertTrue(broker.close(timeout=2))
        self.assertTrue(all(job is None for job in broker._job_roots))

    def test_cancel_drops_broker_snapshot_before_releasing_root_slot(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('owned-one', b'y'))
        reference, released = worker.claims['owned-one']
        retire_done = threading.Event()
        return_allowed = threading.Event()
        original_retire = broker._retire_job

        def paused_after_retire(job):
            result = original_retire(job)
            retire_done.set()
            return_allowed.wait(2)
            return result

        broker._retire_job = paused_after_retire
        canceller = threading.Thread(target=pending.cancel)
        canceller.start()
        try:
            self.assertTrue(retire_done.wait(2))
            gc.collect()
            self.assertIsNone(reference())
            self.assertTrue(released.is_set())
            # The still-paused cancellation frame owns its job, but no longer
            # retains the broker snapshot once the slot can be reused.
            self.assertIn('owned-one', worker.aliases)
        finally:
            return_allowed.set()
            canceller.join(2)
            broker._retire_job = original_retire
            worker.release.set()
            self.assertTrue(broker.close(timeout=2))

    def test_prepublication_error_traceback_does_not_retain_snapshot(self):
        worker = ClaimTrackingWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        errors = []
        try:
            for index in range(5):
                request_id = f'owned-{index}'
                with patch('looking_glass.broker.perf_counter_ns',
                           side_effect=[1, RuntimeError('ready clock failed')]):
                    try:
                        broker.submit('alpha', FakeRequest(request_id, b'x'))
                    except RuntimeError as exc:
                        errors.append(exc)
                    else:
                        self.fail('injected publication failure did not propagate')
                reference, released = worker.claims[request_id]
                gc.collect()
                self.assertIsNone(reference())
                self.assertTrue(released.is_set())
            self.assertEqual(len(errors), 5)
            self.assertTrue(broker.available)
        finally:
            self.assertTrue(broker.close(timeout=2))

    def test_simultaneous_retirement_interruptions_keep_each_input_and_error(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=3, max_pending_per_team=3)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = [broker.submit('beta', FakeRequest(f'owned-{i}', b'x')) for i in range(3)]
        claims = list(worker.claims.values())
        lines, first_line = inspect.getsourcelines(Broker._retire_job)
        disposal_line = next(first_line + i for i, line in enumerate(lines)
                             if line.strip() == 'retiring_request = None' and i > 16)
        gate = threading.Barrier(3, timeout=3)
        errors = []

        def cancel_with_interrupt(handle):
            def inject(frame, event, arg):
                if (event == 'line' and frame.f_code is Broker._retire_job.__code__
                        and frame.f_lineno == disposal_line):
                    sys.settrace(None)
                    gate.wait()
                    raise KeyboardInterrupt('before detached-input disposal')
                return inject
            sys.settrace(inject)
            try:
                handle.cancel()
            except BaseException as exc:
                errors.append(exc)
            finally:
                sys.settrace(None)

        threads = [threading.Thread(target=cancel_with_interrupt, args=(handle,)) for handle in pending]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        try:
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual([type(error) for error in errors], [KeyboardInterrupt] * 3)
            self.assertTrue(all(error.__context__ is None for error in errors))
            del pending
            gc.collect()
            for reference, released in claims:
                self.assertIsNotNone(reference())
                self.assertFalse(released.is_set())
            self.assertTrue(broker.quarantined)
        finally:
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)

    def test_quarantine_before_retirement_commit_preserves_snapshot(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('owned-one', b'y'))
        reference, released = worker.claims['owned-one']
        reached = threading.Event()
        release = threading.Event()
        original_retire = broker._retire_job

        def gated_retire(job):
            reached.set()
            release.wait(2)
            return original_retire(job)

        broker._retire_job = gated_retire
        canceller = threading.Thread(target=pending.cancel)
        canceller.start()
        try:
            self.assertTrue(reached.wait(2))
            with broker._condition:
                broker._mark_cleanup_failed_locked()
            release.set()
            canceller.join(2)
            del pending
            gc.collect()
            self.assertTrue(broker.quarantined)
            self.assertIsNotNone(reference())
            self.assertFalse(released.is_set())
        finally:
            release.set()
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)

    def test_interrupt_at_public_retirement_handoff_quarantines(self):
        for operation in ('cancel', 'close', 'failed_snapshot'):
            with self.subTest(operation=operation):
                class FailsSnapshot(ClaimTrackingWorker):
                    def snapshot_request(self, request):
                        if request.request_id == 'reject':
                            raise ValueError('bad request')
                        return super().snapshot_request(request)

                worker = FailsSnapshot()
                worker.release.clear()
                broker = Broker(worker, max_pending=1, max_pending_per_team=1)
                broker.submit('alpha', FakeRequest('running', b'x'))
                self.assertTrue(worker.started.wait(2))
                if operation == 'failed_snapshot':
                    target, line = Broker._submit, 'self._retire_job(job)'
                    action = lambda: broker.submit('beta', FakeRequest('reject', b'y'))
                else:
                    pending = broker.submit('beta', FakeRequest('owned-one', b'y'))
                    if operation == 'cancel':
                        target, line = type(pending)._cancel_request, 'broker._retire_job(job)'
                        action = pending.cancel
                    else:
                        target, line = Broker._close, 'self._retire_items(removed)'
                        action = lambda: broker.close(wait=False)
                try:
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        interrupt_line(target, line, action)
                    self.assertIsNone(caught.exception.__context__)
                    self.assertTrue(broker.quarantined)
                    self.assertTrue(any(job is not None for job in broker._job_roots))
                finally:
                    worker.release.set()
                    with self.assertRaises(BrokerShutdownError):
                        broker.close(timeout=2)

    def test_interrupt_after_queue_publication_does_not_retire_queued_job(self):
        worker = ClaimTrackingWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        try:
            with self.assertRaises(KeyboardInterrupt) as caught:
                interrupt_line(Broker._submit, 'return handle',
                               lambda: broker.submit('beta', FakeRequest('owned-one', b'y')))
            self.assertIsNone(caught.exception.__context__)
            self.assertTrue(broker.quarantined)
            self.assertEqual(len(broker._queue), 1)
            self.assertEqual(broker._pending_slots, 1)
            reference, released = worker.claims['owned-one']
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertFalse(released.is_set())
        finally:
            worker.release.set()
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=2)

    def test_close_without_timeout_reports_orphaned_preparation_promptly(self):
        broker = Broker(FakeWorker(), max_pending=1, max_pending_per_team=1)
        with self.assertRaises(KeyboardInterrupt):
            interrupt_line(Broker._submit, 'self._next_job_id = next_job_id',
                           lambda: broker.submit('alpha', FakeRequest('one', b'x')))
        self.assertTrue(broker.quarantined)
        self.assertEqual(broker._preparing, 1)
        errors = []

        def close_owner():
            try:
                broker.close()
            except BaseException as exc:
                errors.append(exc)

        closer = threading.Thread(target=close_owner, daemon=True)
        closer.start()
        closer.join(1)
        try:
            self.assertFalse(closer.is_alive())
            self.assertEqual([type(error) for error in errors], [BrokerShutdownError])
            self.assertTrue(broker._thread.is_alive())
        finally:
            # The injected interruption prevented a real submitter from owning
            # this counter. Remove only the synthetic test leak after checking.
            with broker._condition:
                broker._preparing = 0
                broker._condition.notify_all()
            broker._thread.join(2)

    def test_close_waiting_for_worker_wakes_when_quarantine_latches(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        broker.submit('alpha', FakeRequest('running', b'x'))
        self.assertTrue(worker.started.wait(2))
        errors = []

        def close_owner():
            try:
                broker.close()
            except BaseException as exc:
                errors.append(exc)

        closer = threading.Thread(target=close_owner, daemon=True)
        closer.start()
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: broker._closed, 2))
            broker._mark_cleanup_failed_locked()
        closer.join(1)
        try:
            self.assertFalse(closer.is_alive())
            self.assertEqual([type(error) for error in errors], [BrokerShutdownError])
            self.assertFalse(worker.closed)
        finally:
            worker.release.set()
            broker._thread.join(2)

    def test_owner_base_exception_quarantines_and_preserves_queued_work(self):
        worker = FakeWorker()
        worker.release.clear()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        crashed = broker.submit('alpha', FakeRequest('crash', b'x'))
        self.assertTrue(worker.started.wait(2))
        pending = broker.submit('beta', FakeRequest('waiting', b'y'))
        worker.release.set()
        broker._thread.join(2)
        self.assertEqual(crashed.wait(2).error_type, 'SystemExit')
        with self.assertRaises(BrokerUnavailable):
            pending.wait(2)
        self.assertTrue(broker.quarantined)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('gamma', FakeRequest('after', b'z'))
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_interrupt_during_worker_snapshot_quarantines(self):
        class Interrupted(FakeWorker):
            def snapshot_request(self, request):
                raise KeyboardInterrupt('return boundary')

        worker = Interrupted()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(KeyboardInterrupt) as caught:
            broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        self.assertFalse(broker.available)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)
        self.assertFalse(worker.closed)

    def test_quarantined_snapshot_exception_is_not_exposed_to_caller(self):
        class InterruptedSnapshot(FakeWorker):
            def snapshot_request(self, request):
                tensor = request.inputs['input_ids'].clone()
                released = threading.Event()
                tensor.owner = SnapshotOwner(released)
                self.claim_ref = weakref.ref(tensor.owner)
                self.released = released
                self.alias = tensor.detach()
                raise KeyboardInterrupt('after making private tensor')

        worker = InterruptedSnapshot()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with self.assertRaises(KeyboardInterrupt) as caught:
            broker.submit('alpha', FakeRequest('private', b'x'))
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        gc.collect()
        self.assertTrue(broker.quarantined)
        self.assertIsNotNone(worker.claim_ref())
        self.assertFalse(worker.released.is_set())
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_sanitized_error_does_not_attach_caller_active_exception(self):
        worker = FakeWorker()
        broker = Broker(worker, max_pending=1, max_pending_per_team=1)
        with broker._condition:
            broker._mark_cleanup_failed_locked()
        try:
            raise RuntimeError('caller private error')
        except RuntimeError:
            with self.assertRaises(BrokerUnavailable) as caught:
                broker.submit('alpha', FakeRequest('one', b'x'))
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=2)

    def test_raw_torch_oom_from_custom_worker_quarantines_both_phases(self):
        class RawOom(FakeWorker):
            def __init__(self, phase):
                super().__init__()
                self.phase = phase

            def snapshot_request(self, request):
                if self.phase == 'snapshot':
                    raise torch.OutOfMemoryError('raw OOM')
                return super().snapshot_request(request)

            def execute(self, request, cancel_event=None):
                if self.phase == 'execute':
                    raise torch.OutOfMemoryError('raw OOM')
                return super().execute(request, cancel_event)

        for phase in ('snapshot', 'execute'):
            with self.subTest(phase=phase):
                worker = RawOom(phase)
                broker = Broker(worker, max_pending=1, max_pending_per_team=1)
                if phase == 'snapshot':
                    with self.assertRaises(BrokerUnavailable):
                        broker.submit('alpha', FakeRequest('one', b'x'))
                else:
                    handle = broker.submit('alpha', FakeRequest('one', b'x'))
                    broker._thread.join(2)
                    self.assertEqual(handle.wait(2).error_type, 'OutOfMemoryError')
                self.assertTrue(broker.quarantined)
                with self.assertRaises(BrokerShutdownError):
                    broker.close(timeout=2)
                self.assertFalse(worker.closed)

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
