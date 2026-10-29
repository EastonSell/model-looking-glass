import threading
import sys
import unittest
import inspect
import gc
import weakref
from time import monotonic, sleep
from unittest.mock import patch

import torch

from looking_glass.broker import Broker, BrokerShutdownError, RequestState
from looking_glass.resident_worker import ForwardRequest, ForwardResult


class Cancelled(Exception):
    pass


class IdleWorker:
    cancellation_exception = Cancelled
    poisoned = False

    def close(self):
        pass


class FailingWorker(IdleWorker):
    def snapshot_request(self, request):
        return ForwardRequest(request.request_id, {'input_ids': request.inputs['input_ids'].clone()}, request.seed)

    def execute(self, request, cancel_event=None):
        self.poisoned = True
        raise RuntimeError('worker failed')


class BrokerHealthSnapshotTests(unittest.TestCase):
    def test_health_is_observable_while_admission_condition_is_held(self):
        broker = Broker(IdleWorker(), max_pending=1, max_pending_per_team=1)
        entered, release, observed = (threading.Event() for _ in range(3))
        snapshots = []
        errors = []

        def hold_admission_condition():
            with broker._condition:
                entered.set()
                release.wait(3)

        def observe_health():
            try:
                snapshots.append(broker.health_snapshot)
            except BaseException as exc:
                errors.append(exc)
            finally:
                observed.set()

        holder = threading.Thread(target=hold_admission_condition)
        observer = threading.Thread(target=observe_health)
        holder.start()
        try:
            self.assertTrue(entered.wait(2))
            observer.start()
            self.assertTrue(observed.wait(.5), 'health observation blocked on broker admission')
            self.assertEqual(errors, [])
            self.assertEqual(len(snapshots), 1)
            health = snapshots[0]
            self.assertFalse(health.failed)
            self.assertFalse(health.quarantined)
            self.assertFalse(health.closed)
            self.assertTrue(health.owner_alive)
        finally:
            release.set()
            holder.join(3)
            if observer.ident is not None:
                observer.join(3)
            broker.close(timeout=3)

    def test_health_reports_worker_failure_and_orderly_broker_close(self):
        broker = Broker(FailingWorker(), max_pending=1, max_pending_per_team=1)
        request = ForwardRequest('job', {'input_ids': torch.tensor([[1]], dtype=torch.int64)})
        try:
            outcome = broker.submit('team', request).wait(3)
            self.assertEqual(outcome.state, RequestState.FAILED)
            health = broker.health_snapshot
            self.assertTrue(health.failed)
            self.assertFalse(health.quarantined)
            self.assertFalse(health.closed)
        finally:
            broker.close(timeout=3)
        health = broker.health_snapshot
        self.assertTrue(health.failed)
        self.assertTrue(health.closed)
        self.assertFalse(health.owner_alive)

    def test_health_does_not_probe_a_worker_status_property(self):
        class UnreadableStatusWorker(IdleWorker):
            @property
            def poisoned(self):
                raise AssertionError('health polling must not call the worker')

        broker = Broker(UnreadableStatusWorker(), max_pending=1, max_pending_per_team=1)
        try:
            self.assertFalse(broker.health_snapshot.failed)
        finally:
            broker.close(timeout=3)

    def test_health_reports_quarantine_after_invalid_worker_result(self):
        class InvalidResultWorker(FailingWorker):
            def execute(self, request, cancel_event=None):
                return ForwardResult('wrong-job', torch.tensor([1]))

        broker = Broker(InvalidResultWorker(), max_pending=1, max_pending_per_team=1)
        request = ForwardRequest('job', {'input_ids': torch.tensor([[1]], dtype=torch.int64)})
        outcome = broker.submit('team', request).wait(3)
        self.assertEqual(outcome.error_type, 'InvalidWorkerResult')
        health = broker.health_snapshot
        self.assertTrue(health.quarantined)
        self.assertTrue(health.failed)
        with self.assertRaises(BrokerShutdownError):
            broker.close(timeout=3)

    def test_owner_failure_is_visible_before_blocked_retention_gets_condition(self):
        owner_entered, fail_owner, condition_held, release_condition = (threading.Event() for _ in range(4))

        class FailingOwnerBroker(Broker):
            def _serve(self):
                owner_entered.set()
                fail_owner.wait(3)
                raise RuntimeError('unexpected owner failure')

        broker = FailingOwnerBroker(IdleWorker(), max_pending=1, max_pending_per_team=1)

        def hold_condition():
            with broker._condition:
                condition_held.set()
                release_condition.wait(3)

        holder = threading.Thread(target=hold_condition)
        try:
            self.assertTrue(owner_entered.wait(2))
            holder.start()
            self.assertTrue(condition_held.wait(2))
            fail_owner.set()
            deadline = monotonic() + 2
            while monotonic() < deadline:
                frame = sys._current_frames().get(broker._thread.ident)
                while frame is not None and frame.f_code.co_name != '_retain_unknown':
                    frame = frame.f_back
                if frame is not None:
                    break
                sleep(.01)
            else:
                self.fail('owner did not enter failure retention')
            health = broker.health_snapshot
            self.assertTrue(health.quarantined)
            self.assertTrue(health.failed)
        finally:
            fail_owner.set()
            release_condition.set()
            if holder.ident is not None:
                holder.join(3)
            broker._thread.join(3)
            with self.assertRaises(BrokerShutdownError):
                broker.close(timeout=3)

    def test_interrupted_owner_failure_handler_keeps_original_exception_private(self):
        owner_entered, fail_owner, condition_held, release_condition, injected = (
            threading.Event() for _ in range(5))
        refs = []

        class Token:
            pass

        class FailingOwnerBroker(Broker):
            def _serve(self):
                token = Token()
                refs.append(weakref.ref(token))
                owner_entered.set()
                fail_owner.wait(3)
                raise RuntimeError('unexpected owner failure')

        function = getattr(Broker, '_run_inner', Broker._run)
        lines, first = inspect.getsourcelines(function)
        handler = next(i for i, line in enumerate(lines) if line.strip() == 'except BaseException as exc:')
        target = next(first + i for i in range(handler + 1, len(lines)) if lines[i].strip() == 'try:')

        def trace(frame, event, arg):
            if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno == target:
                injected.set()
                raise KeyboardInterrupt('interrupted failure handler')
            return trace

        def hold_condition(broker):
            with broker._condition:
                condition_held.set()
                release_condition.wait(3)

        with patch.object(threading, 'excepthook', lambda args: None):
            threading.settrace(trace)
            try:
                broker = FailingOwnerBroker(IdleWorker(), max_pending=1, max_pending_per_team=1)
            finally:
                threading.settrace(None)
            holder = threading.Thread(target=hold_condition, args=(broker,))
            try:
                self.assertTrue(owner_entered.wait(2))
                holder.start()
                self.assertTrue(condition_held.wait(2))
                fail_owner.set()
                self.assertTrue(injected.wait(2))
                deadline = monotonic() + 2
                while monotonic() < deadline and not broker.health_snapshot.quarantined:
                    sleep(.01)
                self.assertTrue(broker.health_snapshot.quarantined)
            finally:
                fail_owner.set()
                release_condition.set()
                if holder.ident is not None:
                    holder.join(3)
                broker._thread.join(3)
                try:
                    broker.close(timeout=3)
                except BrokerShutdownError:
                    pass
            gc.collect()
            self.assertIsNotNone(refs[0](), 'original failure traceback was not retained')

    def test_interrupted_worker_close_failure_keeps_owner_registered(self):
        refs = []
        injected = threading.Event()

        class Token:
            pass

        class FailingCloseWorker(IdleWorker):
            def close(self):
                token = Token()
                refs.append(weakref.ref(token))
                raise RuntimeError('worker close failed')

        function = Broker._run_inner
        lines, first = inspect.getsourcelines(function)
        target = max(first + i for i, line in enumerate(lines)
                     if line.strip() == 'self._retain_unknown(exc)')

        def trace(frame, event, arg):
            if frame.f_code is function.__code__ and event == 'line' and frame.f_lineno == target:
                injected.set()
                raise KeyboardInterrupt('interrupted close error retention')
            return trace

        with patch.object(threading, 'excepthook', lambda args: None):
            threading.settrace(trace)
            try:
                worker = FailingCloseWorker()
                broker = Broker(worker, max_pending=1, max_pending_per_team=1)
            finally:
                threading.settrace(None)
            try:
                broker.close(timeout=3)
            except BrokerShutdownError:
                pass
            self.assertTrue(injected.wait(2))
            broker._thread.join(3)
            self.assertTrue(broker.health_snapshot.quarantined)
            broker_ref = weakref.ref(broker)
            del broker, worker
            gc.collect()
            self.assertIsNotNone(broker_ref(), 'quarantined broker was unregistered')
            self.assertIsNotNone(refs[0](), 'worker close traceback was not retained')


if __name__ == '__main__':
    unittest.main()
