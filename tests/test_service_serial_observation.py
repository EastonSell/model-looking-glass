"""CPU-only checks for the exact fixed-workspace observation owner."""

import gc
import inspect
import sys
import threading
import unittest
import weakref
from time import perf_counter_ns
from unittest.mock import patch

import torch
from torch import nn

from looking_glass.batched_recorder import BatchedEventRecorder
from looking_glass.broker import Broker, BrokerUnavailable, RequestHandle, RequestState
from looking_glass.budgeted_worker import WorkloadProfile
from looking_glass.ipc import IPCClient, IPCServer
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from looking_glass.resident_worker import ForwardRequest, ResidentWorker
from looking_glass.service_serial_observation import ServiceSerialObservation


class ObservedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def forward(self, input_ids):
        self.calls.append((perf_counter_ns(), tuple(input_ids.shape)))
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('model test timed out')
        return {'logits': input_ids.float().unsqueeze(-1)}


def service_stack(*, device_guard=None, bind=True):
    model = ObservedModel().eval()
    resident = ResidentWorker('a' * 64, model, device='cpu', max_batch_size=1,
                              max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    load = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    guard_times = []
    if device_guard is None:
        device_guard = lambda: guard_times.append(perf_counter_ns())
    authority = OwnerCompletionAuthority(device_guard=device_guard)
    recorder = BatchedEventRecorder()
    observation = ServiceSerialObservation(recorder)
    profile = WorkloadProfile('c' * 64, 'a' * 64, 'cpu', (1, 2),
                              (('input_ids', 'torch.int64'),), 64, 1024)
    worker = LifetimeWorkspaceWorker(resident, budget, load, workspace,
                                     profiles=(profile,), authority=authority,
                                     cpu_observation=observation)
    broker = Broker(worker, max_pending=4, max_pending_per_team=2, completion_authority=authority)
    if bind:
        observation.bind_broker(broker)
    return model, worker, broker, authority, observation, recorder, guard_times


class ServiceSerialObservationTests(unittest.TestCase):
    def test_attach_requires_broker_observation_hook(self):
        _, _, broker, authority, observation, _, _ = service_stack(bind=False)
        # Simulate an older broker before the owner-private hook exists.
        if hasattr(broker, '_serial_observation'):
            del broker._serial_observation
        try:
            with self.assertRaisesRegex(ValueError, 'bind the exact warm owner'):
                observation.bind_broker(broker)
            self.assertFalse(authority.unsafe)
        finally:
            broker.close()

    def test_attach_binds_exact_warm_worker_authority_and_broker(self):
        _, worker, broker, authority, observation, _, _ = service_stack(bind=False)
        try:
            observation.bind_broker(broker)
            self.assertIs(observation._broker, broker)
            self.assertIs(observation._worker, worker)
            self.assertIs(observation._authority, authority)
            self.assertIs(broker._serial_observation, observation)
            with self.assertRaises(ValueError):
                ServiceSerialObservation(BatchedEventRecorder()).bind_broker(broker)
            self.assertFalse(authority.unsafe)
        finally:
            broker.close()
        self.assertTrue(observation.seal(producers_stopped=True)['events_complete'])

    def test_work_before_observation_binding_fails_closed(self):
        model, _, broker, authority, observation, _, _ = service_stack(bind=False)
        try:
            with self.assertRaises(BrokerUnavailable):
                broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
            self.assertTrue(authority.unsafe)
            self.assertFalse(model.entered.is_set())
            with self.assertRaises(ValueError):
                observation.bind_broker(broker)
        finally:
            model.release.set()

    def test_cuda_observation_binding_is_disabled_without_a_profiled_path(self):
        _, worker, broker, authority, observation, _, _ = service_stack(bind=False)
        resident = worker._worker
        original_device = resident._device
        try:
            # This only changes metadata; no CUDA API or model execution runs.
            resident._device = torch.device('cuda:0')
            with self.assertRaisesRegex(ValueError, 'CUDA service observation is disabled'):
                observation.bind_broker(broker)
            self.assertFalse(authority.unsafe)
            self.assertIsNone(broker._serial_observation)
        finally:
            resident._device = original_device
            broker.close()

    def test_worker_capability_requires_exact_observer_and_cpu_device(self):
        _, worker, broker, _, observation, _, _ = service_stack(bind=False)
        resident = worker._worker
        original_device = resident._device
        try:
            args = (resident, worker._budget, worker._resident_lease, worker._workspace_lease)
            profiles = tuple(worker._profiles.values())
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                LifetimeWorkspaceWorker(*args, profiles=profiles,
                    authority=OwnerCompletionAuthority(device_guard=lambda: None),
                    cpu_observation=object())
            # Alter only the device descriptor. Rejection must precede CUDA
            # queries, admission, device allocation or use of the model.
            resident._device = torch.device('cuda:0')
            with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA queried')):
                with self.assertRaisesRegex(ValueError, 'CPU-only'):
                    LifetimeWorkspaceWorker(*args, profiles=profiles,
                        authority=OwnerCompletionAuthority(device_guard=lambda: None),
                        cpu_observation=observation)
        finally:
            resident._device = original_device
            broker.close()

    def test_removing_bound_observation_latches_owner_before_new_work(self):
        model, worker, broker, authority, _, _, _ = service_stack()
        broker._serial_observation = None
        self.assertTrue(worker.quarantined)
        self.assertTrue(authority.unsafe)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        self.assertFalse(model.entered.is_set())
        model.release.set()

    def test_mutating_worker_observation_capability_latches_owner(self):
        model, worker, broker, authority, _, _, _ = service_stack()
        worker._cpu_observation = None
        self.assertTrue(worker.quarantined)
        self.assertTrue(authority.unsafe)
        with self.assertRaises(BrokerUnavailable):
            broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        self.assertFalse(model.entered.is_set())
        model.release.set()

    def test_removing_both_observer_references_cannot_impersonate_disabled_policy(self):
        model, worker, broker, authority, _, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        worker._cpu_observation = None
        broker._serial_observation = None
        with self.assertRaises(BrokerUnavailable):
            handle.wait(3)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        broker._thread.join(3)

    def test_fast_worker_can_finish_before_submit_registration_returns(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        registering, resume = threading.Event(), threading.Event()
        original = observation._registered_handle
        handles, errors = [], []

        def delayed_registration(handle):
            registering.set()
            if not resume.wait(3):
                raise RuntimeError('registration test timed out')
            return original(handle)

        def submit():
            try:
                handles.append(broker.submit('team', ForwardRequest(
                    'job', {'input_ids': torch.tensor([[1, 2]])})))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=submit)
        try:
            with patch.object(observation, '_registered_handle', side_effect=delayed_registration):
                thread.start()
                self.assertTrue(registering.wait(3))
                self.assertTrue(model.entered.wait(3))
                with broker._condition:
                    self.assertTrue(broker._condition.wait_for(
                        lambda: broker._running is None and broker._next_job_id == 1, 3))
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(handles[0].wait(3).state, RequestState.SUCCEEDED)
            self.assertFalse(authority.unsafe)
        finally:
            resume.set()
            if thread.ident is not None:
                thread.join(3)
            broker.close(timeout=3)
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual(sum(e['kind'] == 'forward_finished' for e in report['events']), 1)

    def test_competing_registration_preserves_handles_and_invalidates_report(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        registering, resume = threading.Event(), threading.Event()
        original = observation._registered_handle
        handles, errors = [], []

        def delayed_registration(handle):
            registering.set()
            if not resume.wait(3):
                raise RuntimeError('registration test timed out')
            return original(handle)

        def first_submit():
            try:
                handles.append(broker.submit('team-a', ForwardRequest(
                    'job-a', {'input_ids': torch.tensor([[1, 2]])})))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=first_submit)
        try:
            with patch.object(observation, '_registered_handle', side_effect=delayed_registration):
                thread.start()
                self.assertTrue(registering.wait(3))
                second = broker.submit('team-b', ForwardRequest(
                    'job-b', {'input_ids': torch.tensor([[3, 4]])}))
                self.assertEqual(second.wait(3).state, RequestState.SUCCEEDED)
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(handles[0].wait(3).state, RequestState.SUCCEEDED)
            self.assertFalse(authority.unsafe)
        finally:
            resume.set()
            if thread.ident is not None:
                thread.join(3)
            broker.close(timeout=3)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_terminal_capture_interrupt_is_sanitized_and_fails_owner_closed(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        private = KeyboardInterrupt('private observation error')
        with patch.object(observation, '_captured_handle', side_effect=private):
            with self.assertRaises(KeyboardInterrupt) as caught:
                handle.wait(3)
        self.assertIsNot(caught.exception, private)
        self.assertIsNone(caught.exception.__context__)
        self.assertIs(observation._terminal_capture.failure, private)
        self.assertIs(observation._terminal_capture.handle, handle)
        self.assertIs(observation._terminal_capture.outcome, handle._outcome)
        self.assertIs(broker._observation_failure_owner, observation)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        broker._thread.join(3)

    def test_observation_failure_has_a_root_when_other_quarantine_slots_are_full(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
            unrelated = [object() for _ in broker._quarantine_objects]
            broker._quarantine_objects[:] = unrelated
        private = KeyboardInterrupt('private observation error')
        with patch.object(observation, '_captured_handle', side_effect=private):
            with self.assertRaises(KeyboardInterrupt) as caught:
                handle.wait(3)
        self.assertIsNot(caught.exception, private)
        self.assertIsNone(caught.exception.__context__)
        self.assertIs(broker._observation_failure_owner, observation)
        self.assertIs(observation._terminal_capture.failure, private)
        self.assertEqual(broker._quarantine_objects, unrelated)
        self.assertTrue(authority.unsafe)
        broker._thread.join(3)

    def test_terminal_notification_failure_is_rooted_and_sanitized(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        private = KeyboardInterrupt('private callback failure')
        notification = RuntimeError('private notification failure')
        with patch.object(observation, '_captured_handle', side_effect=private), \
             patch.object(broker, '_mark_cleanup_failed_locked', side_effect=notification):
            with self.assertRaises(BrokerUnavailable) as caught:
                handle.wait(0)
        self.assertIsNot(caught.exception, notification)
        self.assertIsNone(caught.exception.__context__)
        self.assertIs(observation._terminal_capture.failure, private)
        self.assertIs(broker._observation_cleanup_failure, notification)
        self.assertIs(broker._observation_failure_owner, observation)
        self.assertTrue(authority.unsafe)
        broker._thread.join(3)

    def test_registration_notification_failure_is_rooted_and_sanitized(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        private = KeyboardInterrupt('private callback failure')
        notification = RuntimeError('private notification failure')
        with patch.object(observation, '_registered_handle', side_effect=private), \
             patch.object(broker, '_mark_cleanup_failed_locked', side_effect=notification):
            with self.assertRaises(BrokerUnavailable) as caught:
                broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        self.assertIsNot(caught.exception, notification)
        self.assertIsNone(caught.exception.__context__)
        self.assertIs(observation._registration_callback.failure, private)
        self.assertIs(broker._observation_cleanup_failure, notification)
        self.assertIs(broker._observation_failure_owner, observation)
        self.assertTrue(authority.unsafe)
        broker._thread.join(3)

    def test_notification_return_interrupt_is_sanitized_for_wait_and_submit(self):
        for operation in ('wait', 'submit'):
            with self.subTest(operation=operation):
                model, _, broker, authority, observation, _, _ = service_stack()
                model.release.set()
                if operation == 'wait':
                    handle = broker.submit('team', ForwardRequest(
                        'job', {'input_ids': torch.tensor([[1, 2]])}))
                    with broker._condition:
                        self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
                private = KeyboardInterrupt('private callback failure')
                secondary = RuntimeError('private notification return interruption')
                injected = []

                def trace(frame, event, arg):
                    if (frame.f_code is Broker._retain_observation_failure.__code__
                            and event == 'return' and not injected):
                        injected.append(True)
                        raise secondary
                    return trace

                callback = '_captured_handle' if operation == 'wait' else '_registered_handle'
                with patch.object(observation, callback, side_effect=private):
                    try:
                        sys.settrace(trace)
                        with self.assertRaises(BrokerUnavailable) as caught:
                            if operation == 'wait':
                                handle.wait(0)
                            else:
                                broker.submit('team', ForwardRequest(
                                    'job', {'input_ids': torch.tensor([[1, 2]])}))
                    finally:
                        sys.settrace(None)
                self.assertTrue(injected)
                self.assertIsNot(caught.exception, secondary)
                self.assertIsNone(caught.exception.__context__)
                self.assertIs(broker._observation_cleanup_failure, secondary)
                state = (observation._terminal_capture if operation == 'wait'
                         else observation._registration_callback)
                self.assertIs(state.failure, private)
                self.assertIs(broker._observation_failure_owner, observation)
                self.assertTrue(authority.unsafe)
                broker._thread.join(3)

    def test_many_terminal_entry_interrupts_use_one_preallocated_failure_root(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        count = 23  # More than this broker's 19 quarantine slots.
        barrier, errors = threading.Barrier(count, timeout=5), []

        def interrupt_entry(*args, **kwargs):
            barrier.wait()
            raise KeyboardInterrupt('private terminal entry')

        def wait():
            try:
                handle.wait(0)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=wait) for _ in range(count)]
        with patch.object(observation, '_callback_bounded_inner', side_effect=interrupt_entry):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(8)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), count)
        self.assertTrue(all(type(exc) is KeyboardInterrupt and exc.__context__ is None for exc in errors))
        self.assertTrue(authority.unsafe)
        self.assertIs(broker._observation_failure_owner, observation)
        # The previously committed result remains readable without invoking a
        # fresh callback after quarantine.
        self.assertEqual(handle.wait(0).state, RequestState.SUCCEEDED)
        broker._thread.join(3)

    def test_many_terminal_identity_interrupts_share_one_observer_root(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        count = 23
        barrier, errors = threading.Barrier(count, timeout=5), []

        def interrupted_identity():
            barrier.wait()
            raise KeyboardInterrupt('private identity check')

        def wait():
            try:
                handle.wait(0)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=wait) for _ in range(count)]
        with patch.object(broker, '_service_observer', side_effect=interrupted_identity):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(8)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), count)
        self.assertTrue(all(type(exc) is KeyboardInterrupt and exc.__context__ is None for exc in errors))
        self.assertTrue(authority.unsafe)
        self.assertTrue(observation._faulted)
        self.assertIs(broker._observation_failure_owner, observation)
        broker._thread.join(3)

    def test_gate_boundary_interrupts_quarantine_and_keep_callback_state_rooted(self):
        source, first_line = inspect.getsourcelines(ServiceSerialObservation._callback_bounded_inner)
        for target in ('if self._broker.quarantined or state.active or not state.lock.acquire(False):',
                       'state.active = True',
                       'state.handle, state.outcome = handle, outcome',
                       'self._captured_handle(handle, outcome)',
                       'state.lock.release()'):
            with self.subTest(target=target):
                model, _, broker, authority, observation, _, _ = service_stack()
                model.release.set()
                handle = broker.submit('team', ForwardRequest(
                    'job', {'input_ids': torch.tensor([[1, 2]])}))
                with broker._condition:
                    self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
                line = first_line + next(i for i, text in enumerate(source) if text.strip() == target)
                injected = []

                def trace(frame, event, arg):
                    if (frame.f_code is ServiceSerialObservation._callback_bounded_inner.__code__
                            and event == 'line' and frame.f_lineno == line and not injected):
                        injected.append(True)
                        raise KeyboardInterrupt('gate boundary')
                    return trace

                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        handle.wait(3)
                finally:
                    sys.settrace(None)
                self.assertTrue(injected)
                self.assertIsNone(caught.exception.__context__)
                self.assertTrue(authority.unsafe)
                state = observation._terminal_capture
                self.assertIsNotNone(state.interruption)
                self.assertIs(broker._observation_failure_owner, observation)
                self.assertEqual(handle.wait(0).state, RequestState.SUCCEEDED)
                # A release-line interrupt may strand the private nonblocking
                # gate. Clean up only in this isolated fault-injection test.
                if state.lock.locked():
                    state.lock.release()
                broker._thread.join(3)

    def test_callback_failure_survives_a_concurrent_entry_interruption(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        entered, resume = threading.Event(), threading.Event()
        references, errors = [], []

        def fail_owned_callback(*args):
            private_tensor = torch.tensor([42.0])
            references.append(weakref.ref(private_tensor))
            entered.set()
            if not resume.wait(3):
                raise RuntimeError('callback test timed out')
            raise KeyboardInterrupt('private callback failure')

        def wait():
            try:
                handle.wait(3)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=wait)
        try:
            with patch.object(observation, '_captured_handle', side_effect=fail_owned_callback):
                thread.start()
                self.assertTrue(entered.wait(3))
                with patch.object(observation, '_callback_bounded_inner',
                                  side_effect=KeyboardInterrupt('private gate entry')):
                    with self.assertRaises(KeyboardInterrupt):
                        handle.wait(0)
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsNone(errors[0].__context__)
            state = observation._terminal_capture
            self.assertIsNotNone(state.failure)
            self.assertIsNotNone(state.interruption)
            self.assertIs(state.handle, handle)
            self.assertIs(state.outcome, handle._outcome)
            errors.clear()
            gc.collect()
            self.assertIsNotNone(references[0]())
            self.assertTrue(authority.unsafe)
            self.assertIs(broker._observation_failure_owner, observation)
        finally:
            resume.set()
            if thread.ident is not None:
                thread.join(3)
            broker._thread.join(3)

    def test_many_registration_entry_interrupts_use_one_preallocated_failure_root(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        count = 23
        barrier = threading.Barrier(count, timeout=30)
        arrived = threading.Condition()
        arrivals, errors = [], []

        def interrupt_entry(*args, **kwargs):
            with arrived:
                arrivals.append(True)
                arrived.notify_all()
            barrier.wait()
            raise KeyboardInterrupt('private registration entry')

        def submit(index):
            try:
                broker.submit('team', ForwardRequest(
                    f'job-{index}', {'input_ids': torch.tensor([[1, 2]])}))
            except BaseException as exc:
                errors.append(exc)

        threads = []
        with patch.object(observation, '_callback_bounded_inner', side_effect=interrupt_entry):
            for index in range(count):
                thread = threading.Thread(target=submit, args=(index,))
                threads.append(thread)
                thread.start()
                with arrived:
                    self.assertTrue(arrived.wait_for(lambda: len(arrivals) > index, 5))
                if index == count - 1:
                    continue
                deadline = perf_counter_ns() + 5_000_000_000
                while True:
                    with broker._condition:
                        if broker._running is None and broker._pending_slots == 0:
                            break
                        broker._condition.wait(0.02)
                    self.assertLess(perf_counter_ns(), deadline,
                                    f'job {index}, running={broker._running}, pending={broker._pending_slots}, '
                                    f'unsafe={authority.unsafe}, broker_alive={broker._thread.is_alive()}')
            for thread in threads:
                thread.join(12)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), count)
        self.assertTrue(all(type(exc) is KeyboardInterrupt and exc.__context__ is None for exc in errors))
        self.assertTrue(authority.unsafe)
        self.assertIs(broker._observation_failure_owner, observation)
        broker._thread.join(3)

    def test_many_registration_identity_interrupts_share_one_observer_root(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        model.release.set()
        count = 23
        barrier, arrived = threading.Barrier(count, timeout=30), threading.Condition()
        arrivals, errors = [], []
        original = broker._service_observer

        def interrupted_identity():
            if threading.current_thread() is broker._thread:
                return original()
            with arrived:
                arrivals.append(True)
                arrived.notify_all()
            barrier.wait()
            raise KeyboardInterrupt('private identity check')

        def submit(index):
            try:
                broker.submit('team', ForwardRequest(
                    f'job-{index}', {'input_ids': torch.tensor([[1, 2]])}))
            except BaseException as exc:
                errors.append(exc)

        threads = []
        with patch.object(broker, '_service_observer', side_effect=interrupted_identity):
            for index in range(count):
                thread = threading.Thread(target=submit, args=(index,))
                threads.append(thread)
                thread.start()
                with arrived:
                    self.assertTrue(arrived.wait_for(lambda: len(arrivals) > index, 5))
                if index == count - 1:
                    continue
                deadline = perf_counter_ns() + 5_000_000_000
                while True:
                    with broker._condition:
                        if broker._running is None and broker._pending_slots == 0:
                            break
                        broker._condition.wait(0.02)
                    self.assertLess(perf_counter_ns(), deadline)
            for thread in threads:
                thread.join(12)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), count)
        self.assertTrue(all(type(exc) is KeyboardInterrupt and exc.__context__ is None for exc in errors))
        self.assertTrue(authority.unsafe)
        self.assertTrue(observation._faulted)
        self.assertIs(broker._observation_failure_owner, observation)
        broker._thread.join(3)

    def test_service_binding_requires_prestart_per_team_authentication(self):
        for shared_token in (True, False):
            with self.subTest(shared_token=shared_token):
                _, worker, broker, authority, observation, _, _ = service_stack(bind=False)
                server = IPCServer(broker, team_ids=None if shared_token else ('team-a',),
                                   max_batch=1, max_tokens=2)
                if not shared_token:
                    server.start()
                try:
                    with self.assertRaisesRegex(ValueError, 'authenticated listener'):
                        observation.bind_service(worker, broker, server)
                    self.assertFalse(authority.unsafe)
                    self.assertIsNone(broker._serial_observation)
                finally:
                    server.close(timeout=3)
                    broker.close(timeout=3)

    def test_wrong_running_authority_binding_fails_closed_before_model_call(self):
        model, worker, broker, authority, observation, _, _ = service_stack()
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})

        def run_with_mutated_binding(req, cancel_event=None):
            authority._broker = object()
            return observation._execute(worker, broker._running, req, cancel_event)

        with patch.object(worker, 'execute', side_effect=run_with_mutated_binding):
            handle = broker.submit('team', request)
            try:
                outcome = handle.wait(3)
            except BrokerUnavailable:
                outcome = None
            broker._thread.join(3)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertTrue(outcome is None or outcome.state is RequestState.FAILED)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_missing_forward_scope_fails_closed_before_model_call(self):
        model, _, broker, authority, observation, _, _ = service_stack()
        with patch.object(observation, '_start_serial', return_value=None):
            handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
            try:
                outcome = handle.wait(3)
            except BrokerUnavailable:
                outcome = None
            broker._thread.join(3)
        self.assertTrue(authority.unsafe)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(outcome is None or outcome.state is RequestState.FAILED)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_real_ipc_polling_preserves_source_events_and_fixed_worker(self):
        model, worker, broker, authority, observation, _, guard_times = service_stack(bind=False)
        server = IPCServer(broker, team_ids=('team-a',), max_batch=1, max_tokens=2)
        observation.bind_service(worker, broker, server)
        polled_twice = threading.Event()
        polls = []
        original_wait = RequestHandle.wait

        def track_poll(handle, timeout=None):
            if timeout == 0:
                polls.append(True)
                if len(polls) >= 2:
                    polled_twice.set()
            return original_wait(handle, timeout)

        server.start()
        try:
            client = IPCClient(server.address, server.team_tokens['team-a'])
            with patch.object(RequestHandle, 'wait', new=track_poll):
                handle = client.submit(team_id='team-a', request_id='job-a',
                                       inputs={'input_ids': [[1, 2]]}, seed=17)
                self.assertTrue(model.entered.wait(3))
                self.assertTrue(polled_twice.wait(3))
                model.release.set()
                response = handle.wait(3)
            self.assertEqual(response['status'], 'succeeded')
            self.assertEqual(response['result']['logits'], [1.0, 2.0])
            self.assertIs(broker._worker, worker)
            self.assertFalse(authority.unsafe)
        finally:
            model.release.set()
            self.assertTrue(server.close(timeout=3))
            broker.close(timeout=3)
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        events = report['events']
        ready = next(e for e in events if e['kind'] == 'ready')
        selected = next(e for e in events if e['kind'] == 'selected')
        entry = next(e for e in events if e['kind'] == 'forward_entry')
        finished = next(e for e in events if e['kind'] == 'forward_finished')
        terminal = next(e for e in events if e['kind'] == 'broker_terminal')
        self.assertEqual((ready['broker_id'], ready['broker_job_id']), (broker.broker_id, handle.job_id))
        self.assertEqual((ready['time_ns'], selected['time_ns'], terminal['time_ns']),
                         (response['ready_ns'], response['started_ns'], response['finished_ns']))
        self.assertEqual(entry['profile_sha256'], 'c' * 64)
        self.assertEqual(entry['input_shape'], [1, 2])
        self.assertEqual(entry['call_id'], f'lg-{broker.broker_id}-{handle.job_id}')
        self.assertEqual(entry['members'], [{'job_id': 'job-a', 'team_id': 'team-a',
                                            'broker_id': broker.broker_id,
                                            'broker_job_id': handle.job_id, 'row_offset': 0}])
        model_time, shape = model.calls[0]
        self.assertEqual(shape, (1, 2))
        self.assertLessEqual(entry['time_ns'], model_time)
        self.assertTrue(any(entry['time_ns'] <= stamp <= model_time for stamp in guard_times))
        self.assertLessEqual(model_time, finished['time_ns'])
        self.assertLessEqual(finished['time_ns'], terminal['time_ns'])

    def test_pending_direct_cancel_has_terminal_observation_without_a_call(self):
        model, _, broker, _, observation, _, _ = service_stack()
        first = broker.submit('team-a', ForwardRequest('job-a', {'input_ids': torch.tensor([[1, 2]])}))
        try:
            self.assertTrue(model.entered.wait(3))
            second = broker.submit('team-b', ForwardRequest('job-b', {'input_ids': torch.tensor([[3, 4]])}))
            self.assertTrue(second.cancel())
            self.assertEqual(second.wait(3).state, RequestState.CANCELLED)
        finally:
            model.release.set()
        self.assertEqual(first.wait(3).state, RequestState.SUCCEEDED)
        broker.close(timeout=3)
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        member_jobs = [m['job_id'] for e in report['events'] if e['kind'] == 'forward_entry'
                       for m in e['members']]
        self.assertEqual(member_jobs, ['job-a'])
        kinds = [e['kind'] for e in report['events'] if e.get('job_id') == 'job-b']
        self.assertEqual(kinds, ['ready', 'broker_terminal'])

    def test_nonterminal_direct_poll_does_not_invalidate_later_capture(self):
        model, _, broker, _, observation, _, _ = service_stack()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        try:
            self.assertTrue(model.entered.wait(3))
            with self.assertRaises(TimeoutError):
                observation.wait(handle, 0)
            model.release.set()
            self.assertEqual(observation.wait(handle, 3).state, RequestState.SUCCEEDED)
        finally:
            model.release.set()
            broker.close(timeout=3)
        self.assertTrue(observation.seal(producers_stopped=True)['events_complete'])

    def test_window_end_with_unresolved_registered_request_invalidates_report(self):
        model, _, broker, _, observation, _, _ = service_stack()
        handle = broker.submit('team', ForwardRequest('job', {'input_ids': torch.tensor([[1, 2]])}))
        try:
            self.assertTrue(model.entered.wait(3))
            with self.assertRaises(TimeoutError):
                observation.wait(handle, 0)
            observation.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
            model.release.set()
            self.assertEqual(observation.wait(handle, 3).state, RequestState.SUCCEEDED)
        finally:
            model.release.set()
            broker.close(timeout=3)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_two_team_ipc_pending_cancel_keeps_owner_and_peer_results_separate(self):
        model, worker, broker, authority, observation, _, _ = service_stack(bind=False)
        server = IPCServer(broker, team_ids=('team-a', 'team-b'), max_batch=1, max_tokens=2)
        observation.bind_service(worker, broker, server)
        server.start()
        try:
            first = IPCClient(server.address, server.team_tokens['team-a']).submit(
                team_id='team-a', request_id='job-a', inputs={'input_ids': [[1, 2]]}, seed=17)
            self.assertTrue(model.entered.wait(3))
            second = IPCClient(server.address, server.team_tokens['team-b']).submit(
                team_id='team-b', request_id='job-b', inputs={'input_ids': [[3, 4]]}, seed=17)
            second.cancel()
            cancelled = second.wait(3)
            self.assertEqual(cancelled['status'], 'cancelled')
            self.assertTrue(second.cancel_acknowledged)
            self.assertEqual((cancelled['team_id'], cancelled['job_id']), ('team-b', second.job_id))
            model.release.set()
            success = first.wait(3)
            self.assertEqual((success['status'], success['team_id'], success['job_id']),
                             ('succeeded', 'team-a', first.job_id))
            self.assertNotEqual(first.job_id, second.job_id)
            self.assertFalse(authority.unsafe)
        finally:
            model.release.set()
            self.assertTrue(server.close(timeout=3))
            broker.close(timeout=3)
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        events = report['events']
        self.assertEqual([m['job_id'] for e in events if e['kind'] == 'forward_entry'
                          for m in e['members']], ['job-a'])
        second_events = [event for event in events if event.get('job_id') == 'job-b']
        self.assertEqual([event['kind'] for event in second_events], ['ready', 'broker_terminal'])

    def test_ipc_recorder_failure_preserves_real_result_and_invalidates_report(self):
        model, worker, broker, authority, observation, recorder, _ = service_stack(bind=False)
        model.release.set()
        server = IPCServer(broker, team_ids=('team-a',), max_batch=1, max_tokens=2)
        observation.bind_service(worker, broker, server)
        original = recorder.record

        def fail_entry(event):
            if event['kind'] == 'forward_entry':
                raise MemoryError('observation allocation failed')
            return original(event)

        server.start()
        try:
            with patch.object(recorder, 'record', side_effect=fail_entry):
                client = IPCClient(server.address, server.team_tokens['team-a'])
                handle = client.submit(team_id='team-a', request_id='job-a',
                                       inputs={'input_ids': [[1, 2]]}, seed=17)
                response = handle.wait(3)
            self.assertEqual(response['status'], 'succeeded')
            self.assertEqual(response['result']['logits'], [1.0, 2.0])
            self.assertFalse(authority.unsafe)
        finally:
            self.assertTrue(server.close(timeout=3))
            broker.close(timeout=3)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_window_end_during_ipc_capture_invalidates_the_recording(self):
        model, worker, broker, _, observation, _, _ = service_stack(bind=False)
        model.release.set()
        server = IPCServer(broker, team_ids=('team-a',), max_batch=1, max_tokens=2)
        observation.bind_service(worker, broker, server)
        copying, resume = threading.Event(), threading.Event()
        original = observation._capture

        def blocked_capture(handle, outcome):
            copying.set()
            if not resume.wait(3):
                raise RuntimeError('capture test timed out')
            return original(handle, outcome)

        server.start()
        try:
            with patch.object(observation, '_capture', side_effect=blocked_capture):
                client = IPCClient(server.address, server.team_tokens['team-a'])
                handle = client.submit(team_id='team-a', request_id='job-a',
                                       inputs={'input_ids': [[1, 2]]}, seed=17)
                self.assertTrue(copying.wait(3))
                observation.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
                resume.set()
                self.assertEqual(handle.wait(3)['status'], 'succeeded')
        finally:
            resume.set()
            self.assertTrue(server.close(timeout=3))
            broker.close(timeout=3)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_session_guard_failure_after_entry_suppresses_model_call(self):
        from looking_glass.grouped_observation import _local

        def guard():
            scope = getattr(_local, 'forward', None)
            if scope is not None and scope.entry_ns is not None:
                raise RuntimeError('simulated session revocation')

        model, _, broker, authority, observation, _, _ = service_stack(device_guard=guard)
        handle = broker.submit('team-a', ForwardRequest('job-a', {'input_ids': torch.tensor([[1, 2]])}))
        try:
            outcome = handle.wait(3)
        except BrokerUnavailable:
            outcome = None
        broker._thread.join(3)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(outcome is None or outcome.state is RequestState.FAILED)
        report = observation.seal(producers_stopped=True)
        # The recorder can completely capture an uncertain invocation. That
        # is still not a completed model call or successful timing evidence.
        self.assertTrue(any(event['kind'] == 'forward_uncertain' for event in report['events']))
        self.assertNotIn('forward_finished', [event['kind'] for event in report['events']])

    def test_concurrent_terminal_waits_preserve_result_and_flag_capture_race(self):
        model, _, broker, _, observation, recorder, _ = service_stack()
        model.release.set()
        handle = broker.submit('team-a', ForwardRequest('job-a', {'input_ids': torch.tensor([[1, 2]])}))
        with broker._condition:
            self.assertTrue(broker._condition.wait_for(lambda: handle._outcome is not None, 3))
        copying, resume, second_done = threading.Event(), threading.Event(), threading.Event()
        original = recorder.record
        outcomes, errors = [], []

        def block_terminal(event):
            if event['kind'] == 'broker_terminal':
                copying.set()
                if not resume.wait(3):
                    raise RuntimeError('capture test timed out')
            return original(event)

        def wait(done=None):
            try:
                outcomes.append(handle.wait(3))
            except BaseException as exc:
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        first = threading.Thread(target=wait)
        second = threading.Thread(target=wait, args=(second_done,))
        try:
            with patch.object(recorder, 'record', side_effect=block_terminal):
                first.start()
                self.assertTrue(copying.wait(3))
                second.start()
                self.assertTrue(second_done.wait(3))
                resume.set()
                first.join(3)
                second.join(3)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual([o.state for o in outcomes], [RequestState.SUCCEEDED] * 2)
            self.assertIs(outcomes[0], outcomes[1])
        finally:
            resume.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
            broker.close(timeout=3)
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])


if __name__ == '__main__':
    unittest.main()
