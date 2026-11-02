"""CPU tests for the opt-in session-bound grouped workspace owner."""

import importlib
import gc
import sys
import threading
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from looking_glass.budgeted_worker import DeviceQuarantined, WorkloadProfile
from looking_glass.grouped_worker import GroupPlan, GroupedWorker
from looking_glass.host_budget import HostBudget, HostBusy
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.microbatch import BatchProfile, ForwardContract, Microbatch, MicrobatchRequest
from looking_glass.owner_completion import OwnerCompletionAuthority
from looking_glass.resident_worker import ForwardRequest, RequestCancelled, ResidentWorker


MODEL_ID = 'a' * 64


class RowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(tuple(input_ids.shape))
        values = input_ids.float()
        return {'logits': torch.stack((values, values + 1), dim=-1)}


def make_grouped_owner(*, model=None, device_guard=None):
    try:
        module = importlib.import_module('looking_glass.grouped_owner_worker')
    except ModuleNotFoundError as exc:
        if exc.name != 'looking_glass.grouped_owner_worker':
            raise
        raise AssertionError('session-bound grouped workspace owner is missing') from None
    model = (RowModel() if model is None else model).eval()
    resident = ResidentWorker(MODEL_ID, model, device='cpu', max_batch_size=2,
                              max_sequence_length=3, max_input_bytes=96, max_output_bytes=256)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    resident_lease = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace_lease = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    inputs = (('input_ids', 'torch.int64'),)
    serial = WorkloadProfile('serial', MODEL_ID, 'cpu', (1, 3), inputs, 256, 1024)
    combined = WorkloadProfile('combined', MODEL_ID, 'cpu', (2, 3), inputs, 256, 1024)
    authority = OwnerCompletionAuthority(device_guard=device_guard)
    worker = module.GroupedWorkspaceWorker(resident, budget, resident_lease, workspace_lease,
                                           profiles=(serial, combined), authority=authority, seed=17)
    contract = ForwardContract(MODEL_ID, 'checkpoint', 'adapter', 'cpu', torch.float32,
                               runtime_id='cpu-test-runtime', seed=17, deterministic=True,
                               batch_separable=True, mode='eval', rng_policy='none',
                               cache_policy='disabled', mutable_state_policy='none',
                               tracing=False, intervention=False)
    profile = BatchProfile('batch', 'cpu-cert', contract, (('input_ids', torch.int64),), (1, 3),
                           torch.float32, 2, 0., 0., (2,), 2, 1, 2, 6, 48, 48, 96)
    jobs = tuple(MicrobatchRequest(f'job-{i}', f'team-{i}', f'req-{i}', contract,
                                   {'input_ids': torch.tensor([[i, i + 1, i + 2]])})
                 for i in (1, 4))
    host = HostBudget(max_bytes=8192, max_items=128, max_requests=8, max_operations=8)
    request = host.open_request('group-request')
    grouped = GroupedWorker(worker, (profile,), host_budget=host,
                            finite_scratch_bytes={'batch': 256}, finite_scratch_items={'batch': 4})
    needed_bytes, needed_items = grouped.required_host_capacity('batch')
    claim = request.reserve('broker', 'group-envelope', needed_bytes, items=needed_items)
    return grouped, worker, model, budget, resident_lease, workspace_lease, authority, host, request, claim, jobs


def make_grouped_broker(*, observation=None, device_guard=None):
    test_broker = importlib.import_module('test_grouped_broker')
    module = importlib.import_module('looking_glass.grouped_owner_worker')
    owner = []

    def construct(resident, gate, profiles):
        budget = gate._budget
        resident_lease = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        workspace = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        combined = profiles[0]
        serial = replace(combined, profile_id='serial', shape=(1, 3))
        authority = OwnerCompletionAuthority(device_guard=device_guard)
        worker = module.GroupedWorkspaceWorker(resident, budget, resident_lease, workspace,
                                               profiles=(serial, combined), authority=authority, seed=17)
        owner.append((worker, budget, resident_lease, workspace, authority))
        return worker

    with patch.object(test_broker, 'BudgetedWorker', side_effect=construct):
        broker, model, host = test_broker.fixture(observation=observation)
    return broker, model, host, owner[0], test_broker.request


class GroupedOwnerWorkerTests(unittest.TestCase):
    def test_broker_binding_requires_control_result_custody_to_be_transferred(self):
        from looking_glass.grouped_broker import GroupedBroker
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, _, (worker, _, _, _, authority), make_request = make_grouped_broker()
        result = worker.execute_control(make_request('control', [1, 2, 3]))
        with self.assertRaises(HostBusy):
            GroupedBroker(*captured['args'], **captured['kwargs'])
        self.assertIsNone(authority._broker)
        self.assertFalse(authority.unsafe)
        worker.confirm_control_result(result)
        broker = GroupedBroker(*captured['args'], **captured['kwargs'])
        self.assertIs(authority._broker, broker)
        self.assertFalse(authority.unsafe)

    def test_interrupted_binding_publication_quarantines_and_roots_partial_broker(self):
        import inspect
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, host, (worker, budget, resident_lease, workspace, authority), _ = make_grouped_broker()
        lines, first = inspect.getsourcelines(GroupedBroker._initialize)
        target = first + next(i for i, line in enumerate(lines)
                              if line.strip() == 'host_budget._grouped_broker_owner = self')
        fired = []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._initialize.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                fired.append(True)
                raise KeyboardInterrupt('binding handoff interrupted')
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                GroupedBroker(*captured['args'], **captured['kwargs'])
        finally:
            sys.settrace(None)
        self.assertTrue(fired)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._broker_construction_failure, KeyboardInterrupt)
        self.assertIsInstance(authority._broker, GroupedBroker)
        self.assertIsNone(getattr(host, '_grouped_broker_owner', None))
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_uncertain_preflight_acquire_quarantines_without_guessing_lock_ownership(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, _, (worker, _, _, _, authority), _ = make_grouped_broker()

        class InterruptedLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.releases = 0

            def acquire(self, **kwargs):
                self.lock.acquire(**kwargs)
                raise KeyboardInterrupt('acquire result interrupted')

            def release(self):
                self.releases += 1
                self.lock.release()

        gate = worker._preflight_lock = InterruptedLock()
        with self.assertRaises(GroupedBrokerUnavailable) as caught:
            GroupedBroker(*captured['args'], **captured['kwargs'])
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertTrue(gate.lock.locked())
        self.assertEqual(gate.releases, 0)
        self.assertIsInstance(worker._broker_construction_failure, KeyboardInterrupt)
        gate.lock.release()

    def _closed_grouped_owner(self):
        broker, model, host, owner, make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = ticket.wait(1)
            ticket.confirm(result)
            del result
            ticket.release()
            handle.dispose()
        broker.close()
        worker = owner[0]
        worker.close()
        worker._worker.close()
        return broker, model, host, owner

    def test_grouped_teardown_refuses_an_active_execution_gate(self):
        from looking_glass.grouped_owner_worker import retire_grouped_owner
        broker, _, _, (worker, budget, resident_lease, workspace, authority) = self._closed_grouped_owner()
        self.assertTrue(worker._preflight_lock.acquire(blocking=False))
        try:
            with self.assertRaises(HostBusy):
                retire_grouped_owner(worker, broker, callers_quiescent=True)
        finally:
            worker._preflight_lock.release()
        self.assertIs(authority._worker, worker)
        self.assertIs(authority._broker, broker)
        self.assertFalse(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_grouped_teardown_does_not_treat_guard_failure_as_checked_busy(self):
        from looking_glass.grouped_owner_worker import retire_grouped_owner
        from looking_glass.resident_worker import WorkerUnavailable
        broker, _, _, (worker, budget, resident_lease, workspace, authority) = self._closed_grouped_owner()
        with patch.object(authority, '_before_device', side_effect=HostBusy('guard failed')):
            with self.assertRaises(WorkerUnavailable) as caught:
                retire_grouped_owner(worker, broker, callers_quiescent=True)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._teardown_failure, HostBusy)
        self.assertIs(authority._broker, broker)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupted_grouped_teardown_lock_release_marks_unsafe(self):
        import inspect
        from looking_glass.grouped_owner_worker import retire_grouped_owner
        from looking_glass.resident_worker import WorkerUnavailable
        broker, model, _, (worker, budget, resident_lease, workspace, authority) = self._closed_grouped_owner()
        lines, first = inspect.getsourcelines(retire_grouped_owner)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'lock.release()')
        fired = []

        def trace(frame, event, arg):
            if frame.f_code is retire_grouped_owner.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                fired.append(True)
                raise KeyboardInterrupt('teardown lock release interrupted')
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(WorkerUnavailable) as caught:
                retire_grouped_owner(worker, broker, callers_quiescent=True)
        finally:
            sys.settrace(None)
        self.assertTrue(fired)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIs(worker._teardown_broker, broker)
        self.assertIs(worker._model, model)
        self.assertIsInstance(worker._teardown_notification_failure, KeyboardInterrupt)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupted_grouped_teardown_retains_roots_and_marks_unsafe(self):
        import inspect
        from looking_glass.grouped_owner_worker import retire_grouped_owner
        from looking_glass.resident_worker import WorkerUnavailable
        broker, model, _, (worker, budget, resident_lease, workspace, authority) = self._closed_grouped_owner()
        lines, first = inspect.getsourcelines(retire_grouped_owner)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'authority._broker = None')
        fired = []

        def trace(frame, event, arg):
            if frame.f_code is retire_grouped_owner.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                fired.append(True)
                raise KeyboardInterrupt('teardown interrupted')
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(WorkerUnavailable) as caught:
                retire_grouped_owner(worker, broker, callers_quiescent=True)
        finally:
            sys.settrace(None)
        self.assertTrue(fired)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIs(worker._teardown_broker, broker)
        self.assertIs(worker._model, model)
        self.assertIsInstance(worker._teardown_failure, KeyboardInterrupt)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_quiescent_grouped_teardown_detaches_exact_binding_but_keeps_device_leases(self):
        from looking_glass.grouped_owner_worker import retire_grouped_owner
        broker, model, host, (worker, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = ticket.wait(1)
            ticket.confirm(result)
            del result
            ticket.release()
            handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        worker.close()
        worker._worker.close()
        self.assertTrue(retire_grouped_owner(worker, broker, callers_quiescent=True))
        self.assertIsNone(authority._broker)
        self.assertIsNone(authority._worker)
        self.assertTrue(authority._worker_retired)
        self.assertIs(worker._teardown_broker, broker)
        self.assertIs(worker._model, model)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertFalse(authority.unsafe)
        self.assertFalse(retire_grouped_owner(worker, broker, callers_quiescent=True))

    def test_broker_constructor_does_not_confuse_internal_value_error_with_rejection(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, _, (worker, _, _, _, authority), _ = make_grouped_broker()
        with patch.object(GroupedWorker, 'required_host_capacity', side_effect=ValueError('internal fault')):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                GroupedBroker(*captured['args'], **captured['kwargs'])
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._broker_construction_failure, ValueError)

    def test_broker_constructor_rejects_active_control_before_callbacks(self):
        from looking_glass.grouped_broker import GroupedBroker
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, model, _, (worker, _, _, _, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        errors, results = [], []

        def control():
            try:
                results.append(worker.execute_control(make_request('control', [1, 2, 3])))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=control)
        thread.start()
        self.assertTrue(model.entered.wait(5))
        try:
            options = dict(captured['kwargs'])
            options['collect_runtime'] = unittest.mock.Mock(side_effect=MemoryError('runtime'))
            with patch.object(GroupedWorker, 'required_host_capacity', side_effect=MemoryError('capacity preflight')) as capacity:
                with self.assertRaises(HostBusy):
                    GroupedBroker(*captured['args'], **options)
            capacity.assert_not_called()
            options['collect_runtime'].assert_not_called()
            self.assertFalse(authority.unsafe)
            self.assertIsNone(authority._broker)
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        worker.confirm_control_result(results[0])

    def test_interrupted_constructor_setup_stops_an_active_control(self):
        import inspect
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, model, _, (worker, _, _, _, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        errors = []

        def control():
            try:
                worker.execute_control(make_request('control', [1, 2, 3]))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=control)
        thread.start()
        self.assertTrue(model.entered.wait(5))
        lines, first = inspect.getsourcelines(GroupedBroker.__init__)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'self._binding_busy = False')

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker.__init__.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                raise KeyboardInterrupt('construction setup interrupted')
            return trace

        try:
            sys.settrace(trace)
            try:
                with self.assertRaises(GroupedBrokerUnavailable) as caught:
                    GroupedBroker(*captured['args'], **captured['kwargs'])
            finally:
                sys.settrace(None)
            self.assertIsNone(caught.exception.__context__)
            self.assertTrue(authority.unsafe)
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)

    def test_broker_constructor_preflight_memory_error_is_rooted_and_quarantines(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, _, (worker, _, _, _, authority), _ = make_grouped_broker()
        with patch.object(GroupedWorker, 'required_host_capacity', side_effect=MemoryError('capacity preflight')):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                GroupedBroker(*captured['args'], **captured['kwargs'])
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._broker_construction_failure, MemoryError)

    def test_runtime_callback_cannot_spoof_a_retryable_binding_rejection(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        from looking_glass.grouped_owner_worker import _GroupedBindingBusy
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, _, (worker, _, _, _, authority), _ = make_grouped_broker()

        def fail_runtime():
            raise _GroupedBindingBusy('callback failure')

        options = dict(captured['kwargs'])
        options['collect_runtime'] = fail_runtime
        with self.assertRaises(GroupedBrokerUnavailable) as caught:
            GroupedBroker(*captured['args'], **options)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._broker_construction_failure, _GroupedBindingBusy)

    def test_broker_constructor_runtime_failure_is_rooted_and_quarantines(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        captured = {}

        def capture(instance, *args, **kwargs):
            captured['args'], captured['kwargs'] = args, kwargs

        with patch.object(GroupedBroker, '__init__', capture):
            _, _, host, (worker, budget, resident_lease, workspace, authority), _ = make_grouped_broker()
        owners = []

        class Payload:
            pass

        def failed_runtime():
            payload = Payload()
            owners.append(weakref.ref(payload))
            raise MemoryError('constructor runtime failure')

        options = dict(captured['kwargs'])
        options['collect_runtime'] = failed_runtime
        with self.assertRaises(GroupedBrokerUnavailable) as caught:
            GroupedBroker(*captured['args'], **options)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        gc.collect()
        self.assertIsNotNone(owners[0]())
        self.assertIsInstance(worker._broker_construction_failure, MemoryError)
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_session_bound_owner_rejects_unreviewed_timing_observation(self):
        from looking_glass.batched_recorder import BatchedEventRecorder
        from looking_glass.grouped_observation import GroupedObservation
        observation = GroupedObservation(BatchedEventRecorder())
        with self.assertRaisesRegex(ValueError, 'observation'):
            make_grouped_broker(observation=observation)

    def test_interrupted_registration_handoff_stops_active_cohort_publication(self):
        import inspect
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, model, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        errors = []

        def dispatch():
            try:
                broker.run_once(2)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        self.assertTrue(model.entered.wait(5))
        lines, first = inspect.getsourcelines(GroupedBroker.submit)
        target = first + next(i for i, line in enumerate(lines) if line.strip() == 'ready_ns = None')

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker.submit.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                raise KeyboardInterrupt('registration handoff interrupted')
            return trace

        try:
            sys.settrace(trace)
            try:
                with self.assertRaises(GroupedBrokerUnavailable) as caught:
                    broker.submit('team-c', make_request('c', [7, 8, 9]))
            finally:
                sys.settrace(None)
            self.assertIsNone(caught.exception.__context__)
            self.assertTrue(authority.unsafe)
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertIsNotNone(broker.lookup('team-c', 'c'))

    def test_pre_admission_policy_failure_stops_active_cohort_publication(self):
        from looking_glass.certified_batch_policy import CertifiedBatchPolicy
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        errors = []

        def dispatch():
            try:
                broker.run_once(2)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        self.assertTrue(model.entered.wait(5))
        try:
            with patch.object(CertifiedBatchPolicy, 'require_request', side_effect=MemoryError('policy failure')):
                with self.assertRaises(GroupedBrokerUnavailable) as caught:
                    broker.submit('team-c', make_request('c', [7, 8, 9]))
            self.assertIsNone(caught.exception.__context__)
            self.assertIsInstance(broker._admission_failure, MemoryError)
            self.assertTrue(authority.unsafe)
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertTrue(all(group.state._state.__class__.__name__ != 'CohortPublication'
                            for group in broker._groups.values()))

    def test_live_runtime_failure_quarantines_bound_authority_before_submission(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        with patch.object(broker, '_collect_runtime', side_effect=RuntimeError('owner check failed')):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                broker.submit('team-a', make_request('a', [1, 2, 3]))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertEqual(model.calls, [])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupt_during_health_failure_still_marks_shared_owner(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, host, (_, budget, resident_lease, workspace, authority), _ = make_grouped_broker()
        interrupted = threading.Event()

        def read_health():
            raise MemoryError('device health unreadable')

        def trace(frame, event, arg):
            if frame.f_code is read_health.__code__ and event == 'call':
                sys.settrace(None)
                interrupted.set()
                raise KeyboardInterrupt('health notification interrupted')
            return trace

        with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=read_health):
            sys.settrace(trace)
            try:
                self.assertTrue(broker.quarantined)
            finally:
                sys.settrace(None)
        self.assertTrue(interrupted.is_set())
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_healthy_publication_is_unavailable_without_terminal_quarantine(self):
        import inspect
        from looking_glass.grouped_broker_state import CohortState
        broker, _, _, (_, _, _, _, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        lines, first = inspect.getsourcelines(CohortState._publish)
        target = first + next(i for i, line in enumerate(lines) if line.strip().startswith('if (authority.unsafe'))
        entered, resume = threading.Event(), threading.Event()
        errors = []

        def dispatch():
            def trace(frame, event, arg):
                if frame.f_code is CohortState._publish.__code__ and event == 'line' and frame.f_lineno == target:
                    entered.set()
                    if not resume.wait(5):
                        raise RuntimeError('publication timed out')
                return trace
            sys.settrace(trace)
            try:
                broker.run_once(2)
            except BaseException as exc:
                errors.append(exc)
            finally:
                sys.settrace(None)

        thread = threading.Thread(target=dispatch)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(authority.unsafe)
            self.assertFalse(broker.quarantined)
            self.assertFalse(broker.failed)
            self.assertTrue(broker.unavailable)
        finally:
            resume.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(broker.quarantined)
        self.assertFalse(broker.unavailable)

    def test_success_publication_waits_for_pending_health_read(self):
        from looking_glass.grouped_broker_state import CohortState, CohortPublication
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, _, (_, _, _, _, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        validating, continue_validation, reading, continue_health, done = (threading.Event() for _ in range(5))
        errors = []
        original = CohortState._validate_outcome

        def validate(state, outcome):
            original(state, outcome)
            validating.set()
            if not continue_validation.wait(5):
                raise RuntimeError('validation timed out')

        def slow_health():
            reading.set()
            if not continue_health.wait(5):
                raise RuntimeError('health timed out')
            return False

        def dispatch():
            try:
                broker.run_once(2)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(CohortState, '_validate_outcome', validate):
            thread = threading.Thread(target=dispatch)
            thread.start()
            self.assertTrue(validating.wait(5))
            with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                              side_effect=slow_health):
                reader = threading.Thread(target=lambda: broker.quarantined)
                reader.start()
                self.assertTrue(reading.wait(5))
                try:
                    continue_validation.set()
                    self.assertFalse(done.wait(.2))
                    self.assertTrue(all(type(group.state._state) is not CohortPublication
                                        for group in broker._groups.values()))
                    self.assertFalse(authority.unsafe)
                finally:
                    continue_health.set()
                    reader.join(5)
                    thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(all(type(group.state._state) is CohortPublication for group in broker._groups.values()))

    def test_interrupted_retry_cleanup_retains_tensor_custody(self):
        import inspect
        from opcode_support import offsets_at_line, run_opcode_trace
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        lines, first_line = inspect.getsourcelines(GroupedBroker._wait)
        target_line = first_line + next(i for i, line in enumerate(lines)
                                        if line.strip() == 'result = publication = member = None')
        offsets = offsets_at_line(GroupedBroker._wait, target_line, 'STORE_FAST')
        self.assertEqual(len(offsets), 3)
        for offset in offsets:
            for error_type in (HostBusy, TimeoutError, GroupedBrokerUnavailable):
                with self.subTest(offset=offset, error=error_type.__name__):
                    broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
                    first = broker.submit('team-a', make_request('a', [1, 2, 3]))
                    broker.submit('team-b', make_request('b', [4, 5, 6]))
                    broker.run_once(1)
                    ticket = first.delivery()
                    before = host.snapshot().used_bytes
                    tensor_ref = weakref.ref(first._job.group.state._state.members[0].logits)
                    entered, resume = threading.Event(), threading.Event()
                    fired = []

                    def slow_health():
                        entered.set()
                        if not resume.wait(5):
                            raise RuntimeError('health timed out')
                        return False

                    def trace(frame, event, arg):
                        if frame.f_code is GroupedBroker._wait.__code__:
                            if event == 'call':
                                frame.f_trace_opcodes = True
                            elif event == 'opcode' and frame.f_lasti == offset:
                                fired.append(True)
                                raise error_type('synthetic cleanup interruption')
                        return trace

                    with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                                      side_effect=slow_health):
                        reader = threading.Thread(target=lambda: broker.quarantined)
                        reader.start()
                        self.assertTrue(entered.wait(5))
                        try:
                            with self.assertRaises(GroupedBrokerUnavailable):
                                run_opcode_trace(trace, lambda: ticket.wait(1))
                        finally:
                            resume.set()
                            reader.join(5)
                    self.assertTrue(fired)
                    self.assertTrue(authority.unsafe)
                    self.assertEqual(host.snapshot().used_bytes, before)
                    self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
                    gc.collect()
                    self.assertIsNotNone(tensor_ref())
                    with self.assertRaises(GroupedBrokerUnavailable):
                        ticket.release()

    def test_success_delivery_retries_while_health_is_indeterminate(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        entered, resume = threading.Event(), threading.Event()

        def slow_health():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('health timed out')
            return False

        with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=slow_health):
            reader = threading.Thread(target=lambda: broker.quarantined)
            reader.start()
            self.assertTrue(entered.wait(5))
            try:
                retained_error = None
                try:
                    ticket.wait(1)
                except HostBusy as exc:
                    retained_error = exc
                self.assertIsNotNone(retained_error)
                self.assertIsNotNone(retained_error.__traceback__)
                self.assertFalse(authority.unsafe)
            finally:
                resume.set()
                reader.join(5)
        result = ticket.wait(1)
        self.assertEqual(result.status, 'ok')
        result_ref = weakref.ref(result.logits)
        ticket.confirm(result)
        del result
        ticket.release()
        first.dispose()
        other = second.delivery()
        result = other.wait(1)
        other.confirm(result)
        del result
        other.release()
        second.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        gc.collect()
        self.assertIsNotNone(retained_error)
        self.assertIsNone(result_ref())
        self.assertTrue(broker.close())

    def test_concurrent_health_poll_does_not_block_on_another_poll(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from unittest.mock import PropertyMock
        broker, _, _, (_, _, _, _, authority), make_request = make_grouped_broker()
        entered, resume, done = threading.Event(), threading.Event(), threading.Event()
        values = []

        def slow_health():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('health timed out')
            return False

        def read():
            values.append(broker.quarantined)
            done.set()

        with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=slow_health) as health:
            first = threading.Thread(target=lambda: broker.quarantined)
            first.start()
            self.assertTrue(entered.wait(5))
            second = threading.Thread(target=read)
            second.start()
            try:
                self.assertTrue(done.wait(1))
                self.assertEqual(values, [False])
                self.assertTrue(broker.unavailable)
                self.assertEqual(health.call_count, 1)
                self.assertFalse(authority.unsafe)
                with self.assertRaises(HostBusy):
                    broker.submit('team-a', make_request('a', [1, 2, 3]))
                self.assertFalse(authority.unsafe)
            finally:
                resume.set()
                first.join(5)
                second.join(5)
        self.assertFalse(broker.quarantined)
        self.assertFalse(broker.unavailable)

    def test_pending_healthy_health_probe_does_not_poison_staged_snapshot(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        snap_done, resume_snap, reading, resume_health, done = (threading.Event() for _ in range(5))
        handles, errors = [], []
        original = GroupedWorker.snapshot_request

        def snapshot(grouped, request):
            value = original(grouped, request)
            snap_done.set()
            if not resume_snap.wait(5):
                raise RuntimeError('snapshot timed out')
            return value

        def slow_health():
            reading.set()
            if not resume_health.wait(5):
                raise RuntimeError('health timed out')
            return False

        def submit():
            try:
                handles.append(broker.submit('team-a', make_request('a', [1, 2, 3])))
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(GroupedWorker, 'snapshot_request', snapshot):
            thread = threading.Thread(target=submit)
            thread.start()
            self.assertTrue(snap_done.wait(5))
            with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                              side_effect=slow_health):
                reader = threading.Thread(target=lambda: broker.quarantined)
                reader.start()
                self.assertTrue(reading.wait(5))
                try:
                    resume_snap.set()
                    self.assertFalse(done.wait(.2))
                    self.assertFalse(authority.unsafe)
                    self.assertGreater(host.snapshot().used_bytes, 0)
                finally:
                    resume_health.set()
                    reader.join(5)
                    thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(handles), 1)
        self.assertFalse(authority.unsafe)

    def test_pending_healthy_health_probe_does_not_poison_queued_dispatch(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, model, _, (_, _, _, _, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        entered, resume, done = threading.Event(), threading.Event(), threading.Event()
        outcomes, errors = [], []

        def slow_health():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('health timed out')
            return False

        def dispatch():
            try:
                outcomes.append(broker.run_once(2))
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=slow_health):
            reader = threading.Thread(target=lambda: broker.quarantined)
            reader.start()
            self.assertTrue(entered.wait(5))
            thread = threading.Thread(target=dispatch)
            thread.start()
            try:
                self.assertFalse(done.wait(.2))
                self.assertFalse(authority.unsafe)
            finally:
                resume.set()
                reader.join(5)
                thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(outcomes, [True])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertFalse(authority.unsafe)

    def test_pending_healthy_health_probe_does_not_poison_staged_cohort(self):
        from looking_glass.grouped_broker import GroupedBroker
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, model, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        staged, resume_stage, reading, resume_health, done = (threading.Event() for _ in range(5))
        outcomes, errors = [], []
        original = GroupedBroker._execute

        def execute(owner, group):
            staged.set()
            if not resume_stage.wait(5):
                raise RuntimeError('staged cohort timed out')
            return original(owner, group)

        def slow_health():
            reading.set()
            if not resume_health.wait(5):
                raise RuntimeError('health timed out')
            return False

        def dispatch():
            try:
                outcomes.append(broker.run_once(2))
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(GroupedBroker, '_execute', execute):
            thread = threading.Thread(target=dispatch)
            thread.start()
            self.assertTrue(staged.wait(5))
            with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                              side_effect=slow_health):
                reader = threading.Thread(target=lambda: broker.quarantined)
                reader.start()
                self.assertTrue(reading.wait(5))
                try:
                    resume_stage.set()
                    self.assertFalse(done.wait(.2))
                    self.assertFalse(authority.unsafe)
                    self.assertGreater(host.snapshot().used_bytes, 0)
                finally:
                    resume_health.set()
                    reader.join(5)
                    thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(outcomes, [True])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertFalse(authority.unsafe)

    def test_pending_health_wait_preserves_preclaim_dispatch_timeout(self):
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, _, (_, _, _, _, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        entered, resume = threading.Event(), threading.Event()

        def slow_health():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('health timed out')
            return False

        with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=slow_health):
            reader = threading.Thread(target=lambda: broker.quarantined)
            reader.start()
            self.assertTrue(entered.wait(5))
            try:
                self.assertFalse(broker.run_once(.05))
                self.assertFalse(authority.unsafe)
            finally:
                resume.set()
                reader.join(5)
        self.assertTrue(broker.run_once(1))

    def test_resident_state_inspection_failure_remains_privately_rooted(self):
        from looking_glass.resident_worker import WorkerUnavailable
        _, worker, model, budget, resident_lease, workspace, authority, host, _, _, _ = make_grouped_owner()
        owners = []

        class Payload:
            pass

        def fail_state():
            payload = Payload()
            owners.append(weakref.ref(payload))
            raise RuntimeError('private model-state failure')

        request = ForwardRequest('control', {'input_ids': torch.tensor([[1, 2, 3]])}, seed=17)
        with patch.object(worker._worker, '_model_state', side_effect=fail_state):
            with self.assertRaises(WorkerUnavailable) as caught:
                worker.execute_control(request)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(worker._custody.state_error, RuntimeError)
        gc.collect()
        self.assertIsNotNone(owners[0]())
        self.assertEqual(model.calls, [])
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_competing_control_preflight_cannot_fault_during_forward(self):
        import inspect
        from test_grouped_broker import ObservedModel
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from looking_glass.resident_worker import WorkerUnavailable
        _, worker, model, budget, resident_lease, workspace, authority, host, _, _, _ = make_grouped_owner(model=ObservedModel())
        model.entered, model.proceed = threading.Event(), threading.Event()
        outputs, errors, reached = [], [], threading.Event()
        lines, first = inspect.getsourcelines(GroupedWorkspaceWorker._execute_inner)
        target = first + next(i for i, line in enumerate(lines) if line.strip().startswith('signature = '))

        def first_control():
            try:
                outputs.append(worker.execute_control(ForwardRequest('first', {'input_ids': torch.tensor([[1, 2, 3]])}, seed=17)))
            except BaseException as exc:
                errors.append(exc)

        def second_control():
            def trace(frame, event, arg):
                if frame.f_code is GroupedWorkspaceWorker._execute_inner.__code__ and event == 'line' and frame.f_lineno == target:
                    sys.settrace(None)
                    reached.set()
                    raise MemoryError('private preflight failure')
                return trace
            sys.settrace(trace)
            try:
                worker.execute_control(ForwardRequest('second', {'input_ids': torch.tensor([[4, 5, 6]])}, seed=17))
            except BaseException as exc:
                errors.append(exc)
            finally:
                sys.settrace(None)

        first_thread = threading.Thread(target=first_control)
        second_thread = threading.Thread(target=second_control)
        first_thread.start()
        self.assertTrue(model.entered.wait(5))
        second_thread.start()
        try:
            self.assertFalse(reached.wait(.2))
            self.assertFalse(authority.unsafe)
        finally:
            model.proceed.set()
            first_thread.join(5)
            second_thread.join(5)
        self.assertFalse(first_thread.is_alive() or second_thread.is_alive())
        self.assertTrue(reached.is_set())
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], WorkerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(worker._preflight_failure, MemoryError)
        self.assertTrue(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_two_row_control_is_explicit_and_uses_the_same_owned_workspace(self):
        _, worker, model, budget, resident_lease, workspace, authority, _, request, claim, _ = make_grouped_owner()
        combined = ForwardRequest('control', {'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]])}, seed=17)
        self.assertTrue(callable(getattr(worker, 'execute_control', None)), 'two-row control API is missing')
        output = worker.execute_control(combined)
        self.assertTrue(torch.equal(output.logits[..., 0], torch.tensor([[1., 2., 3.], [4., 5., 6.]])))
        worker.confirm_control_result(output)
        del output
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertFalse(authority.unsafe)
        claim.release()
        request.close()

    def test_cancel_before_serial_forward_preserves_a_healthy_owner(self):
        _, worker, model, budget, resident_lease, workspace, authority, _, request, claim, _ = make_grouped_owner()
        cancelled = threading.Event()
        cancelled.set()
        serial = ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17)
        with self.assertRaises(RequestCancelled):
            worker.execute(serial, cancelled)
        self.assertEqual(model.calls, [])
        self.assertFalse(authority.unsafe)
        output = worker.execute(serial)
        worker.confirm_control_result(output)
        del output
        self.assertEqual(model.calls, [(1, 3)])
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        claim.release()
        request.close()

    def test_already_cancelled_request_does_not_make_a_private_snapshot(self):
        _, worker, _, _, _, _, authority, _, request, claim, _ = make_grouped_owner()
        cancelled = threading.Event()
        cancelled.set()
        serial = ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17)
        with patch.object(ResidentWorker, 'snapshot_request', side_effect=AssertionError('unexpected snapshot')):
            with self.assertRaises(RequestCancelled):
                worker.execute(serial, cancelled)
        self.assertFalse(authority.unsafe)
        claim.release()
        request.close()

    def test_retained_drained_cancel_exception_does_not_keep_private_snapshots(self):
        cancelled = threading.Event()

        class CancelModel(RowModel):
            def forward(self, input_ids):
                output = super().forward(input_ids)
                cancelled.set()
                return output

        _, worker, _, _, _, _, authority, _, request, claim, _ = make_grouped_owner(model=CancelModel())
        serial = ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17)
        original = ResidentWorker.snapshot_request
        snapshots = []

        def capture(resident, value):
            snapshot = original(resident, value)
            snapshots.append(weakref.ref(snapshot.inputs['input_ids']))
            return snapshot

        caught = []
        with patch.object(ResidentWorker, 'snapshot_request', capture):
            try:
                worker.execute(serial, cancelled)
            except RequestCancelled as exc:
                caught.append(exc)
        self.assertEqual(len(caught), 1)
        gc.collect()
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all(reference() is None for reference in snapshots))
        self.assertFalse(authority.unsafe)
        claim.release()
        request.close()

    def test_cancel_after_drained_serial_forward_preserves_a_healthy_owner(self):
        class BlockingModel(RowModel):
            def __init__(self):
                super().__init__()
                self.entered, self.proceed = threading.Event(), threading.Event()

            def forward(self, input_ids):
                self.entered.set()
                if not self.proceed.wait(5):
                    raise RuntimeError('forward timed out')
                return super().forward(input_ids)

        model = BlockingModel()
        _, worker, _, budget, resident_lease, workspace, authority, _, request, claim, _ = make_grouped_owner(model=model)
        cancelled = threading.Event()
        serial = ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17)
        errors = []

        def execute():
            try:
                worker.execute(serial, cancelled)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=execute)
        thread.start()
        try:
            self.assertTrue(model.entered.wait(5))
            cancelled.set()
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelled)
        self.assertFalse(authority.unsafe)
        errors.clear()
        output = worker.execute(serial)
        worker.confirm_control_result(output)
        del output
        self.assertEqual(model.calls, [(1, 3), (1, 3)])
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        claim.release()
        request.close()

    def test_owner_loss_before_cancel_retirement_keeps_copied_output(self):
        cancelled = threading.Event()

        class CancelModel(RowModel):
            def forward(self, input_ids):
                output = super().forward(input_ids)
                cancelled.set()
                return output

        _, worker, _, budget, resident_lease, workspace, authority, host, _, _, _ = make_grouped_owner(model=CancelModel())
        original = ResidentWorker._execute_with_custody
        retained = []

        def revoke_after_drain(resident, request, cancel_event, custody):
            try:
                return original(resident, request, cancel_event, custody)
            except RequestCancelled:
                retained.append(weakref.ref(custody.logits))
                authority.mark_unsafe()
                raise

        serial = ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17)
        with patch.object(ResidentWorker, '_execute_with_custody', revoke_after_drain):
            with self.assertRaises(DeviceQuarantined):
                worker.execute(serial, cancelled)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(retained[0]())
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_serial_and_two_row_calls_share_fixed_workspace_through_split(self):
        grouped, worker, model, budget, resident_lease, workspace, authority, host, request, claim, jobs = make_grouped_owner()
        serial = worker.execute(ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17))
        self.assertTrue(torch.equal(serial.logits[..., 0], torch.tensor([[7., 8., 9.]])))
        worker.confirm_control_result(serial)
        del serial
        group = grouped.prepare('batch', jobs, request, claim, 'group')
        launch = group.launch()
        original_finish = Microbatch.finish

        def finish(plan, logits, *, launch_id):
            self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
            with self.assertRaises(HostBusy):
                claim.release()
            return original_finish(plan, logits, launch_id=launch_id)

        with patch.object(Microbatch, 'finish', finish):
            outcome = grouped.execute(group, launch)
        self.assertEqual(model.calls, [(1, 3), (2, 3)])
        self.assertEqual([result.status for result in outcome.results], ['ok', 'ok'])
        self.assertTrue(torch.equal(outcome.results[0].logits[..., 0], torch.tensor([[1., 2., 3.]])))
        self.assertTrue(torch.equal(outcome.results[1].logits[..., 0], torch.tensor([[4., 5., 6.]])))
        self.assertFalse(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertGreater(host.snapshot().used_bytes, 0)
        del outcome, launch, jobs
        group.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        request.close()

    def test_unsafe_owner_cannot_retire_a_completed_group_claim(self):
        grouped, _, _, budget, resident_lease, workspace, authority, host, request, claim, jobs = make_grouped_owner()
        group = grouped.prepare('batch', jobs, request, claim, 'group')
        launch = group.launch()
        outcome = grouped.execute(group, launch)
        self.assertEqual([result.status for result in outcome.results], ['ok', 'ok'])
        held = host.snapshot().used_bytes
        self.assertGreater(held, 0)
        del outcome, launch, jobs
        authority.mark_unsafe()
        with self.assertRaises(DeviceQuarantined):
            group.dispose()
        self.assertEqual(host.snapshot().used_bytes, held)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupted_launch_claim_quarantines_before_another_forward(self):
        from looking_glass.resident_worker import WorkerUnavailable
        grouped, worker, model, budget, resident_lease, workspace, authority, host, request, claim, jobs = make_grouped_owner()
        group = grouped.prepare('batch', jobs, request, claim, 'group')
        launch = group.launch()
        original = GroupPlan._start_under_gate

        def claim_then_interrupt(plan, launch, workload, snapshot):
            original(plan, launch, workload, snapshot)
            raise RuntimeError('interrupted launch handoff')

        with patch.object(GroupPlan, '_start_under_gate', claim_then_interrupt):
            with self.assertRaises(WorkerUnavailable):
                grouped.execute(group, launch)
        self.assertEqual(model.calls, [])
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(DeviceQuarantined):
            worker.execute(ForwardRequest('serial', {'input_ids': torch.tensor([[7, 8, 9]])}, seed=17))
        with self.assertRaises(HostBusy):
            group.dispose()

    def test_owner_revoked_in_forward_blocks_later_copy_and_retains_output(self):
        live = [True]
        after_revocation = []

        def check_owner():
            if not live[0]:
                raise RuntimeError('physical owner was revoked')

        class RevokingModel(RowModel):
            def forward(self, input_ids):
                output = super().forward(input_ids)
                self.output = weakref.ref(output['logits'])
                live[0] = False
                return output

        model = RevokingModel()
        grouped, _, _, budget, resident_lease, workspace, authority, host, request, claim, jobs = make_grouped_owner(
            model=model, device_guard=check_owner)
        group = grouped.prepare('batch', jobs, request, claim, 'group')
        launch = group.launch()
        original_to = torch.Tensor.to

        def observed_to(tensor, *args, **kwargs):
            if not live[0]:
                after_revocation.append((args, kwargs))
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, 'to', observed_to):
            with self.assertRaises(Exception):
                grouped.execute(group, launch)
        self.assertEqual(after_revocation, [])
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(model.output())
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(HostBusy):
            group.dispose()

    def test_unsafe_during_final_validation_suppresses_cohort_publication(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        original = CohortState._validate_outcome

        def validate_then_lose_owner(state, outcome):
            original(state, outcome)
            authority.mark_unsafe()

        with patch.object(CohortState, '_validate_outcome', validate_then_lose_owner):
            with self.assertRaises(GroupedBrokerUnavailable):
                broker.run_once(1)
        self.assertEqual(model.calls, [(2, 3)])
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()

    def test_revoked_owner_after_validation_prevents_success_publication(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        live = [True]

        def check_owner():
            if not live[0]:
                raise RuntimeError('physical owner revoked')

        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker(
            device_guard=check_owner)
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        original = CohortState._validate_outcome

        def validate_then_revoke(state, outcome):
            original(state, outcome)
            live[0] = False

        with patch.object(CohortState, '_validate_outcome', validate_then_revoke):
            with self.assertRaises(GroupedBrokerUnavailable):
                broker.run_once(1)
        self.assertEqual(model.calls, [(2, 3)])
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_unsafe_owner_blocks_new_success_delivery_and_retains_claim(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        prior_ticket = first.delivery()
        broker.run_once(1)
        before = host.snapshot().used_bytes
        authority.mark_unsafe()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.delivery()
        with self.assertRaises(GroupedBrokerUnavailable):
            prior_ticket.wait(1)
        self.assertEqual(host.snapshot().used_bytes, before)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_queued_cancellation_retirement_waits_for_pending_health_decision(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        handle = broker.submit('team-a', make_request('a', [1, 2, 3]))
        before = host.snapshot().used_bytes
        at_decision, allow_decision, reading, finish_health, done = (threading.Event() for _ in range(5))
        errors = []
        original = GroupedWorkspaceWorker._allow_retirement

        def delayed(worker):
            at_decision.set()
            if not allow_decision.wait(5):
                raise RuntimeError('retirement timed out')
            return original(worker)

        def slow_health():
            reading.set()
            if not finish_health.wait(5):
                raise RuntimeError('health timed out')
            return True

        def cancel():
            try:
                handle.cancel()
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(GroupedWorkspaceWorker, '_allow_retirement', delayed):
            thread = threading.Thread(target=cancel)
            thread.start()
            self.assertTrue(at_decision.wait(5))
            with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                              side_effect=slow_health):
                reader = threading.Thread(target=lambda: broker.quarantined)
                reader.start()
                self.assertTrue(reading.wait(5))
                try:
                    allow_decision.set()
                    self.assertFalse(done.wait(.2))
                    self.assertEqual(host.snapshot().used_bytes, before)
                finally:
                    finish_health.set()
                    reader.join(5)
                    thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertTrue(authority.unsafe)
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_group_retirement_waits_for_pending_health_decision(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_owner_worker import GroupedWorkspaceWorker
        from unittest.mock import PropertyMock
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = ticket.wait(1)
            ticket.confirm(result)
            del result
            ticket.release()
        first.dispose()
        before = host.snapshot().used_bytes
        at_decision, allow_decision, reading, finish_health, done = (threading.Event() for _ in range(5))
        errors = []
        original = GroupedWorkspaceWorker._allow_retirement

        def delayed(worker):
            at_decision.set()
            if not allow_decision.wait(5):
                raise RuntimeError('retirement timed out')
            return original(worker)

        def slow_health():
            reading.set()
            if not finish_health.wait(5):
                raise RuntimeError('health timed out')
            return True

        def dispose():
            try:
                second.dispose()
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        with patch.object(GroupedWorkspaceWorker, '_allow_retirement', delayed):
            thread = threading.Thread(target=dispose)
            thread.start()
            self.assertTrue(at_decision.wait(5))
            with patch.object(GroupedWorkspaceWorker, 'quarantined', new_callable=PropertyMock,
                              side_effect=slow_health):
                reader = threading.Thread(target=lambda: broker.quarantined)
                reader.start()
                self.assertTrue(reading.wait(5))
                try:
                    allow_decision.set()
                    self.assertFalse(done.wait(.2))
                    self.assertEqual(host.snapshot().used_bytes, before)
                finally:
                    finish_health.set()
                    reader.join(5)
                    thread.join(5)
        self.assertFalse(thread.is_alive() or reader.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertTrue(authority.unsafe)
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_validation_claim_remains_charged_when_owner_becomes_unsafe_after_publication(self):
        from looking_glass.grouped_broker_state import CohortState
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        original = CohortState.publish_success
        retained = []

        def publish_then_lose_owner(state, outcome):
            publication = original(state, outcome)
            retained.append(next(iter(broker._groups.values())).validation_claim)
            authority.mark_unsafe()
            return publication

        with patch.object(CohortState, 'publish_success', publish_then_lose_owner):
            broker.run_once(1)
        self.assertTrue(authority.unsafe)
        self.assertEqual(retained[0].info().nbytes, 256)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_submission_fault_latches_authority_before_waiting_for_publication(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        validating, resume = threading.Event(), threading.Event()
        errors = []
        original = CohortState._validate_outcome

        def blocked_validation(state, outcome):
            original(state, outcome)
            validating.set()
            if not resume.wait(5):
                raise RuntimeError('validation timed out')

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                errors.append(exc)

        def failed_submission():
            try:
                broker.submit('team-c', make_request('c', [7, 8, 9]))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(CohortState, '_validate_outcome', blocked_validation):
            dispatch_thread = threading.Thread(target=dispatch)
            dispatch_thread.start()
            self.assertTrue(validating.wait(5))
            with patch.object(GroupedWorker, 'snapshot_request', side_effect=MemoryError('snapshot failure')):
                submit_thread = threading.Thread(target=failed_submission)
                submit_thread.start()
                try:
                    for _ in range(200):
                        if broker.quarantined:
                            break
                        threading.Event().wait(.005)
                    self.assertTrue(broker.quarantined)
                    self.assertTrue(authority.unsafe, 'fault must reach authority before waiting on cohort state')
                    self.assertGreater(host.snapshot().used_bytes, 0)
                    self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
                finally:
                    resume.set()
                    dispatch_thread.join(5)
                    submit_thread.join(5)
        self.assertFalse(dispatch_thread.is_alive())
        self.assertFalse(submit_thread.is_alive())
        self.assertTrue(errors)
        self.assertTrue(all(isinstance(exc, GroupedBrokerUnavailable) for exc in errors))

    def test_concurrent_delivery_failures_each_keep_private_custody(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        tickets = (first.delivery(), second.delivery())
        barrier = threading.Barrier(2)
        owners, public_errors, release_errors = [], [], []

        class Payload:
            pass

        def fail_delivery(*args):
            owner = Payload()
            owners.append(weakref.ref(owner))
            barrier.wait(5)
            raise MemoryError('private delivery failure')

        def consume(ticket):
            try:
                ticket.wait(5)
            except BaseException as exc:
                public_errors.append(exc)
            try:
                ticket.release()
            except BaseException as exc:
                release_errors.append(exc)

        with patch('looking_glass.grouped_broker.GroupedResult', side_effect=fail_delivery):
            threads = [threading.Thread(target=consume, args=(ticket,)) for ticket in tickets]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(public_errors), 2)
        self.assertTrue(all(isinstance(exc, GroupedBrokerUnavailable) and exc.__context__ is None
                            for exc in public_errors))
        self.assertEqual(len(release_errors), 2)
        gc.collect()
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(reference() is not None for reference in owners))
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_delivery_creation_failure_retains_private_custody_and_blocks_retry(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        owners = []

        class Payload:
            pass

        def fail_ticket(*args):
            owner = Payload()
            owners.append(weakref.ref(owner))
            raise MemoryError('private ticket construction failure')

        with patch('looking_glass.grouped_broker.GroupedDelivery', side_effect=fail_ticket):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                first.delivery()
        self.assertIsNone(caught.exception.__context__)
        gc.collect()
        self.assertIsNotNone(owners[0]())
        self.assertIsInstance(first._job._delivery_failure, MemoryError)
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(GroupedBrokerUnavailable):
            first.delivery()

    def test_delivery_retains_both_distinct_notification_failures(self):
        import inspect
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        before = host.snapshot().used_bytes
        lines, first_line = inspect.getsourcelines(GroupedBroker._wait)
        target = first_line + next(i for i, line in enumerate(lines) if line.strip() == 'ticket._failure = exc')
        references = []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._wait.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                tensor = torch.ones(31)
                references.append(weakref.ref(tensor))
                raise KeyboardInterrupt(tensor)
            return trace

        def fail_notification(*args, **kwargs):
            tensor = torch.ones(47)
            references.append(weakref.ref(tensor))
            raise RuntimeError(tensor)

        with patch('looking_glass.grouped_broker.GroupedResult', side_effect=MemoryError('first failure')), \
             patch.object(broker, '_fail_closed', side_effect=fail_notification):
            sys.settrace(trace)
            try:
                with self.assertRaises(GroupedBrokerUnavailable):
                    ticket.wait(1)
            finally:
                sys.settrace(None)
        gc.collect()
        self.assertEqual(len(references), 2)
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertTrue(authority.unsafe)
        self.assertEqual(host.snapshot().used_bytes, before)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_failed_ticket_cannot_be_waited_again_or_drop_its_first_error(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        with patch('looking_glass.grouped_broker.GroupedResult', side_effect=MemoryError('first failure')):
            with self.assertRaises(GroupedBrokerUnavailable):
                ticket.wait(1)
        original_failure = ticket._failure
        with self.assertRaises(GroupedBrokerUnavailable):
            ticket.wait(1)
        self.assertIs(ticket._failure, original_failure)
        with self.assertRaises(GroupedBrokerUnavailable):
            ticket.release()
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_interrupted_delivery_error_root_is_recovered_before_public_return(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        lines, first_line = __import__('inspect').getsourcelines(GroupedBroker._wait)
        target = first_line + next(i for i, line in enumerate(lines) if line.strip() == 'ticket._failure = exc')
        interrupted, errors = threading.Event(), []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._wait.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                interrupted.set()
                raise KeyboardInterrupt('delivery custody interrupted')
            return trace

        with patch('looking_glass.grouped_broker.GroupedResult', side_effect=MemoryError('private delivery')):
            sys.settrace(trace)
            try:
                try:
                    ticket.wait(1)
                except BaseException as exc:
                    errors.append(exc)
            finally:
                sys.settrace(None)
        self.assertTrue(interrupted.is_set())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(ticket._failure, MemoryError)
        self.assertIsInstance(ticket._notification_failure, KeyboardInterrupt)
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        with self.assertRaises(GroupedBrokerUnavailable):
            ticket.release()

    def test_concurrent_submission_failures_each_keep_private_custody(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        barrier = threading.Barrier(2)
        owners, public_errors = [], []

        class Payload:
            pass

        def fail_snapshot(*args):
            owner = Payload()
            owners.append(weakref.ref(owner))
            barrier.wait(5)
            raise MemoryError('private snapshot failure')

        def submit(team, request_id, row):
            try:
                broker.submit(team, make_request(request_id, row))
            except BaseException as exc:
                public_errors.append(exc)

        with patch.object(GroupedWorker, 'snapshot_request', side_effect=fail_snapshot):
            threads = [threading.Thread(target=submit, args=('team-a', 'a', [1, 2, 3])),
                       threading.Thread(target=submit, args=('team-b', 'b', [4, 5, 6]))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(public_errors), 2)
        self.assertTrue(all(isinstance(exc, GroupedBrokerUnavailable) and exc.__context__ is None
                            for exc in public_errors))
        gc.collect()
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(reference() is not None for reference in owners))
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_failed_cancellation_retry_preserves_first_error(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        broker, _, host, (_, _, _, _, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        with patch.object(CohortState, 'cancel', side_effect=MemoryError('first failure')):
            with self.assertRaises(GroupedBrokerUnavailable):
                first.cancel()
        original = first._job._cancel_failure
        with self.assertRaises(GroupedBrokerUnavailable):
            first.cancel()
        self.assertIs(first._job._cancel_failure, original)
        self.assertIsInstance(original, MemoryError)
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_concurrent_cancellation_failures_each_keep_private_custody(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        barrier = threading.Barrier(2)
        owners, public_errors, dispatch_errors = [], [], []

        class Payload:
            pass

        def fail_cancel(*args):
            owner = Payload()
            owners.append(weakref.ref(owner))
            barrier.wait(5)
            raise MemoryError('private cancel failure')

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                dispatch_errors.append(exc)

        def cancel(handle):
            try:
                handle.cancel()
            except BaseException as exc:
                public_errors.append(exc)

        dispatch_thread = threading.Thread(target=dispatch)
        dispatch_thread.start()
        self.assertTrue(model.entered.wait(5))
        try:
            with patch.object(CohortState, 'cancel', fail_cancel):
                threads = [threading.Thread(target=cancel, args=(handle,)) for handle in (first, second)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(5)
        finally:
            model.proceed.set()
            dispatch_thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(public_errors), 2)
        self.assertTrue(all(isinstance(exc, GroupedBrokerUnavailable) and exc.__context__ is None
                            for exc in public_errors))
        gc.collect()
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(reference() is not None for reference in owners))
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_concurrent_disposal_failures_each_keep_private_custody(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        from unittest.mock import PropertyMock
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        barrier = threading.Barrier(2)
        owners, public_errors = [], []

        class Payload:
            pass

        def fail_phase():
            owner = Payload()
            owners.append(weakref.ref(owner))
            barrier.wait(5)
            raise MemoryError('private phase failure')

        def dispose(handle):
            try:
                handle.dispose()
            except BaseException as exc:
                public_errors.append(exc)

        with patch.object(CohortState, 'phase', new_callable=PropertyMock, side_effect=fail_phase):
            threads = [threading.Thread(target=dispose, args=(handle,)) for handle in (first, second)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(public_errors), 2)
        self.assertTrue(all(isinstance(exc, GroupedBrokerUnavailable) and exc.__context__ is None
                            for exc in public_errors))
        gc.collect()
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(reference() is not None for reference in owners))
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupt_entering_fault_latch_cannot_publish_an_active_cohort(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        validating, resume, interrupted = threading.Event(), threading.Event(), threading.Event()
        errors = []
        original = CohortState._validate_outcome

        def blocked_validation(state, outcome):
            original(state, outcome)
            validating.set()
            if not resume.wait(5):
                raise RuntimeError('validation timed out')

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                errors.append(exc)

        def failed_submission():
            def trace(frame, event, arg):
                if frame.f_code is GroupedBroker._latch_owner_unsafe.__code__ and event == 'call':
                    sys.settrace(None)
                    interrupted.set()
                    raise KeyboardInterrupt('interrupted unsafe notification')
                return trace
            sys.settrace(trace)
            try:
                broker.submit('team-c', make_request('c', [7, 8, 9]))
            except BaseException as exc:
                errors.append(exc)
            finally:
                sys.settrace(None)

        with patch.object(CohortState, '_validate_outcome', blocked_validation):
            dispatch_thread = threading.Thread(target=dispatch)
            dispatch_thread.start()
            self.assertTrue(validating.wait(5))
            with patch.object(GroupedWorker, 'snapshot_request', side_effect=MemoryError('snapshot failure')):
                submit_thread = threading.Thread(target=failed_submission)
                submit_thread.start()
                try:
                    self.assertTrue(interrupted.wait(5))
                    for _ in range(200):
                        if authority.unsafe:
                            break
                        threading.Event().wait(.005)
                    self.assertTrue(authority.unsafe)
                    self.assertTrue(broker.quarantined)
                    self.assertGreater(host.snapshot().used_bytes, 0)
                    self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
                finally:
                    resume.set()
                    dispatch_thread.join(5)
                    submit_thread.join(5)
        self.assertFalse(dispatch_thread.is_alive())
        self.assertFalse(submit_thread.is_alive())
        self.assertTrue(errors)
        self.assertTrue(all(isinstance(exc, (GroupedBrokerUnavailable, KeyboardInterrupt)) for exc in errors))
        self.assertTrue(all(not isinstance(exc, GroupedBrokerUnavailable) or exc.__context__ is None for exc in errors))

    def test_interrupt_entering_dispatch_fault_latch_blocks_future_work(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        original = GroupedBroker._claim
        errors = []

        def claim_then_fail(owner, timeout):
            self.assertIsNotNone(original(owner, timeout))
            raise MemoryError('staged claim failed')

        def dispatch():
            def trace(frame, event, arg):
                if frame.f_code is GroupedBroker._latch_owner_unsafe.__code__ and event == 'call':
                    sys.settrace(None)
                    raise KeyboardInterrupt('interrupted unsafe notification')
                return trace
            sys.settrace(trace)
            try:
                broker.run_once(1)
            except BaseException as exc:
                errors.append(exc)
            finally:
                sys.settrace(None)

        with patch.object(GroupedBroker, '_claim', claim_then_fail):
            thread = threading.Thread(target=dispatch)
            thread.start()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(broker._failure, MemoryError)
        self.assertIsInstance(broker._notification_failure, KeyboardInterrupt)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertEqual(model.calls, [])
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(GroupedBrokerUnavailable):
            broker.submit('team-c', make_request('c', [7, 8, 9]))

    def test_interrupt_entering_dispatch_handler_has_no_public_failure_context(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        original = GroupedBroker._claim
        errors = []

        def claim_then_fail(owner, timeout):
            original(owner, timeout)
            raise MemoryError('staged claim failed')

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._fail_closed.__code__ and event == 'call':
                sys.settrace(None)
                raise KeyboardInterrupt('fault handler interrupted')
            return trace

        with patch.object(GroupedBroker, '_claim', claim_then_fail):
            sys.settrace(trace)
            try:
                try:
                    broker.run_once(1)
                except BaseException as exc:
                    errors.append(exc)
            finally:
                sys.settrace(None)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(broker._failure, MemoryError)
        self.assertIsInstance(broker._notification_failure, KeyboardInterrupt)
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupt_entering_submission_handler_has_no_public_failure_context(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        errors = []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._fail_closed.__code__ and event == 'call':
                sys.settrace(None)
                raise KeyboardInterrupt('fault handler interrupted')
            return trace

        with patch.object(GroupedWorker, 'snapshot_request', side_effect=MemoryError('snapshot failure')):
            sys.settrace(trace)
            try:
                try:
                    broker.submit('team-a', make_request('a', [1, 2, 3]))
                except BaseException as exc:
                    errors.append(exc)
            finally:
                sys.settrace(None)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(broker._failure, MemoryError)
        self.assertIsInstance(broker._notification_failure, KeyboardInterrupt)
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})

    def test_interrupt_entering_disposal_failure_handler_keeps_claims(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        from looking_glass.grouped_broker_state import CohortState
        from unittest.mock import PropertyMock
        broker, _, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        broker.run_once(1)
        before = host.snapshot().used_bytes
        errors = []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._fail_closed.__code__ and event == 'call':
                sys.settrace(None)
                raise KeyboardInterrupt('interrupted fault handler')
            return trace

        with patch.object(CohortState, 'phase', new_callable=PropertyMock, side_effect=MemoryError('state read')):
            sys.settrace(trace)
            try:
                try:
                    first.dispose()
                except BaseException as exc:
                    errors.append(exc)
            finally:
                sys.settrace(None)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GroupedBrokerUnavailable)
        self.assertIsNone(errors[0].__context__)
        self.assertIsInstance(broker._failure, MemoryError)
        self.assertIsInstance(broker._notification_failure, KeyboardInterrupt)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)
        self.assertEqual(host.snapshot().used_bytes, before)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()

    def test_unsafe_owner_wakes_waiter_before_blocked_forward_returns(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        broker.submit('team-b', make_request('b', [4, 5, 6]))
        ticket = first.delivery()
        dispatch_errors, wait_errors = [], []
        done = threading.Event()

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                dispatch_errors.append(exc)

        def wait():
            try:
                ticket.wait()
            except BaseException as exc:
                wait_errors.append(exc)
            finally:
                done.set()

        dispatch_thread = threading.Thread(target=dispatch)
        wait_thread = threading.Thread(target=wait)
        dispatch_thread.start()
        self.assertTrue(model.entered.wait(5))
        wait_thread.start()
        try:
            authority.mark_unsafe()
            self.assertTrue(done.wait(1), 'the waiter must observe unsafe state before forward returns')
            self.assertIsInstance(wait_errors[0], GroupedBrokerUnavailable)
            self.assertGreater(host.snapshot().used_bytes, 0)
            self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        finally:
            model.proceed.set()
            dispatch_thread.join(5)
            wait_thread.join(5)
        self.assertFalse(dispatch_thread.is_alive())
        self.assertFalse(wait_thread.is_alive())

    def test_member_cancel_during_forward_suppresses_only_that_member(self):
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        errors = []

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        try:
            self.assertTrue(model.entered.wait(5))
            self.assertTrue(first.cancel())
            self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertEqual(errors, [])
        self.assertFalse(thread.is_alive())
        left_ticket, right_ticket = first.delivery(), second.delivery()
        left, right = left_ticket.wait(1), right_ticket.wait(1)
        left_ticket.confirm(left)
        right_ticket.confirm(right)
        self.assertEqual((left.status, right.status), ('cancelled', 'ok'))
        self.assertIsNone(left.logits)
        self.assertTrue(torch.equal(right.logits, torch.tensor([[[4.], [5.], [6.]]])))
        self.assertFalse(authority.unsafe)
        del left, right
        left_ticket.release()
        right_ticket.release()
        first.dispose()
        second.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
        self.assertTrue(broker.close())

    def test_batch_oom_takes_precedence_over_member_cancellation(self):
        broker, model, host, (_, budget, resident_lease, workspace, authority), make_request = make_grouped_broker()
        model.fail = True
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', make_request('a', [1, 2, 3]))
        second = broker.submit('team-b', make_request('b', [4, 5, 6]))
        errors = []

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        try:
            self.assertTrue(model.entered.wait(5))
            self.assertTrue(first.cancel())
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertEqual(errors, [])
        self.assertFalse(thread.is_alive())
        left_ticket, right_ticket = first.delivery(), second.delivery()
        left, right = left_ticket.wait(1), right_ticket.wait(1)
        left_ticket.confirm(left)
        right_ticket.confirm(right)
        self.assertEqual((left.status, right.status), ('failed', 'failed'))
        self.assertEqual((left.error_code, right.error_code), ('out_of_memory', 'out_of_memory'))
        self.assertTrue(authority.unsafe)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(set(budget.active_leases()), {resident_lease, workspace})
