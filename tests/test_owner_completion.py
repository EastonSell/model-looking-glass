"""CPU-only integration checks for the fixed-workspace owner boundary."""

import threading
import unittest
import weakref
import sys
import inspect
import dis
import gc
import traceback
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from looking_glass.broker import Broker, BrokerShutdownError, BrokerUnavailable, RequestState
from looking_glass.budgeted_worker import DeviceQuarantined, WorkerContractError, WorkloadProfile
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.owner_completion import OwnerCompletionAuthority, LifetimeWorkspaceWorker
from looking_glass.resident_worker import (ForwardRequest, RequestCancelled, ResidentWorker,
                                           WorkerOutOfMemory, WorkerUnavailable)
from looking_glass.grouped_observation import _local as observation_local
import looking_glass.resident_worker as resident_module


MODEL_ID = 'a' * 64


class BlockingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def forward(self, input_ids, **kwargs):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('forward timed out')
        values = input_ids.float()
        return {'logits': torch.stack((values, values + 1), dim=-1)}


def make_owner(*, model=None, cold_warm=False, device_guard=None):
    model = (BlockingModel() if model is None else model).eval()
    resident = ResidentWorker(MODEL_ID, model, device='cpu', max_batch_size=1,
                              max_sequence_length=2, max_input_bytes=64,
                              max_output_bytes=64)
    budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    load = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    workspace = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    profile = WorkloadProfile('test', MODEL_ID, 'cpu', (1, 2),
                              (('input_ids', 'torch.int64'),), 64, 1024)
    authority = OwnerCompletionAuthority(device_guard=device_guard)
    profiles = (profile, replace(profile, profile_id='warm')) if cold_warm else (profile,)
    worker = LifetimeWorkspaceWorker(resident, budget, load, workspace,
                                     profiles=profiles, authority=authority)
    broker = None if cold_warm else Broker(worker, max_pending=2, max_pending_per_team=2,
                                            completion_authority=authority)
    request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
    return model, resident, budget, load, workspace, authority, worker, broker, request


class OwnerCompletionTests(unittest.TestCase):
    def test_unsafe_during_real_forward_suppresses_success_and_retains_workspace(self):
        model, resident, budget, load, workspace, authority, worker, broker, request = make_owner()
        handle = broker.submit('team', request)
        try:
            self.assertTrue(model.entered.wait(5))
            self.assertEqual(set(budget.active_leases()), {load, workspace})
            authority.mark_unsafe()
        finally:
            model.release.set()
        try:
            outcome = handle.wait(5)
        except BrokerUnavailable:
            outcome = None
        self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
        self.assertTrue(broker.quarantined)
        self.assertTrue(worker.quarantined)
        self.assertEqual(set(budget.active_leases()), {load, workspace})
        self.assertIs(resident._model, model)
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_resident_close_preserves_model_after_owner_unsafe(self):
        model, resident, _, _, _, authority, _, broker, _ = make_owner()
        authority.mark_unsafe()
        resident.close()
        self.assertIs(resident._model, model)
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_resident_close_rechecks_process_quarantine_at_decision(self):
        model, resident, _, _, _, authority, _, broker, _ = make_owner()
        with patch.object(resident_module, '_is_process_quarantined', side_effect=[False, True]):
            resident.close()
        self.assertTrue(authority.unsafe)
        self.assertIs(resident._model, model)
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_success_decided_before_unsafe_may_become_visible_after_signal(self):
        model, _, budget, load, workspace, authority, _, broker, request = make_owner()
        original = broker._finish_locked
        decided, continue_publish = threading.Event(), threading.Event()

        def delayed(handle, state, *args, **kwargs):
            if state is RequestState.SUCCEEDED:
                decided.set()
                if not continue_publish.wait(5):
                    raise RuntimeError('publication timed out')
            return original(handle, state, *args, **kwargs)

        with patch.object(broker, '_finish_locked', side_effect=delayed):
            handle = broker.submit('team', request)
            self.assertTrue(model.entered.wait(5))
            model.release.set()
            try:
                self.assertTrue(decided.wait(5))
                authority.mark_unsafe()
            finally:
                continue_publish.set()
            outcome = handle.wait(5)
        self.assertEqual(outcome.state, RequestState.SUCCEEDED)
        self.assertTrue(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {load, workspace})
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_unsafe_wakes_an_indefinite_wait_while_forward_is_blocked(self):
        model, _, _, _, _, authority, _, broker, request = make_owner()
        handle = broker.submit('team', request)
        self.assertTrue(model.entered.wait(5))
        waiting, done, observed = threading.Event(), threading.Event(), []

        def wait():
            try:
                handle.wait()
            except BaseException as exc:
                observed.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=wait)
        original_wait = broker._condition.wait
        def observed_wait(timeout=None):
            if threading.current_thread() is thread:
                waiting.set()
            return original_wait(timeout)
        with patch.object(broker._condition, 'wait', observed_wait):
            thread.start()
            try:
                self.assertTrue(waiting.wait(5))
                # Acquiring this condition proves the client has released it
                # inside wait(), rather than merely approaching the call.
                with broker._condition:
                    authority.mark_unsafe()
                self.assertTrue(done.wait(1), 'unsafe owner must wake the waiter without waiting for forward')
                self.assertIsInstance(observed[0], BrokerUnavailable)
            finally:
                model.release.set()
                thread.join(5)

    def test_workspace_loss_during_forward_is_detected_before_success(self):
        model, _, budget, load, workspace, authority, _, broker, request = make_owner()
        handle = broker.submit('team', request)
        try:
            self.assertTrue(model.entered.wait(5))
            workspace.release()  # Simulate a broken trusted owner.
        finally:
            model.release.set()
        try:
            outcome = handle.wait(5)
        except BrokerUnavailable:
            outcome = None
        self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
        self.assertTrue(authority.unsafe)
        self.assertEqual(budget.active_leases(), (load,))
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_mutated_fixed_lease_amount_is_rejected_before_forward(self):
        for changed in ('resident', 'workspace'):
            with self.subTest(changed=changed):
                model, _, _, load, workspace, authority, _, broker, request = make_owner()
                (load if changed == 'resident' else workspace)._estimated_incremental_bytes = 1
                handle = broker.submit('team', request)
                try:
                    outcome = handle.wait(5)
                except BrokerUnavailable:
                    outcome = None
                self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
                self.assertFalse(model.entered.is_set())
                self.assertTrue(authority.unsafe)

    def test_cold_profile_transitions_to_warm_only_after_validated_result(self):
        model, _, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        signature = ((1, 2), (('input_ids', 'torch.int64'),))
        self.assertEqual(worker._profiles[signature].profile_id, 'test')
        with self.assertRaisesRegex(ValueError, 'cold warmup'):
            Broker(worker, max_pending=1, max_pending_per_team=1, completion_authority=authority)
        model.release.set()
        result = worker.execute(request)
        self.assertEqual(worker._profiles[signature].profile_id, 'test')
        worker._finish_cold(result)
        self.assertEqual(worker._profiles[signature].profile_id, 'warm')
        broker = Broker(worker, max_pending=1, max_pending_per_team=1, completion_authority=authority)
        self.assertEqual(broker.submit('team', request).wait(5).state, RequestState.SUCCEEDED)
        self.assertTrue(broker.close(timeout=5))
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_unbrokered_closed_worker_can_retire_only_once_without_rebinding(self):
        model, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        model.release.set()
        result = worker.execute(request)
        worker._finish_cold(result)
        worker.close()
        resident.close()
        authority.retire_unbrokered_worker(worker)
        self.assertIsNone(authority._worker)
        self.assertTrue(authority._worker_retired)
        with self.assertRaisesRegex(ValueError, 'already bound'):
            authority._bind_worker(worker)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_unbrokered_retirement_keeps_worker_if_unsafe_wins(self):
        model, resident, _, _, _, authority, worker, _, _ = make_owner(cold_warm=True)
        authority.mark_unsafe()
        worker.close()
        resident.close()
        with self.assertRaises(WorkerUnavailable):
            authority.retire_unbrokered_worker(worker)
        self.assertIs(authority._worker, worker)
        self.assertIs(resident._model, model)

    def test_unbrokered_retirement_rejects_live_custody_or_worker_execution(self):
        _, resident, _, _, _, authority, worker, _, request = make_owner(cold_warm=True)
        worker.close()
        resident.close()
        worker._custody.request = request
        with self.assertRaises(WorkerUnavailable):
            authority.retire_unbrokered_worker(worker)
        self.assertIs(authority._worker, worker)
        worker._custody.request = None
        self.assertTrue(worker._lock.acquire(blocking=False))
        try:
            with self.assertRaises(WorkerUnavailable):
                authority.retire_unbrokered_worker(worker)
        finally:
            worker._lock.release()
        self.assertIs(authority._worker, worker)

    def test_collector_retirement_rejects_a_broker_bound_worker(self):
        _, resident, _, _, _, authority, worker, broker, _ = make_owner()
        self.assertTrue(broker.close(timeout=5))
        resident.close()
        with self.assertRaises(WorkerUnavailable):
            authority.retire_unbrokered_worker(worker)
        self.assertIs(authority._worker, worker)

    def test_unbrokered_retirement_rejects_attached_model_or_poison(self):
        model, resident, _, _, _, authority, worker, _, _ = make_owner(cold_warm=True)
        worker.close()
        resident.close()
        resident._model = model  # Simulate an incomplete resident close.
        with self.assertRaises(WorkerUnavailable):
            authority.retire_unbrokered_worker(worker)
        self.assertIs(authority._worker, worker)
        resident._model = None
        with patch.object(ResidentWorker, 'poisoned', new_callable=PropertyMock,
                          return_value=True):
            with self.assertRaises(WorkerUnavailable):
                authority.retire_unbrokered_worker(worker)
        self.assertTrue(authority.unsafe)
        self.assertIs(authority._worker, worker)

    def test_interrupt_during_collector_retirement_latches_authority(self):
        _, resident, _, _, _, authority, worker, _, _ = make_owner(cold_warm=True)
        worker.close()
        resident.close()
        source, first = inspect.getsourcelines(OwnerCompletionAuthority.retire_unbrokered_worker)
        target = first + next(i for i, line in enumerate(source)
                              if line.strip() == 'self._worker = None')
        fired = []

        def trace(frame, event, arg):
            if (frame.f_code is OwnerCompletionAuthority.retire_unbrokered_worker.__code__
                    and event == 'line' and frame.f_lineno == target and not fired):
                fired.append(True)
                raise KeyboardInterrupt()
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                authority.retire_unbrokered_worker(worker)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(authority.unsafe)
        self.assertTrue(authority._worker_retired)
        self.assertIs(authority._worker, worker)

    def test_fixed_worker_rejects_attached_recorder_before_admission(self):
        model, _, _, _, _, authority, _, broker, request = make_owner()
        broker._serial_observation = object()
        with self.assertRaises(BrokerUnavailable):
            broker.submit('team', request)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(authority.unsafe)
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_fixed_worker_rejects_active_scope_on_actual_worker_thread(self):
        model, resident, _, _, _, authority, _, broker, request = make_owner()
        original = resident._execute_with_custody
        scope = SimpleNamespace(owner=SimpleNamespace(_fault=lambda: None))

        def execute_with_scope(*args):
            observation_local.forward = scope
            try:
                return original(*args)
            finally:
                observation_local.forward = None

        with patch.object(resident, '_execute_with_custody', side_effect=execute_with_scope):
            handle = broker.submit('team', request)
            try:
                outcome = handle.wait(5)
            except BrokerUnavailable:
                outcome = None
        self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(authority.unsafe)
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_active_scope_is_rejected_before_device_guard_is_called(self):
        calls = []
        authority = OwnerCompletionAuthority(device_guard=lambda: calls.append('guard'))
        old_scope = getattr(observation_local, 'forward', None)
        observation_local.forward = object()
        try:
            with self.assertRaises(WorkerContractError):
                authority._before_device()
        finally:
            observation_local.forward = old_scope
        self.assertEqual(calls, [])
        self.assertTrue(authority.unsafe)

    def test_recorder_attached_during_forward_suppresses_publication(self):
        model, _, budget, load, workspace, authority, _, broker, request = make_owner()
        handle = broker.submit('team', request)
        try:
            self.assertTrue(model.entered.wait(5))
            broker._serial_observation = object()
        finally:
            model.release.set()
        try:
            outcome = handle.wait(5)
        except BrokerUnavailable:
            outcome = None
        self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
        self.assertTrue(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {load, workspace})
        with self.assertRaises(BrokerShutdownError):
            broker.close()

    def test_binding_inspection_failure_marks_owner_unsafe(self):
        model, _, _, _, _, authority, worker, broker, request = make_owner()
        model.config = SimpleNamespace(use_cache=True)
        with self.assertRaises(Exception):
            worker.snapshot_request(request)
        self.assertTrue(authority.unsafe)
        self.assertTrue(broker.quarantined)

    def test_state_inspection_failure_signals_before_event_and_retains_error_owner(self):
        class ClaimOwner:
            pass

        for stage in ('before_forward', 'after_forward'):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(stage=stage, error=error_type.__name__):
                    model, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
                    model.release.set()
                    entered, release, finalized = threading.Event(), threading.Event(), threading.Event()
                    tracked, aliases, observed, calls = [], [], [], []
                    original_set = resident._poison_pending.set

                    def inspect_state():
                        calls.append(True)
                        if stage == 'after_forward' and len(calls) == 1:
                            return resident._baseline
                        tensor = torch.tensor([1.0])
                        owner = ClaimOwner()
                        tensor.claim_owner = owner
                        tracked.append(weakref.ref(owner))
                        weakref.finalize(owner, finalized.set)
                        aliases.append(tensor.detach())
                        raise error_type('inspection failure')

                    def blocked_set():
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError('event publication timed out')
                        original_set()

                    def execute():
                        try:
                            worker.execute(request)
                        except BaseException as exc:
                            observed.append(exc)

                    thread = threading.Thread(target=execute)
                    with patch.object(resident, '_model_state', new=inspect_state), \
                            patch.object(resident._poison_pending, 'set', new=blocked_set):
                        thread.start()
                        try:
                            self.assertTrue(entered.wait(5))
                            self.assertTrue(authority.unsafe)
                            with self.assertRaises(DeviceQuarantined):
                                authority._before_device()
                            gc.collect()
                            self.assertIsNotNone(tracked[0]())
                            self.assertFalse(finalized.is_set())
                        finally:
                            release.set()
                            thread.join(5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(len(observed), 1)
                    if error_type is KeyboardInterrupt:
                        self.assertIsInstance(observed[0], KeyboardInterrupt)
                        self.assertIsNone(observed[0].__context__)
                    traceback.clear_frames(observed[0].__traceback__)
                    gc.collect()
                    self.assertIsNotNone(tracked[0]())
                    self.assertFalse(finalized.is_set())
                    self.assertIsNotNone(worker._custody.state_error)
                    self.assertEqual(set(budget.active_leases()), {load, workspace})
                    self.assertEqual(len(aliases), 1)

    def test_forward_and_later_state_exceptions_keep_distinct_private_owners(self):
        class ClaimOwner:
            pass

        tracked, aliases = [], []
        def fail_with_owner(message):
            tensor = torch.tensor([1.0])
            owner = ClaimOwner()
            tensor.claim_owner = owner
            tracked.append(weakref.ref(owner))
            aliases.append(tensor.detach())
            raise RuntimeError(message)

        class FailingModel(BlockingModel):
            def forward(self, input_ids, **kwargs):
                fail_with_owner('forward')

        model, resident, budget, load, workspace, authority, worker, _, request = make_owner(
            model=FailingModel(), cold_warm=True)
        calls = []
        def inspect_state():
            calls.append(True)
            if len(calls) == 1:
                return resident._baseline
            fail_with_owner('post-forward inspection')

        with patch.object(resident, '_model_state', new=inspect_state):
            with self.assertRaises(Exception):
                worker.execute(request)
        gc.collect()
        self.assertEqual(len(tracked), 2)
        self.assertTrue(all(ref() is not None for ref in tracked))
        self.assertEqual(len(aliases), 2)
        self.assertIsNotNone(worker._custody.forward_error)
        self.assertIsNotNone(worker._custody.state_error)
        self.assertTrue(authority.unsafe)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_snapshot_failure_keeps_owner_and_blocks_overlapping_preparations(self):
        class ClaimOwner:
            pass

        _, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
        tracked, aliases, calls, errors = [], [], [], []

        def fail_clone(tensor, *args, **kwargs):
            calls.append(True)
            temporary = torch.tensor([1.0])
            owner = ClaimOwner()
            temporary.claim_owner = owner
            tracked.append(weakref.ref(owner))
            aliases.append(temporary.detach())
            entered.set()
            if not release.wait(5):
                raise RuntimeError('snapshot test timed out')
            raise MemoryError('snapshot allocation failed')

        def prepare():
            if threading.current_thread() is second:
                second_started.set()
            try:
                worker.snapshot_request(request)
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=prepare)
        second = threading.Thread(target=prepare)
        with patch.object(torch.Tensor, 'clone', new=fail_clone):
            first.start()
            try:
                self.assertTrue(entered.wait(5))
                second.start()
                self.assertTrue(second_started.wait(5))
            finally:
                release.set()
                first.join(5)
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(errors), 2)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(resident._snapshot_error)
        gc.collect()
        self.assertIsNotNone(tracked[0]())
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_interrupt_entering_snapshot_failure_handler_retains_original_error(self):
        class ClaimOwner:
            pass

        _, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        tracked, aliases, fired = [], [], []

        def fail_clone(tensor, *args, **kwargs):
            temporary = torch.tensor([1.0])
            owner = ClaimOwner()
            temporary.claim_owner = owner
            tracked.append(weakref.ref(owner))
            aliases.append(temporary.detach())
            raise RuntimeError('snapshot failure')

        source, first = inspect.getsourcelines(ResidentWorker.snapshot_request)
        target = first + next(i for i, line in enumerate(source)
                              if line.strip() == 'authority._unsafe = True')
        def trace(frame, event, arg):
            if (frame.f_code is ResidentWorker.snapshot_request.__code__ and event == 'line'
                    and frame.f_lineno == target and not fired):
                fired.append(True)
                raise KeyboardInterrupt()
            return trace

        previous = sys.gettrace()
        try:
            with patch.object(torch.Tensor, 'clone', new=fail_clone):
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as raised:
                    worker.snapshot_request(request)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(authority.unsafe)
        self.assertIsInstance(resident._snapshot_error, RuntimeError)
        self.assertIsNone(raised.exception.__context__)
        gc.collect()
        self.assertIsNotNone(tracked[0]())
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_snapshot_oom_does_not_expose_an_active_private_exception(self):
        class ClaimOwner:
            pass

        _, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        tracked, aliases, observed = [], [], []
        def original_failure():
            temporary = torch.tensor([1.0])
            owner = ClaimOwner()
            temporary.claim_owner = owner
            tracked.append(weakref.ref(owner))
            aliases.append(temporary.detach())
            raise MemoryError('private snapshot failure')

        try:
            original_failure()
        except MemoryError as active:
            def fail_clone(*args, **kwargs):
                raise active

            with patch.object(torch.Tensor, 'clone', new=fail_clone):
                try:
                    worker.snapshot_request(request)
                except WorkerOutOfMemory as public:
                    observed.append(public)
                    self.assertIsNone(public.__context__)
                    self.assertIsNone(public.__cause__)
                    self.assertIs(resident._snapshot_error, active)
                else:
                    self.fail('expected a sanitized snapshot OOM')
        self.assertEqual(len(observed), 1)
        traceback.clear_frames(observed[0].__traceback__)
        gc.collect()
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(tracked[0]())
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_clean_cancellation_has_neutral_public_message(self):
        cancel = threading.Event()
        class CancellingModel(BlockingModel):
            def forward(self, input_ids, **kwargs):
                cancel.set()
                values = input_ids.float()
                return {'logits': torch.stack((values, values + 1), dim=-1)}

        _, _, _, _, _, authority, worker, _, request = make_owner(
            model=CancellingModel(), cold_warm=True)
        with self.assertRaises(RequestCancelled) as raised:
            worker.execute(request, cancel)
        self.assertEqual(str(raised.exception), 'request cancelled')
        self.assertFalse(authority.unsafe)

    def test_state_failure_does_not_expose_an_active_private_exception(self):
        class ClaimOwner:
            pass

        _, resident, budget, load, workspace, authority, worker, _, request = make_owner(cold_warm=True)
        tracked, aliases, observed = [], [], []
        def original_failure():
            temporary = torch.tensor([1.0])
            owner = ClaimOwner()
            temporary.claim_owner = owner
            tracked.append(weakref.ref(owner))
            aliases.append(temporary.detach())
            raise RuntimeError('private inspection failure')

        try:
            original_failure()
        except RuntimeError as active:
            def inspect_state():
                raise active

            with patch.object(resident, '_model_state', new=inspect_state):
                try:
                    worker.execute(request)
                except WorkerUnavailable as public:
                    observed.append(public)
                    self.assertIsNone(public.__context__)
                    self.assertIsNone(public.__cause__)
                    self.assertIs(worker._custody.state_error, active)
                else:
                    self.fail('expected a sanitized state failure')
        traceback.clear_frames(observed[0].__traceback__)
        gc.collect()
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(tracked[0]())
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_guard_mutation_during_forward_fails_before_result(self):
        class Mutating(BlockingModel):
            def forward(self, input_ids, **kwargs):
                self.authority._device_guard = None
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        model, _, _, _, _, authority, _, broker, request = make_owner(
            model=Mutating(), device_guard=lambda: None)
        model.authority = authority
        handle = broker.submit('team', request)
        try:
            outcome = handle.wait(5)
        except BrokerUnavailable:
            outcome = None
        self.assertTrue(outcome is None or outcome.state is not RequestState.SUCCEEDED)
        self.assertTrue(authority.unsafe)

    def test_cuda_worker_rejects_authority_without_owner_lease_guard(self):
        model = BlockingModel().eval()
        resident = ResidentWorker(MODEL_ID, model, device='cpu')
        resident._device = torch.device('cuda:0')  # Constructor-only CPU fake.
        budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                              max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        load = budget.acquire(2048, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        workspace = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with self.assertRaisesRegex(ValueError, 'owner lease guard'):
            LifetimeWorkspaceWorker(resident, budget, load, workspace, profiles=(),
                                    authority=OwnerCompletionAuthority())

    def test_owner_guard_does_not_hold_decision_lock_while_blocked(self):
        armed, entered, release = threading.Event(), threading.Event(), threading.Event()
        def guard():
            if not armed.is_set():
                return
            entered.set()
            if not release.wait(5):
                raise RuntimeError('guard timed out')
        model, _, _, _, _, authority, _, broker, request = make_owner(device_guard=guard)
        armed.set()
        first = broker.submit('team', request)
        try:
            self.assertTrue(entered.wait(5))
            second = broker.submit('team', replace(request, request_id='second'))
            self.assertTrue(second.cancel())
            self.assertEqual(second.wait(1).state, RequestState.CANCELLED)
            authority.mark_unsafe()
        finally:
            release.set()
            model.release.set()
        try:
            first.wait(5)
        except BrokerUnavailable:
            pass

    def test_output_finalizer_runs_outside_authority_decision_lock(self):
        entered, release, condition_held, finalizing = (threading.Event() for _ in range(4))
        observed = {}
        class FinalizerOutput(dict):
            def __del__(self):
                finalizing.set()
                acquired = broker._condition.acquire(timeout=3)
                observed['finalizer_condition'] = acquired
                if acquired:
                    broker._condition.release()
        class FinalizerModel(nn.Module):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('model gate timed out')
                return FinalizerOutput(logits=torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1))
        _, _, _, _, _, authority, _, broker, request = make_owner(model=FinalizerModel())
        handle = broker.submit('team', request)
        self.assertTrue(entered.wait(5))
        def competitor():
            with broker._condition:
                condition_held.set()
                if finalizing.wait(5):
                    acquired = authority._decision_lock.acquire(timeout=2)
                    observed['competitor_authority'] = acquired
                    if acquired:
                        authority._decision_lock.release()
        thread = threading.Thread(target=competitor)
        thread.start()
        try:
            self.assertTrue(condition_held.wait(5))
        finally:
            release.set()
        thread.join(7)
        self.assertFalse(thread.is_alive())
        self.assertEqual(handle.wait(5).state, RequestState.SUCCEEDED)
        self.assertTrue(observed.get('competitor_authority'))
        self.assertTrue(observed.get('finalizer_condition'))
        self.assertTrue(broker.close(timeout=5))

    def test_unsafe_after_telemetry_prevents_new_model_launch(self):
        model, _, _, _, _, authority, worker, broker, request = make_owner()
        check = worker._check_leases
        def unsafe_after_check():
            check()
            authority.mark_unsafe()
        with patch.object(worker, '_check_leases', side_effect=unsafe_after_check):
            handle = broker.submit('team', request)
            try:
                handle.wait(5)
            except BrokerUnavailable:
                pass
        broker._thread.join(5)
        self.assertFalse(model.entered.is_set())
        self.assertTrue(authority.unsafe)

    def test_model_oom_marks_unsafe_before_retiring_input(self):
        observed = {}
        class Oom(BlockingModel):
            def forward(self, input_ids, **kwargs):
                observed['input'] = weakref.ref(input_ids)
                raise torch.OutOfMemoryError('simulated')
        _, resident, budget, load, workspace, authority, _, broker, request = make_owner(model=Oom())
        handle = broker.submit('team', request)
        try:
            handle.wait(5)
        except BrokerUnavailable:
            pass
        broker._thread.join(5)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(observed['input']())
        self.assertIsNotNone(resident._live_buffers[1])
        self.assertEqual(set(budget.active_leases()), {load, workspace})

    def test_owner_unsafe_retains_known_inner_input_and_output(self):
        observed = {}

        class Capturing(BlockingModel):
            def forward(self, input_ids, **kwargs):
                output = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
                observed['input'] = weakref.ref(input_ids)
                observed['output'] = weakref.ref(output)
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError('forward timed out')
                return {'logits': output}

        model, resident, _, _, _, authority, _, broker, request = make_owner(model=Capturing())
        handle = broker.submit('team', request)
        try:
            self.assertTrue(model.entered.wait(5))
            authority.mark_unsafe()
        finally:
            model.release.set()
        try:
            handle.wait(5)
        except BrokerUnavailable:
            pass
        broker._thread.join(5)
        self.assertFalse(broker._thread.is_alive())
        self.assertIsNotNone(observed['input']())
        self.assertIsNotNone(observed['output']())
        self.assertIsNotNone(resident._live_buffers[2])

    def test_broker_cleanup_failure_marks_authority_during_other_forward(self):
        model, _, _, _, _, authority, _, broker, request = make_owner()
        first = broker.submit('team', request)
        try:
            self.assertTrue(model.entered.wait(5))
            second = broker.submit('team', replace(request, request_id='second'))
            with patch.object(broker, '_retire_job', side_effect=RuntimeError('cleanup failed')):
                with self.assertRaises(BrokerUnavailable):
                    second.cancel()
            self.assertTrue(broker.quarantined)
            self.assertTrue(authority.unsafe)
        finally:
            model.release.set()
        try:
            first.wait(5)
        except BrokerUnavailable:
            pass

    def test_cancelled_result_is_rooted_if_unsafe_wins_retirement(self):
        model, _, _, _, _, authority, worker, broker, request = make_owner()
        transferred, resume, retiring, finish_retire = (threading.Event() for _ in range(4))
        seen = {}
        transfer = worker._transfer_result
        retire = broker._retire_job
        def delayed_transfer(result):
            seen['logits'] = weakref.ref(result.logits)
            transfer(result)
            transferred.set()
            if not resume.wait(5):
                raise RuntimeError('transfer timed out')
        def delayed_retire(job):
            retiring.set()
            if not finish_retire.wait(5):
                raise RuntimeError('retirement timed out')
            return retire(job)
        with patch.object(worker, '_transfer_result', delayed_transfer), \
             patch.object(broker, '_retire_job', delayed_retire):
            handle = broker.submit('team', request)
            self.assertTrue(model.entered.wait(5))
            model.release.set()
            try:
                self.assertTrue(transferred.wait(5))
                self.assertTrue(handle.cancel())
                resume.set()
                self.assertTrue(retiring.wait(5))
                authority.mark_unsafe()
            finally:
                resume.set()
                finish_retire.set()
            broker._thread.join(5)
        self.assertIsNotNone(seen['logits']())
        self.assertTrue(any(job is not None for job in broker._job_roots))

    def test_unsafe_before_result_transfer_retains_private_worker_snapshot(self):
        model, _, _, _, _, authority, worker, broker, request = make_owner()
        transfer = worker._transfer_result
        seen = {}
        def unsafe_before_transfer(result):
            seen['snapshot'] = weakref.ref(worker._custody.request.inputs['input_ids'])
            authority.mark_unsafe()
            return transfer(result)
        with patch.object(worker, '_transfer_result', unsafe_before_transfer):
            handle = broker.submit('team', request)
            self.assertTrue(model.entered.wait(5))
            model.release.set()
            try:
                handle.wait(5)
            except BrokerUnavailable:
                pass
            broker._thread.join(5)
        self.assertIsNotNone(seen['snapshot']())
        self.assertIsNotNone(worker._custody.request)
        self.assertTrue(authority.unsafe)

    def test_unsafe_before_cancel_retirement_retains_private_worker_snapshot(self):
        cancel = threading.Event()
        class CancelModel(nn.Module):
            def forward(self, input_ids, **kwargs):
                cancel.set()
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        _, _, _, _, _, authority, worker, _, request = make_owner(model=CancelModel(), cold_warm=True)
        source, first = inspect.getsourcelines(LifetimeWorkspaceWorker.execute)
        decisions = [first + index for index, line in enumerate(source)
                     if line.strip() == 'if (self._authority.unsafe or self._worker.quarantined']
        self.assertEqual(len(decisions), 1)
        refs = []
        def trace(frame, event, arg):
            if (frame.f_code is LifetimeWorkspaceWorker.execute.__code__ and event == 'line'
                    and frame.f_lineno == decisions[0] and not refs):
                refs.append(weakref.ref(worker._custody.request.inputs['input_ids']))
                authority.mark_unsafe()
            return trace
        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(RequestCancelled):
                worker.execute(request, cancel)
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(worker._custody.request)
        self.assertIsNotNone(refs[0]())

    def test_post_forward_cancel_retains_copied_output_when_unsafe_wins(self):
        cancel = threading.Event()
        class CancelModel(nn.Module):
            def forward(self, input_ids, **kwargs):
                cancel.set()
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        _, resident, _, _, _, authority, worker, _, request = make_owner(model=CancelModel(), cold_warm=True)
        state_fn, real_to = resident._model_state, torch.Tensor.to
        observed, states = [], []
        def tracked_to(tensor, *args, **kwargs):
            result = real_to(tensor, *args, **kwargs)
            if kwargs.get('device') == 'cpu' and kwargs.get('copy') is True:
                observed.append(weakref.ref(result))
            return result
        def signal_after_perform():
            result = state_fn()
            states.append(None)
            if len(states) == 2:
                authority.mark_unsafe()
            return result
        with patch.object(torch.Tensor, 'to', tracked_to), \
             patch.object(resident, '_model_state', signal_after_perform):
            with self.assertRaises(RequestCancelled):
                worker.execute(request, cancel)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(observed[0]())
        self.assertIs(worker._custody.logits, observed[0]())

    def test_perform_return_interrupt_retains_copied_output(self):
        class Tiny(nn.Module):
            def forward(self, input_ids, **kwargs):
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        _, _, _, _, _, authority, worker, _, request = make_owner(model=Tiny(), cold_warm=True)
        instructions = list(dis.get_instructions(ResidentWorker._execute))
        positions = [index for index, item in enumerate(instructions)
                     if item.opname in ('LOAD_METHOD', 'LOAD_ATTR') and item.argval == '_perform']
        offsets = {next(item.offset for item in instructions[position:]
                        if item.opname == 'UNPACK_SEQUENCE') for position in positions}
        real_to, refs, injected = torch.Tensor.to, [], []
        def tracked_to(tensor, *args, **kwargs):
            result = real_to(tensor, *args, **kwargs)
            if kwargs.get('device') == 'cpu' and kwargs.get('copy') is True:
                refs.append(weakref.ref(result))
            return result
        def trace(frame, event, arg):
            if frame.f_code is ResidentWorker._execute.__code__:
                frame.f_trace_opcodes = True
                if event == 'opcode' and frame.f_lasti in offsets and not injected:
                    injected.append(True)
                    raise KeyboardInterrupt('before perform result assignment')
            return trace
        previous_trace = sys.gettrace()
        with patch.object(torch.Tensor, 'to', tracked_to):
            try:
                sys.settrace(trace)
                worker.execute(request)
            except KeyboardInterrupt:
                pass
            finally:
                sys.settrace(previous_trace)
        self.assertTrue(injected)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(refs[0]())
        self.assertIs(worker._custody.logits, refs[0]())

    def test_interrupt_before_live_output_slot_retains_known_local_output(self):
        class Tiny(nn.Module):
            def forward(self, input_ids, **kwargs):
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        _, resident, _, _, _, authority, worker, _, request = make_owner(model=Tiny(), cold_warm=True)
        source, first = inspect.getsourcelines(ResidentWorker._perform_with_custody)
        targets = [first + index for index, line in enumerate(source)
                   if line.strip() == 'self._live_buffers[4] = outcome']
        self.assertGreaterEqual(len(targets), 2)
        target = targets[1]  # Normal post-copy assignment, not recovery.
        real_to, refs, injected = torch.Tensor.to, [], []
        def tracked_to(tensor, *args, **kwargs):
            result = real_to(tensor, *args, **kwargs)
            if kwargs.get('device') == 'cpu' and kwargs.get('copy') is True:
                refs.append(weakref.ref(result))
            return result
        def trace(frame, event, arg):
            if (frame.f_code is ResidentWorker._perform_with_custody.__code__ and event == 'line'
                    and frame.f_lineno == target and not injected):
                injected.append(True)
                raise KeyboardInterrupt('before live output assignment')
            return trace
        previous_trace = sys.gettrace()
        with patch.object(torch.Tensor, 'to', tracked_to):
            try:
                sys.settrace(trace)
                worker.execute(request)
            except KeyboardInterrupt:
                pass
            finally:
                sys.settrace(previous_trace)
        self.assertTrue(injected)
        self.assertTrue(authority.unsafe)
        self.assertIsNotNone(refs[0]())
        self.assertIs(worker._custody.logits, refs[0]())
        self.assertIs(resident._live_buffers[4], refs[0]())

    def test_revoked_owner_does_not_start_post_forward_device_operations(self):
        for direct_mark in (False, True):
            with self.subTest(direct_mark=direct_mark):
                state = {'live': True, 'calls': []}
                def guard():
                    if not state['live']:
                        raise RuntimeError('supervisor lease revoked')
                authority = OwnerCompletionAuthority(device_guard=guard)

                class Revoking(nn.Module):
                    def forward(self, input_ids, **kwargs):
                        state['live'] = False
                        if direct_mark:
                            authority.mark_unsafe()
                        return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}

                resident = ResidentWorker(MODEL_ID, Revoking().eval(), device='cpu', max_batch_size=1,
                                          max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
                resident._device = torch.device('cuda:0')  # CPU fakes only, never a real CUDA call.
                resident._owner_completion = authority
                real_to = torch.Tensor.to
                def fake_to(tensor, *args, **kwargs):
                    target = kwargs.get('device', args[0] if args else None)
                    state['calls'].append(('to', str(target), state['live']))
                    return tensor.clone() if str(target).startswith('cuda') else real_to(tensor, *args, **kwargs)
                def fake_sync(device):
                    state['calls'].append(('sync', str(device), state['live']))
                fake_gen = SimpleNamespace(manual_seed=lambda seed: None)
                request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
                with patch.object(resident_module, '_process_quarantined', threading.Event()), \
                     patch.object(resident_module, '_process_quarantine_latch', None), \
                     patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
                     patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
                     patch.object(torch.cuda, 'set_rng_state', side_effect=AssertionError('unsafe CUDA restore')), \
                     patch.object(torch.cuda, 'default_generators', (fake_gen,)), \
                     patch.object(torch.Tensor, 'to', fake_to), \
                     patch.object(torch.cuda, 'synchronize', fake_sync):
                    outcome, failure = resident._perform(request)
                self.assertIsNone(outcome)
                self.assertIsNotNone(failure)
                self.assertTrue(authority.unsafe)
                self.assertFalse(any(not live for _, _, live in state['calls']))
                self.assertIsNotNone(resident._live_buffers[2])

    def test_fixed_owner_synchronizes_after_cuda_rng_restore(self):
        events = []
        class Tiny(nn.Module):
            def forward(self, input_ids, **kwargs):
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        resident = ResidentWorker(MODEL_ID, Tiny().eval(), device='cpu', max_batch_size=1,
                                  max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
        resident._device = torch.device('cuda:0')
        resident._owner_completion = OwnerCompletionAuthority(device_guard=lambda: None)
        real_to = torch.Tensor.to
        def fake_to(tensor, *args, **kwargs):
            target = kwargs.get('device', args[0] if args else None)
            return tensor.clone() if str(target).startswith('cuda') else real_to(tensor, *args, **kwargs)
        def restore(*args):
            events.append(('restore', resident._drain_pending))
        def synchronize(*args):
            events.append(('sync', resident._drain_pending))
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
        with patch.object(resident_module, '_process_quarantined', threading.Event()), \
             patch.object(resident_module, '_process_quarantine_latch', None), \
             patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
             patch.object(torch.cuda, 'set_rng_state', side_effect=restore), \
             patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda seed: None),)), \
             patch.object(torch.Tensor, 'to', fake_to), \
             patch.object(torch.cuda, 'synchronize', side_effect=synchronize):
            result, failure = resident._perform(request)
        self.assertIsNone(failure)
        self.assertIsNotNone(result)
        self.assertEqual(events, [('restore', True), ('sync', True)])
        self.assertFalse(resident._drain_pending)

    def test_cuda_oom_signals_unsafe_before_any_later_sync_or_restore(self):
        events = []
        class Oom(nn.Module):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('simulated')
        resident = ResidentWorker(MODEL_ID, Oom().eval(), device='cpu', max_batch_size=1,
                                  max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
        resident._device = torch.device('cuda:0')
        authority = OwnerCompletionAuthority(device_guard=lambda: None)
        resident._owner_completion = authority
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
        with patch.object(resident_module, '_process_quarantined', threading.Event()), \
             patch.object(resident_module, '_process_quarantine_latch', None), \
             patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
             patch.object(torch.cuda, 'set_rng_state', side_effect=lambda *a: events.append('restore')), \
             patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda seed: None),)), \
             patch.object(torch.Tensor, 'to', lambda tensor, *a, **kw: tensor.clone()), \
             patch.object(torch.cuda, 'synchronize', side_effect=lambda *a: events.append('sync')):
            result, failure = resident._perform(request)
        self.assertIsNone(result)
        self.assertIsNotNone(failure)
        self.assertTrue(authority.unsafe)
        self.assertEqual(events, [])
        self.assertIsNotNone(resident._live_buffers[1])

    def test_interrupted_oom_notification_skips_later_cuda_operations(self):
        events = []
        class Oom(nn.Module):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('simulated')
        resident = ResidentWorker(MODEL_ID, Oom().eval(), device='cpu', max_batch_size=1,
                                  max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
        resident._device = torch.device('cuda:0')
        authority = OwnerCompletionAuthority(device_guard=lambda: None)
        resident._owner_completion = authority
        mark = authority.mark_unsafe
        calls = []
        def interrupted_once():
            calls.append(None)
            if len(calls) == 1:
                raise KeyboardInterrupt('injected')
            mark()
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
        with patch.object(resident_module, '_process_quarantined', threading.Event()), \
             patch.object(resident_module, '_process_quarantine_latch', None), \
             patch.object(authority, 'mark_unsafe', interrupted_once), \
             patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
             patch.object(torch.cuda, 'set_rng_state', side_effect=lambda *a: events.append('restore')), \
             patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda seed: None),)), \
             patch.object(torch.Tensor, 'to', lambda tensor, *a, **kw: tensor.clone()), \
             patch.object(torch.cuda, 'synchronize', side_effect=lambda *a: events.append('sync')):
            try:
                resident._perform(request)
            except KeyboardInterrupt:
                pass
        self.assertTrue(authority.unsafe)
        self.assertTrue(resident._drain_pending)
        self.assertEqual(events, [])
        self.assertIsNotNone(resident._live_buffers[1])

    def test_lease_lost_inside_sync_keeps_drain_and_device_buffers(self):
        state = {'live': True}
        def guard():
            if not state['live']:
                raise RuntimeError('lease lost')
        class Tiny(nn.Module):
            def forward(self, input_ids, **kwargs):
                return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
        resident = ResidentWorker(MODEL_ID, Tiny().eval(), device='cpu', max_batch_size=1,
                                  max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
        resident._device = torch.device('cuda:0')
        authority = OwnerCompletionAuthority(device_guard=guard)
        resident._owner_completion = authority
        real_to = torch.Tensor.to
        def fake_to(tensor, *args, **kwargs):
            target = kwargs.get('device', args[0] if args else None)
            return tensor.clone() if str(target).startswith('cuda') else real_to(tensor, *args, **kwargs)
        def revoke_during_sync(*args):
            state['live'] = False
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
        with patch.object(resident_module, '_process_quarantined', threading.Event()), \
             patch.object(resident_module, '_process_quarantine_latch', None), \
             patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
             patch.object(torch.cuda, 'set_rng_state', return_value=None), \
             patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda seed: None),)), \
             patch.object(torch.Tensor, 'to', fake_to), \
             patch.object(torch.cuda, 'synchronize', side_effect=revoke_during_sync):
            result, failure = resident._perform(request)
        self.assertIsNone(result)
        self.assertIsNotNone(failure)
        self.assertTrue(authority.unsafe)
        self.assertTrue(resident._drain_pending)
        self.assertIsNotNone(resident._live_buffers[1])
        self.assertIsNotNone(resident._live_buffers[2])

    def test_interrupted_signal_at_rng_acquisition_entry_keeps_drain_pending(self):
        resident = ResidentWorker(MODEL_ID, BlockingModel().eval(), device='cpu', max_batch_size=1,
                                  max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
        resident._device = torch.device('cuda:0')
        authority = OwnerCompletionAuthority(device_guard=lambda: None)
        resident._owner_completion = authority
        events, injected = [], []
        def trace(frame, event, arg):
            if event == 'call' and frame.f_code is ResidentWorker._signal_owner_unsafe.__code__ and not injected:
                injected.append(True)
                raise KeyboardInterrupt('before unsafe helper')
            return trace
        def fail_get(*args):
            events.append('get_rng')
            raise torch.OutOfMemoryError('simulated')
        request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
        previous_trace = sys.gettrace()
        with patch.object(resident_module, '_process_quarantined', threading.Event()), \
             patch.object(resident_module, '_process_quarantine_latch', None), \
             patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
             patch.object(torch.cuda, 'get_rng_state', side_effect=fail_get), \
             patch.object(torch.cuda, 'set_rng_state', side_effect=lambda *a: events.append('restore')), \
             patch.object(torch.cuda, 'synchronize', side_effect=lambda *a: events.append('sync')):
            try:
                sys.settrace(trace)
                resident._perform(request)
            except KeyboardInterrupt:
                pass
            finally:
                sys.settrace(previous_trace)
        self.assertTrue(injected)
        self.assertEqual(events, ['get_rng'])
        self.assertTrue(authority.unsafe)
        self.assertTrue(resident._drain_pending)
        self.assertIs(resident._live_buffers[0], request)

    def test_observation_completion_follows_rng_restore_and_final_sync(self):
        for restore_fails in (False, True):
            with self.subTest(restore_fails=restore_fails):
                events = []
                class Tiny(nn.Module):
                    def forward(self, input_ids, **kwargs):
                        return {'logits': torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)}
                resident = ResidentWorker(MODEL_ID, Tiny().eval(), device='cpu', max_batch_size=1,
                                          max_sequence_length=2, max_input_bytes=64, max_output_bytes=64)
                resident._device = torch.device('cuda:0')
                authority = OwnerCompletionAuthority(device_guard=lambda: None)
                resident._owner_completion = authority
                real_to = torch.Tensor.to
                def fake_to(tensor, *args, **kwargs):
                    target = kwargs.get('device', args[0] if args else None)
                    return tensor.clone() if str(target).startswith('cuda') else real_to(tensor, *args, **kwargs)
                def restore(*args):
                    events.append('restore')
                    if restore_fails:
                        raise RuntimeError('restore failed')
                request = ForwardRequest('client', {'input_ids': torch.tensor([[1, 2]])})
                with patch.object(resident_module, '_process_quarantined', threading.Event()), \
                     patch.object(resident_module, '_process_quarantine_latch', None), \
                     patch.object(resident_module, '_finish_forward', side_effect=lambda *a: events.append('finish')), \
                     patch.object(torch.random, 'fork_rng', return_value=nullcontext()), \
                     patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([0])), \
                     patch.object(torch.cuda, 'set_rng_state', side_effect=restore), \
                     patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda seed: None),)), \
                     patch.object(torch.Tensor, 'to', fake_to), \
                     patch.object(torch.cuda, 'synchronize', side_effect=lambda *a: events.append('sync')):
                    result, failure = resident._perform(request)
                if restore_fails:
                    self.assertIsNone(result)
                    self.assertIsNotNone(failure)
                    self.assertTrue(authority.unsafe)
                    self.assertEqual(events, ['restore'])
                else:
                    self.assertIsNone(failure)
                    self.assertEqual(events, ['restore', 'sync', 'finish'])

