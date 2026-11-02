"""CPU-only integration tests of trusted admission, execution and cleanup."""
import threading
import inspect
import dis
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

import looking_glass.resident_worker as resident_worker_module
from looking_glass.broker import Broker, RequestState
from looking_glass.budgeted_worker import (
    BudgetedWorker, DeviceAdmission, DeviceQuarantined, ProfileUnavailable,
    WorkerContractError, WorkloadProfile,
)
from looking_glass.memory_budget import AdmissionError, CudaMemorySnapshot, GIB, MemoryBudget, MemoryLease
from looking_glass.resident_worker import (
    ForwardRequest, ForwardResult, RequestCancelled, ResidentWorker,
    WorkerExecutionError, WorkerOutOfMemory, WorkerStateError, WorkerSynchronizationError,
)


MODEL_ID = 'a' * 64


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        x = input_ids.float()
        return {'logits': torch.stack((x, x + 1), dim=-1)}


def request(request_id='request', values=(1, 2), dtype=torch.int64):
    return ForwardRequest(request_id, {'input_ids': torch.tensor([values], dtype=dtype)})


def profile(**changes):
    values = {
        'profile_id': 'cpu-profile-v1', 'model_id': MODEL_ID, 'device': 'cpu',
        'shape': (1, 2), 'input_dtypes': (('input_ids', 'torch.int64'),),
        'output_limit_bytes': 64, 'estimated_incremental_bytes': 1024,
    }
    values.update(changes)
    return WorkloadProfile(**values)


def make_budget(free=3 * GIB):
    return MemoryBudget(lambda: CudaMemorySnapshot(free, 4 * GIB, 0, 0),
                        max_gpu_trace_bytes=0, max_host_trace_bytes=0)


def make_worker(model=None):
    model = SmallModel() if model is None else model
    return ResidentWorker(MODEL_ID, model.eval(), device='cpu', max_batch_size=2,
                          max_sequence_length=8, max_output_bytes=64)


def make_wrapper(worker=None, budget=None, profiles=None):
    worker = make_worker() if worker is None else worker
    budget = make_budget() if budget is None else budget
    admission = DeviceAdmission(budget, device='cpu', max_estimated_incremental_bytes=4096)
    wrapper = BudgetedWorker(worker, admission, (profile(),) if profiles is None else profiles)
    return wrapper, worker, budget, admission


class BudgetedWorkerTests(unittest.TestCase):
    def test_failed_status_distinguishes_clean_close_from_execution_poison(self):
        healthy, _, _, _ = make_wrapper()
        self.assertFalse(healthy.failed)
        healthy.close()
        self.assertTrue(healthy.poisoned)
        self.assertFalse(healthy.failed)

        class Oom(SmallModel):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('simulated')

        failed, _, _, _ = make_wrapper(make_worker(Oom()))
        with self.assertRaises(WorkerOutOfMemory):
            failed.execute(request())
        self.assertTrue(failed.failed)
        failed.close()
        self.assertTrue(failed.failed)

    def test_exposes_bound_input_limit_read_only_and_rejects_changed_binding(self):
        wrapper, worker, _, _ = make_wrapper()
        self.assertEqual(wrapper.max_input_bytes, worker.max_input_bytes)
        with self.assertRaises(AttributeError):
            wrapper.max_input_bytes = 1
        worker._model_id = 'b' * 64
        with self.assertRaises(Exception):
            _ = wrapper.max_input_bytes
        self.assertTrue(wrapper.poisoned)

    def test_exposes_bound_output_limit_read_only_and_rejects_changed_binding(self):
        wrapper, worker, _, _ = make_wrapper()
        self.assertEqual(wrapper.max_output_bytes, worker.max_output_bytes)
        with self.assertRaises(AttributeError):
            wrapper.max_output_bytes = 1
        worker._model_id = 'b' * 64
        with self.assertRaises(Exception):
            _ = wrapper.max_output_bytes
        self.assertTrue(wrapper.poisoned)

    def test_real_worker_success_releases_lease_only_after_forward_returns(self):
        entered, finish = threading.Event(), threading.Event()

        class Blocking(SmallModel):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timed out')
                return super().forward(input_ids, **kwargs)

        wrapper, _, budget, _ = make_wrapper(make_worker(Blocking()))
        results, errors = [], []

        def run():
            try:
                results.append(wrapper.execute(request()))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(budget.snapshot().active_leases, 1)
            self.assertFalse(wrapper.quarantined, 'a running forward is not stale')
            self.assertEqual(budget.snapshot().committed_incremental_bytes, 1024)
        finally:
            finish.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(results[0].logits, torch.tensor([[[1., 2.], [2., 3.]]])))
        self.assertEqual(results[0].logits.device.type, 'cpu')
        self.assertEqual(budget.snapshot().active_leases, 0)
        wrapper.close()

    def test_missing_shape_or_dtype_profile_never_executes_or_acquires(self):
        model = SmallModel()
        wrapper, _, budget, _ = make_wrapper(make_worker(model))
        for bad in (request(values=(1,)), request(dtype=torch.int32)):
            with self.subTest(shape=bad.inputs['input_ids'].shape, dtype=bad.inputs['input_ids'].dtype):
                with self.assertRaises(ProfileUnavailable):
                    wrapper.execute(bad)
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_admission_uses_live_telemetry_before_any_model_work(self):
        model = SmallModel()
        wrapper, _, budget, _ = make_wrapper(make_worker(model), make_budget(free=GIB))
        with self.assertRaises(AdmissionError):
            wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_ambient_cpu_or_cuda_autocast_is_rejected_before_admission(self):
        for device_type in ('cpu', 'cuda'):
            model = SmallModel()
            wrapper, _, budget, _ = make_wrapper(make_worker(model))
            # CUDA autocast is simulated, so this test never initializes a GPU.
            fake_state = lambda name: name == device_type
            with self.subTest(device_type=device_type):
                with patch('torch.is_autocast_enabled', side_effect=fake_state):
                    with self.assertRaisesRegex(RuntimeError, 'autocast'):
                        wrapper.execute(request())
                self.assertEqual(model.calls, 0)
                self.assertEqual(budget.snapshot().active_leases, 0)

    def test_actual_cpu_autocast_is_rejected_before_forward(self):
        model = SmallModel()
        wrapper, _, budget, _ = make_wrapper(make_worker(model))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, 'autocast'):
                wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_enabled_model_cache_is_rejected_or_detected_before_admission(self):
        model = SmallModel()
        model.config = SimpleNamespace(use_cache=True)
        with self.assertRaisesRegex(ValueError, 'cache'):
            make_wrapper(make_worker(model))

        model.config.use_cache = False
        wrapper, _, budget, _ = make_wrapper(make_worker(model))
        model.config.use_cache = True
        with self.assertRaisesRegex(Exception, 'cache|identity|limits'):
            wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_cache_policy_change_during_forward_quarantines_lease(self):
        class Mutating(SmallModel):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)

            def forward(self, input_ids, **kwargs):
                self.config.use_cache = True
                return super().forward(input_ids, **kwargs)

        wrapper, _, budget, admission = make_wrapper(make_worker(Mutating()))
        with self.assertRaises(WorkerContractError):
            wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_unexpected_returned_cache_quarantines_the_profile_lease(self):
        class UnexpectedCache(SmallModel):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)

            def forward(self, input_ids, **kwargs):
                output = super().forward(input_ids, **kwargs)
                output['past_key_values'] = (torch.ones(1),)
                return output

        wrapper, _, budget, admission = make_wrapper(make_worker(UnexpectedCache()))
        with self.assertRaises(WorkerContractError):
            wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_cache_change_during_admission_prevents_forward_and_releases_unused_lease(self):
        model = SmallModel()
        model.config = SimpleNamespace(use_cache=False)

        def telemetry():
            model.config.use_cache = True
            return CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0)

        budget = MemoryBudget(telemetry, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        wrapper, _, _, _ = make_wrapper(make_worker(model), budget)
        with self.assertRaisesRegex(Exception, 'cache|identity|limits'):
            wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertEqual(len(budget.active_leases()), 0)

    def test_unreadable_cache_policy_after_forward_quarantines_lease(self):
        class Unreadable:
            @property
            def use_cache(self):
                raise RuntimeError('cache state unavailable')

        class Mutating(SmallModel):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)

            def forward(self, input_ids, **kwargs):
                self.config = Unreadable()
                return super().forward(input_ids, **kwargs)

        wrapper, _, budget, admission = make_wrapper(make_worker(Mutating()))
        with self.assertRaises(WorkerContractError):
            wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_changed_worker_binding_fails_before_admission_and_execution(self):
        for attribute, replacement in (('model_id', 'b' * 64), ('device', torch.device('cuda:0')),
                                       ('max_output_bytes', 128), ('max_sequence_length', 16)):
            model = SmallModel()
            wrapper, worker, budget, _ = make_wrapper(make_worker(model))
            setattr(worker, '_' + attribute, replacement)
            with self.subTest(attribute=attribute), self.assertRaises(Exception):
                wrapper.execute(request())
            self.assertEqual(model.calls, 0)
            self.assertEqual(budget.snapshot().active_leases, 0)
            self.assertTrue(wrapper.poisoned)

    def test_changed_admission_configuration_fails_before_execution(self):
        for attribute, replacement in (('device', torch.device('cuda:0')),
                                       ('max_estimated_incremental_bytes', 512)):
            model = SmallModel()
            wrapper, _, budget, admission = make_wrapper(make_worker(model))
            setattr(admission, attribute, replacement)
            with self.subTest(attribute=attribute), self.assertRaises(Exception):
                wrapper.execute(request())
            self.assertEqual(model.calls, 0)
            self.assertEqual(budget.snapshot().active_leases, 0)

    def test_binding_changed_during_forward_quarantines_lease(self):
        entered, finish = threading.Event(), threading.Event()

        class Blocking(SmallModel):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timed out')
                return super().forward(input_ids, **kwargs)

        wrapper, worker, budget, admission = make_wrapper(make_worker(Blocking()))
        errors = []

        def run():
            try:
                wrapper.execute(request())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            worker._max_output_bytes = 128
        finally:
            finish.set()
            thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], WorkerContractError)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_profile_cannot_exceed_hard_caps_or_mismatch_model_and_device(self):
        worker = make_worker()
        admission = DeviceAdmission(make_budget(), device='cpu', max_estimated_incremental_bytes=4096)
        invalid = (
            (), (profile(model_id='b' * 64),), (profile(device='cuda:0'),),
            (profile(estimated_incremental_bytes=4097),),
            (profile(estimated_incremental_bytes=0),),
            (profile(estimated_incremental_bytes=64),),
            (profile(output_limit_bytes=32),), (profile(shape=(3, 2)),),
            (profile(), profile(profile_id='duplicate-signature')),
        )
        for profiles in invalid:
            with self.subTest(profiles=profiles), self.assertRaises((TypeError, ValueError)):
                BudgetedWorker(worker, admission, profiles)

    def test_running_cancellation_keeps_lease_until_worker_is_drained(self):
        entered, finish, cancelled = threading.Event(), threading.Event(), threading.Event()

        class Blocking(SmallModel):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timed out')
                return super().forward(input_ids, **kwargs)

        wrapper, worker, budget, _ = make_wrapper(make_worker(Blocking()))
        errors = []

        def run():
            try:
                wrapper.execute(request(), cancelled)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            cancelled.set()
            self.assertEqual(budget.snapshot().active_leases, 1)
        finally:
            finish.set()
            thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelled)
        self.assertFalse(worker.poisoned)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_oom_after_verified_drain_releases_lease_but_poisoned_worker_stays_unavailable(self):
        class Oom(SmallModel):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('simulated')

        wrapper, worker, budget, admission = make_wrapper(make_worker(Oom()))
        with self.assertRaises(WorkerOutOfMemory):
            wrapper.execute(request())
        self.assertTrue(worker.poisoned)
        self.assertTrue(wrapper.poisoned)
        self.assertFalse(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_unexpected_interrupted_acquire_quarantines_and_keeps_unknown_lease(self):
        model = SmallModel()
        wrapper, _, budget, admission = make_wrapper(make_worker(model))
        acquire = budget.try_acquire

        def interrupt_after_registration(*args, **kwargs):
            acquire(*args, **kwargs)
            raise KeyboardInterrupt('acquire completion unknown')

        with patch.object(budget, 'try_acquire', side_effect=interrupt_after_registration):
            with self.assertRaises(KeyboardInterrupt):
                wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)
        with self.assertRaises(DeviceQuarantined):
            wrapper.execute(request('later'))

    def test_admission_error_after_publication_is_not_mistaken_for_rejection(self):
        wrapper, _, budget, admission = make_wrapper()
        acquire = budget.try_acquire

        def fail_after_registration(*args, **kwargs):
            acquire(*args, **kwargs)
            raise AdmissionError('interrupted after registration')

        with patch.object(budget, 'try_acquire', side_effect=fail_after_registration):
            with self.assertRaises(AdmissionError):
                wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_ordinary_admission_rejection_does_not_quarantine(self):
        model = SmallModel()
        wrapper, _, budget, admission = make_wrapper(make_worker(model))
        with patch.object(budget, 'try_acquire', return_value=None):
            with self.assertRaises(AdmissionError):
                wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertFalse(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_uncertain_acquire_does_not_wait_for_unavailable_budget_lock(self):
        model = SmallModel()
        wrapper, _, budget, admission = make_wrapper(make_worker(model))
        locked, release, finished = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def hold_ledger_lock():
            with budget._lock:
                locked.set()
                release.wait(5)

        def execute():
            try:
                wrapper.execute(request())
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        holder = threading.Thread(target=hold_ledger_lock, daemon=True)
        holder.start()
        self.assertTrue(locked.wait(5))
        runner = threading.Thread(target=execute, daemon=True)
        try:
            with patch.object(budget, 'try_acquire', side_effect=KeyboardInterrupt('unknown ledger state')):
                runner.start()
                self.assertTrue(finished.wait(2), 'quarantine attempted to acquire the budget lock')
                self.assertTrue(admission.quarantined)
                with self.assertRaises(DeviceQuarantined):
                    wrapper.execute(request('later'))
        finally:
            release.set()
            holder.join(5)
            runner.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], KeyboardInterrupt)
        self.assertEqual(model.calls, 0)

    def test_interruption_across_lease_lifetime_quarantines(self):
        source, first = inspect.getsourcelines(BudgetedWorker._execute)
        markers = ('if self._process_is_quarantined():', 'self._release(lease)',
                   'try-before-precheck', 'try-before-forward')
        for marker in markers:
            if marker.startswith('try-before-'):
                following = 'self._check_available()' if marker.endswith('precheck') else 'result = self._worker.execute(snapshot, cancel_event)'
                offset = next(i for i, line in enumerate(source[:-1])
                              if line.strip() == 'try:' and source[i + 1].strip() == following)
            else:
                offset = max(i for i, line in enumerate(source) if line.strip() == marker)
            target = first + offset
            wrapper, _, budget, admission = make_wrapper()

            def trace(frame, event, arg):
                if frame.f_code is BudgetedWorker._execute.__code__ and event == 'line' and frame.f_lineno == target:
                    sys.settrace(None)
                    raise KeyboardInterrupt('lease lifetime interrupted')
                return trace

            with self.subTest(marker=marker):
                try:
                    sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        wrapper.execute(request())
                finally:
                    sys.settrace(None)
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)
                with self.assertRaises(DeviceQuarantined):
                    wrapper.execute(request('later'))

    def test_stranded_marker_is_visible_to_health_and_rejects_other_workers(self):
        wrapper, worker, budget, admission = make_wrapper()
        other = BudgetedWorker(make_worker(), admission, (profile(),))
        admission._lease_inflight = True  # Simulate a gate exit with no receipt.
        self.assertTrue(admission.quarantined)
        self.assertTrue(wrapper.quarantined)
        with self.assertRaises(DeviceQuarantined):
            other.execute(request('later'))
        self.assertEqual(budget.snapshot().active_leases, 0)
        self.assertFalse(worker.poisoned)

    def test_close_retains_model_if_an_admission_marker_was_stranded(self):
        model = SmallModel()
        wrapper, worker, budget, admission = make_wrapper(make_worker(model))
        lease = budget.acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        admission._lease_inflight = True
        wrapper.close()
        self.assertTrue(admission.quarantined)
        self.assertIs(worker._model, model)
        self.assertEqual(budget.snapshot().active_leases, 1)
        self.assertFalse(lease.released)
        held_count = len(admission._holds)
        for _ in range(100):
            wrapper.close()
        self.assertEqual(len(admission._holds), held_count)

    def test_interrupted_gate_cleanup_then_close_retains_model_and_lease(self):
        model = SmallModel()
        wrapper, worker, budget, admission = make_wrapper(make_worker(model))
        source, first = inspect.getsourcelines(BudgetedWorker._gate.__wrapped__)
        target = first + max(i for i, line in enumerate(source)
                             if line.strip() == 'if self._admission._lease_inflight:')
        fired = []

        def trace(frame, event, arg):
            if (frame.f_code is BudgetedWorker._gate.__wrapped__.__code__
                    and event == 'line' and frame.f_lineno == target):
                sys.settrace(None)
                fired.append(True)
                raise KeyboardInterrupt('gate cleanup interrupted')
            return trace

        try:
            with self.assertRaises(KeyboardInterrupt):
                with wrapper._gate(None):
                    lease = budget.try_acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
                    admission._lease_inflight = True
                    sys.settrace(trace)
        finally:
            sys.settrace(None)
        self.assertEqual(fired, [True])
        self.assertTrue(wrapper.quarantined)
        wrapper.close()
        self.assertTrue(admission._quarantined.is_set())
        self.assertIs(worker._model, model)
        self.assertFalse(lease.released)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_interrupted_gate_entry_releases_only_the_acquired_gate(self):
        model = SmallModel()
        wrapper, _, budget, admission = make_wrapper(make_worker(model))
        source, first = inspect.getsourcelines(BudgetedWorker._gate.__wrapped__)
        target = first + next(i for i, line in enumerate(source)
                              if line.strip() == 'if self._admission._lease_inflight:')
        fired = []

        def trace(frame, event, arg):
            if (frame.f_code is BudgetedWorker._gate.__wrapped__.__code__
                    and event == 'line' and frame.f_lineno == target):
                sys.settrace(None)
                fired.append(True)
                raise RuntimeError('gate entry interrupted')
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaisesRegex(RuntimeError, 'gate entry interrupted'):
                wrapper.execute(request())
        finally:
            sys.settrace(None)
        self.assertEqual(fired, [True])
        self.assertFalse(admission._lock.locked())
        self.assertFalse(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(wrapper.execute(request('later')).request_id, 'later')

    def test_interrupted_gate_acquire_is_reported_as_uncertain(self):
        wrapper, _, budget, admission = make_wrapper()
        original_lock = admission._lock

        class InterruptingLock:
            def acquire(self, *args, **kwargs):
                acquired = original_lock.acquire(*args, **kwargs)
                if 'timeout' in kwargs and acquired:
                    raise RuntimeError('gate acquire return interrupted')
                return acquired

            def release(self):
                return original_lock.release()

            def locked(self):
                return original_lock.locked()

        admission._lock = InterruptingLock()
        try:
            with self.assertRaisesRegex(RuntimeError, 'gate acquire return interrupted'):
                wrapper.execute(request())
            self.assertTrue(admission.quarantined)
            self.assertTrue(wrapper.quarantined)
            self.assertEqual(budget.snapshot().active_leases, 0)
            with self.assertRaises(DeviceQuarantined):
                wrapper.execute(request('later'))
        finally:
            # The injected test knows no other owner held the real gate.
            if original_lock.locked():
                original_lock.release()

    def test_opcode_interrupt_at_gate_acquire_return_when_supported(self):
        def probe():
            return 1

        seen = []

        def trace_probe(frame, event, arg):
            if frame.f_code is not probe.__code__:
                return None
            if event == 'call':
                frame.f_trace_opcodes = True
            if event == 'opcode':
                seen.append(True)
            return trace_probe

        previous_trace = sys.gettrace()
        trace_frame = sys._getframe()
        previous_opcodes = trace_frame.f_trace_opcodes
        # CPython 3.12 needs a frame opted into opcodes before settrace().
        trace_frame.f_trace_opcodes = True
        try:
            sys.settrace(trace_probe)
            probe()
        finally:
            sys.settrace(previous_trace)
            trace_frame.f_trace_opcodes = previous_opcodes
            del trace_frame
        if not seen:
            self.skipTest('interpreter does not emit opcode trace events')

        wrapper, _, budget, admission = make_wrapper()
        code = BudgetedWorker._gate.__wrapped__.__code__
        stores = [ins.offset for ins in dis.get_instructions(code)
                  if ins.opname == 'STORE_FAST' and ins.argval == 'acquired']
        self.assertGreaterEqual(len(stores), 2)
        target = stores[1]
        fired = []

        def trace(frame, event, arg):
            if frame.f_code is code and event == 'call':
                frame.f_trace_opcodes = True
            if frame.f_code is code and event == 'opcode' and frame.f_lasti == target:
                sys.settrace(None)
                fired.append(True)
                raise RuntimeError('gate acquire return interrupted')
            return trace

        trace_frame = sys._getframe()
        previous_opcodes = trace_frame.f_trace_opcodes
        trace_frame.f_trace_opcodes = True
        try:
            try:
                sys.settrace(trace)
                with self.assertRaisesRegex(RuntimeError, 'gate acquire return interrupted'):
                    wrapper.execute(request())
            finally:
                sys.settrace(previous_trace)
                trace_frame.f_trace_opcodes = previous_opcodes
                del trace_frame
            self.assertEqual(fired, [True])
            self.assertTrue(admission.quarantined)
            self.assertTrue(wrapper.quarantined)
            self.assertEqual(budget.snapshot().active_leases, 0)
            with self.assertRaises(DeviceQuarantined):
                wrapper.execute(request('later'))
        finally:
            # Test-only recovery: this probe knows no other owner held the gate.
            if admission._lock.locked():
                admission._lock.release()

    def test_interrupt_after_release_but_before_return_quarantines(self):
        wrapper, _, budget, admission = make_wrapper()

        def release_then_interrupt(lease, original=None):
            lease.release()
            raise KeyboardInterrupt('release completion uncertain')

        with patch.object(wrapper, '_release', side_effect=release_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_state_mutation_before_failed_forward_quarantines_lease(self):
        for exception, classified in ((torch.OutOfMemoryError('injected oom'), 'WorkerOutOfMemory'),
                                      (RuntimeError('injected failure'), 'WorkerExecutionError')):
            with self.subTest(exception=type(exception).__name__):
                class MutatingThenFail(SmallModel):
                    def __init__(self):
                        super().__init__()
                        self.config = SimpleNamespace(use_cache=False)
                        self.register_buffer('counter', torch.zeros(1))

                    def forward(self, input_ids, **kwargs):
                        self.counter.add_(1)
                        self.config.use_cache = True
                        raise exception

                model = MutatingThenFail()
                wrapper, worker, budget, admission = make_wrapper(make_worker(model))
                with self.assertRaises(WorkerContractError) as caught:
                    wrapper.execute(request())
                cause = caught.exception.__cause__
                self.assertIsInstance(cause, WorkerStateError)
                self.assertIn(classified, str(cause))
                self.assertIn('injected', str(cause))
                self.assertTrue(worker.poisoned)
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)
                self.assertTrue(model.config.use_cache)

    def test_unreadable_post_state_after_failure_quarantines_lease(self):
        class Oom(SmallModel):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('injected oom')

        wrapper, worker, budget, admission = make_wrapper(make_worker(Oom()))
        with patch.object(worker, '_model_state', side_effect=[worker._baseline, RuntimeError('unknown state')]):
            with self.assertRaises(WorkerContractError) as caught:
                wrapper.execute(request())
        self.assertIsInstance(caught.exception.__cause__, WorkerStateError)
        self.assertIn('WorkerOutOfMemory', str(caught.exception.__cause__))
        self.assertIn('injected oom', str(caught.exception.__cause__))
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_broker_reports_state_violation_and_original_oom(self):
        class MutatingOom(SmallModel):
            def __init__(self):
                super().__init__()
                self.register_buffer('counter', torch.zeros(1))

            def forward(self, input_ids, **kwargs):
                self.counter.add_(1)
                raise torch.OutOfMemoryError('injected oom ' + 'x' * 10000)

        wrapper, _, budget, admission = make_wrapper(make_worker(MutatingOom()))
        with Broker(wrapper, max_pending=2, max_pending_per_team=1) as broker:
            outcome = broker.submit('team', request()).wait(5)
        self.assertEqual(outcome.state, RequestState.FAILED)
        self.assertEqual(outcome.error_type, 'WorkerContractError')
        self.assertIn('WorkerStateError', outcome.error_message)
        self.assertIn('WorkerOutOfMemory', outcome.error_message)
        self.assertIn('injected oom', outcome.error_message)
        self.assertLessEqual(len(outcome.error_message), 1200)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_unchanged_failed_forward_releases_lease(self):
        class Failing(SmallModel):
            def forward(self, input_ids, **kwargs):
                raise RuntimeError('injected failure')

        wrapper, worker, budget, admission = make_wrapper(make_worker(Failing()))
        with self.assertRaisesRegex(WorkerExecutionError, 'injected failure'):
            wrapper.execute(request())
        self.assertTrue(worker.poisoned)
        self.assertFalse(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_broker_reports_oom_and_cleanup_failure_without_losing_either(self):
        class Oom(SmallModel):
            def forward(self, input_ids, **kwargs):
                raise torch.OutOfMemoryError('simulated allocation')

        wrapper, _, budget, admission = make_wrapper(make_worker(Oom()))
        with patch.object(MemoryLease, 'release', side_effect=RuntimeError('ledger failure')):
            with Broker(wrapper, max_pending=2, max_pending_per_team=1) as broker:
                outcome = broker.submit('team', request()).wait(5)
        self.assertEqual(outcome.state, RequestState.FAILED)
        self.assertEqual(outcome.error_type, 'WorkerContractError')
        self.assertIn('WorkerOutOfMemory', outcome.error_message)
        self.assertIn('simulated allocation', outcome.error_message)
        self.assertIn('RuntimeError', outcome.error_message)
        self.assertIn('ledger failure', outcome.error_message)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_failed_synchronization_quarantines_lease_and_all_wrappers_on_device(self):
        wrapper, worker, budget, admission = make_wrapper()
        second_worker = make_worker()
        second = BudgetedWorker(second_worker, admission, (profile(),))
        with patch.object(worker, 'execute', side_effect=WorkerSynchronizationError('device drain failed')):
            with self.assertRaises(WorkerSynchronizationError):
                wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertTrue(wrapper.quarantined)
        self.assertTrue(wrapper.poisoned)
        self.assertTrue(second.poisoned)
        self.assertEqual(budget.snapshot().active_leases, 1)
        with self.assertRaises(DeviceQuarantined):
            second.execute(request())
        wrapper.close()
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_worker_process_quarantine_latch_overrides_masked_error_or_success(self):
        for should_raise in (True, False):
            wrapper, worker, budget, admission = make_wrapper()
            latched = threading.Event()

            def finish_with_latch(snapshot, cancel_event):
                latched.set()
                if should_raise:
                    raise WorkerExecutionError('masked drain error')
                return ForwardResult(snapshot.request_id, torch.ones(1, 2, 2))

            with self.subTest(should_raise=should_raise):
                with patch.object(ResidentWorker, 'quarantined', new_callable=PropertyMock) as state:
                    state.side_effect = latched.is_set
                    with patch.object(worker, 'execute', side_effect=finish_with_latch):
                        with self.assertRaises(WorkerSynchronizationError):
                            wrapper.execute(request())
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)

    def test_peer_quarantine_during_forward_retains_lease_and_suppresses_result(self):
        for should_raise in (False, True):
            with self.subTest(should_raise=should_raise):
                wrapper, worker, budget, admission = make_wrapper()
                entered, finish = threading.Event(), threading.Event()
                results, errors = [], []

                def blocked_forward(snapshot, cancel_event):
                    entered.set()
                    self.assertTrue(finish.wait(5))
                    if should_raise:
                        raise WorkerExecutionError('verified worker failure')
                    return ForwardResult(snapshot.request_id, torch.ones(1, 2, 2))

                def run():
                    try:
                        results.append(wrapper.execute(request()))
                    except BaseException as exc:
                        errors.append(exc)

                with patch.object(worker, 'execute', side_effect=blocked_forward):
                    thread = threading.Thread(target=run)
                    thread.start()
                    try:
                        self.assertTrue(entered.wait(5))
                        self.assertEqual(budget.snapshot().active_leases, 1)
                        admission._quarantined.set()
                    finally:
                        finish.set()
                        thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(results, [])
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], DeviceQuarantined)
                if should_raise:
                    self.assertIn('WorkerExecutionError', str(errors[0]))
                    self.assertIn('verified worker failure', str(errors[0]))
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)

    def test_peer_quarantine_before_execution_retains_acquired_lease(self):
        model = SmallModel()
        wrapper, _, budget, admission = make_wrapper(make_worker(model))
        original = budget.try_acquire

        def acquire_and_quarantine(*args, **kwargs):
            lease = original(*args, **kwargs)
            admission._quarantined.set()
            return lease

        with patch.object(budget, 'try_acquire', side_effect=acquire_and_quarantine):
            with self.assertRaises(DeviceQuarantined):
                wrapper.execute(request())
        self.assertEqual(model.calls, 0)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_failed_quarantine_notification_still_stops_running_peer(self):
        entered, finish = threading.Event(), threading.Event()
        results, errors = [], []

        class Blocking(SmallModel):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timed out')
                return super().forward(input_ids, **kwargs)

        first, _, budget, admission = make_wrapper(make_worker(Blocking()))
        second = BudgetedWorker(make_worker(), admission, (profile(),))
        original_lock = admission._lock

        class InterruptedWaitLock:
            def acquire(self, *args, **kwargs):
                acquired = original_lock.acquire(*args, **kwargs)
                if 'timeout' in kwargs and not acquired:
                    raise KeyboardInterrupt('gate acquisition result unavailable')
                return acquired

            def release(self):
                return original_lock.release()

            def locked(self):
                return original_lock.locked()

        admission._lock = InterruptedWaitLock()

        def run(worker, name):
            try:
                results.append(worker.execute(request(name)))
            except BaseException as exc:
                errors.append(exc)

        a = threading.Thread(target=run, args=(first, 'first'))
        b = threading.Thread(target=run, args=(second, 'second'))
        with patch.object(admission._quarantined, 'set', side_effect=MemoryError('notification failed')):
            a.start()
            try:
                self.assertTrue(entered.wait(5))
                b.start()
                b.join(5)
                self.assertFalse(b.is_alive())
            finally:
                finish.set()
                a.join(5)
        self.assertFalse(a.is_alive())
        self.assertEqual(results, [])
        self.assertTrue(any(isinstance(exc, KeyboardInterrupt) for exc in errors))
        self.assertTrue(any(isinstance(exc, DeviceQuarantined) for exc in errors))
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_process_quarantine_survives_failed_admission_notification(self):
        wrapper, worker, budget, admission = make_wrapper()
        with patch.object(ResidentWorker, 'quarantined', new_callable=PropertyMock, return_value=True):
            with patch.object(admission._quarantined, 'set', side_effect=MemoryError('notification failed')):
                with self.assertRaises(DeviceQuarantined):
                    wrapper.execute(request())
                wrapper.close()
        self.assertTrue(admission.quarantined)
        self.assertIsNotNone(worker._model)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_close_latches_stranded_lease_when_notification_fails(self):
        wrapper, worker, budget, admission = make_wrapper()
        lease = budget.try_acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        admission._lease_inflight = True
        with patch.object(admission._quarantined, 'set', side_effect=MemoryError('notification failed')):
            wrapper.close()
        self.assertTrue(admission.quarantined)
        self.assertIsNotNone(worker._model)
        self.assertEqual(budget.snapshot().active_leases, 1)
        self.assertIsNotNone(lease)

    def test_interrupted_notification_propagates_with_sticky_quarantine(self):
        wrapper, worker, budget, admission = make_wrapper()
        lease = budget.try_acquire(1024, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        admission._lease_inflight = True
        with patch.object(admission._quarantined, 'set', side_effect=KeyboardInterrupt('notification interrupted')):
            with self.assertRaises(KeyboardInterrupt):
                wrapper.close()
        self.assertTrue(admission.quarantined)
        self.assertIsNotNone(worker._model)
        self.assertEqual(budget.snapshot().active_leases, 1)
        self.assertIsNotNone(lease)

    def test_peer_quarantine_after_release_suppresses_result(self):
        wrapper, _, budget, admission = make_wrapper()
        original = wrapper._release

        def release_and_quarantine(lease, original_error=None):
            original(lease, original_error)
            admission._quarantined.set()

        with patch.object(wrapper, '_release', side_effect=release_and_quarantine):
            with self.assertRaises(DeviceQuarantined):
                wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_health_poll_does_not_acquire_the_admission_event_lock(self):
        wrapper, _, _, admission = make_wrapper()
        finished = threading.Event()
        seen = []

        def poll():
            seen.append((wrapper.quarantined, wrapper.failed, wrapper.poisoned))
            finished.set()

        with patch.object(ResidentWorker, 'quarantined', new_callable=PropertyMock, return_value=True):
            with admission._quarantined._cond:
                reader = threading.Thread(target=poll, daemon=True)
                reader.start()
                self.assertTrue(finished.wait(2), 'health poll blocked on admission event lock')
            reader.join(5)
        self.assertEqual(seen, [(True, True, True)])

    def test_repeated_rejection_after_process_quarantine_does_not_accumulate_holds(self):
        wrapper, _, budget, admission = make_wrapper()
        with patch.object(ResidentWorker, 'quarantined', new_callable=PropertyMock, return_value=True):
            for _ in range(50):
                with self.assertRaises(DeviceQuarantined):
                    wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(len(admission._holds), 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_idle_broker_close_honors_external_process_quarantine(self):
        wrapper, worker, _, admission = make_wrapper()
        broker = Broker(wrapper, max_pending=2, max_pending_per_team=1)
        with patch.object(ResidentWorker, 'quarantined', new_callable=PropertyMock, return_value=True):
            self.assertTrue(broker.close())
        self.assertTrue(admission.quarantined)
        # A CPU-only check that close retained the worker instead of disposing it.
        self.assertTrue(torch.equal(worker.execute(request()).logits,
                                    torch.tensor([[[1., 2.], [2., 3.]]])))
        worker.close()

    def test_unclassified_exception_and_contract_violation_quarantine(self):
        for failure in (RuntimeError('unknown'), KeyboardInterrupt('unknown'),
                        ForwardResult('request', torch.ones(1, 100, 2)),
                        ForwardResult('wrong-id', torch.ones(1, 2, 2))):
            with self.subTest(failure=type(failure).__name__):
                wrapper, worker, budget, admission = make_wrapper()
                if isinstance(failure, BaseException):
                    mocked = patch.object(worker, 'execute', side_effect=failure)
                    expected = type(failure)
                else:
                    mocked = patch.object(worker, 'execute', return_value=failure)
                    expected = WorkerContractError
                with mocked, self.assertRaises(expected):
                    wrapper.execute(request())
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)

    def test_quarantine_flag_survives_failure_to_retain_additional_references(self):
        class StorageExhausted:
            def append(self, value):
                raise MemoryError('cannot retain reference')

        wrapper, worker, budget, admission = make_wrapper()
        other = BudgetedWorker(make_worker(), admission, (profile(),))
        admission._holds = StorageExhausted()
        with patch.object(worker, 'execute', side_effect=WorkerSynchronizationError('unverified')):
            with self.assertRaises(WorkerSynchronizationError):
                wrapper.execute(request())
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)
        with self.assertRaises(DeviceQuarantined):
            other.execute(request())

    def test_device_waiter_can_cancel_without_blocking_poison_status(self):
        entered, finish, cancelled, queried = (threading.Event() for _ in range(4))

        class Blocking(SmallModel):
            def forward(self, input_ids, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timed out')
                return super().forward(input_ids, **kwargs)

        first, _, budget, admission = make_wrapper(make_worker(Blocking()))
        second_model = SmallModel()
        second = BudgetedWorker(make_worker(second_model), admission, (profile(),))
        errors = []

        def run_first():
            try:
                first.execute(request('first'))
            except BaseException as exc:
                errors.append(exc)

        def run_second():
            try:
                second.execute(request('second'), cancelled)
            except BaseException as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run_first)
        t2 = threading.Thread(target=run_second)
        status = threading.Thread(target=lambda: (second.poisoned, queried.set()))
        t1.start()
        self.assertTrue(entered.wait(5))
        t2.start()
        status.start()
        try:
            self.assertTrue(queried.wait(1), 'poisoned property blocked on running execution')
            cancelled.set()
            t2.join(1)
            self.assertFalse(t2.is_alive(), 'cancelled waiter stayed blocked on the device gate')
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RequestCancelled)
            self.assertTrue(t1.is_alive(), 'the running forward must still be blocked')
            self.assertTrue(admission._lock.locked(), 'cancelled waiter released another owner\'s gate')
        finally:
            finish.set()
            for thread in (t1, t2, status):
                thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelled)
        self.assertEqual(second_model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_broker_executes_two_teams_with_real_worker_and_budget(self):
        wrapper, _, budget, _ = make_wrapper()
        with Broker(wrapper, max_pending=4, max_pending_per_team=2) as broker:
            first = broker.submit('team-a', request('original-a', (1, 2)))
            second = broker.submit('team-b', request('original-b', (3, 4)))
            one, two = first.wait(5), second.wait(5)
        self.assertEqual((one.state, two.state), (RequestState.SUCCEEDED, RequestState.SUCCEEDED))
        self.assertEqual((one.result.request_id, two.result.request_id), ('original-a', 'original-b'))
        self.assertTrue(torch.equal(one.result.logits, torch.tensor([[[1., 2.], [2., 3.]]])))
        self.assertTrue(torch.equal(two.result.logits, torch.tensor([[[3., 4.], [4., 5.]]])))
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_broker_fails_pending_work_and_retains_lease_after_unverified_drain(self):
        entered, finish = threading.Event(), threading.Event()
        wrapper, worker, budget, admission = make_wrapper()

        def failed_drain(snapshot, cancel_event):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('test timed out')
            raise WorkerSynchronizationError('unverified drain')

        with patch.object(worker, 'execute', side_effect=failed_drain):
            broker = Broker(wrapper, max_pending=4, max_pending_per_team=2)
            try:
                first = broker.submit('team-a', request('first'))
                self.assertTrue(entered.wait(5))
                second = broker.submit('team-b', request('second'))
                finish.set()
                one, two = first.wait(5), second.wait(5)
            finally:
                finish.set()
                broker.close()
        self.assertEqual(one.state, RequestState.FAILED)
        self.assertEqual(one.error_type, 'WorkerSynchronizationError')
        self.assertEqual(two.state, RequestState.FAILED)
        self.assertTrue(admission.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)

    def test_real_resident_worker_failed_drain_propagates_through_broker(self):
        # Fail CUDA setup and synchronization before any actual GPU allocation.
        with patch.object(resident_worker_module, '_process_quarantined', threading.Event()):
            with patch.object(resident_worker_module, '_quarantine_references', []):
                worker = make_worker()
                worker._device = torch.device('cuda:0')
                budget = make_budget()
                admission = DeviceAdmission(budget, device='cuda:0', max_estimated_incremental_bytes=4096)
                wrapper = BudgetedWorker(worker, admission, (profile(device='cuda:0'),))
                with patch('torch.random.fork_rng', side_effect=RuntimeError('setup failed')):
                    with patch('torch.cuda.synchronize', side_effect=RuntimeError('drain failed')):
                        with Broker(wrapper, max_pending=2, max_pending_per_team=1) as broker:
                            outcome = broker.submit('team', request()).wait(5)
                self.assertEqual(outcome.state, RequestState.FAILED)
                self.assertEqual(outcome.error_type, 'WorkerSynchronizationError')
                self.assertTrue(worker.quarantined)
                self.assertTrue(admission.quarantined)
                self.assertEqual(budget.snapshot().active_leases, 1)


if __name__ == '__main__':
    unittest.main()
