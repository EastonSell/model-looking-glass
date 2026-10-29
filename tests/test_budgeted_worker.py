"""CPU-only integration tests of trusted admission, execution and cleanup."""
import threading
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
    WorkerExecutionError, WorkerOutOfMemory, WorkerSynchronizationError,
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
        worker.model_id = 'b' * 64
        with self.assertRaises(Exception):
            _ = wrapper.max_input_bytes
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
            setattr(worker, attribute, replacement)
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
            worker.max_output_bytes = 128
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
                worker.device = torch.device('cuda:0')
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
