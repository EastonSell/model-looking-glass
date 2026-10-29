"""CPU integration tests for grouped admission, handoff, and host custody."""
import threading
import inspect
import sys
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
from looking_glass.grouped_worker import GroupedWorker, GroupPlan
from looking_glass.host_budget import HostBudget, HostBusy, HostOperation, HostRequestCancelled
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.microbatch import BatchProfile, ForwardContract, Microbatch, MicrobatchRequest
from looking_glass.resident_worker import ResidentWorker, WorkerOutOfMemory


MODEL_ID = 'a' * 64


class RowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        x = input_ids.float() * attention_mask
        return {'logits': torch.stack((x, x + 1), dim=-1)}


def setup_group():
    model = RowModel().eval()
    resident = ResidentWorker(MODEL_ID, model, device='cpu', max_batch_size=2,
                              max_sequence_length=3, max_input_bytes=96, max_output_bytes=256)
    device_budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                                 max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    gate = DeviceAdmission(device_budget, device='cpu', max_estimated_incremental_bytes=4096)
    gpu_profile = WorkloadProfile('whole-2', MODEL_ID, 'cpu', (2, 3),
                                  (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')),
                                  output_limit_bytes=256, estimated_incremental_bytes=1024)
    worker = BudgetedWorker(resident, gate, (gpu_profile,))
    contract = ForwardContract(MODEL_ID, 'checkpoint-sha', 'tiny-adapter', 'cpu', torch.float32,
                               runtime_id='cpu-test-runtime', seed=17, deterministic=True,
                               batch_separable=True, mode='eval', rng_policy='none',
                               cache_policy='disabled', mutable_state_policy='none',
                               tracing=False, intervention=False)
    profile = BatchProfile('batch-v1', 'cpu-cert-v1', contract,
                           input_schema=(('attention_mask', torch.int64), ('input_ids', torch.int64)),
                           input_shape=(1, 3), output_dtype=torch.float32, output_width=2,
                           serial_atol=0, serial_rtol=0, approved_rows=(1, 2),
                           max_requests=2, max_per_owner=1, max_batch_rows=2,
                           max_token_positions=6, max_logical_input_bytes=96,
                           max_logical_output_bytes=48, max_logical_total_bytes=144)
    jobs = [MicrobatchRequest(f'job-{i}', f'team-{i}', f'req-{i}', contract,
                              {'input_ids': torch.tensor([[i, i + 1, i + 2]]),
                               'attention_mask': torch.ones((1, 3), dtype=torch.int64)})
            for i in (1, 4)]
    host = HostBudget(max_bytes=8192, max_items=128, max_requests=8, max_operations=8)
    request = host.open_request('group-request')
    grouped = GroupedWorker(worker, (profile,), host_budget=host,
                            finite_scratch_bytes={'batch-v1': 256},
                            finite_scratch_items={'batch-v1': 4})
    needed_bytes, needed_items = grouped.required_host_capacity('batch-v1')
    claim = request.reserve('broker', 'group-envelope', needed_bytes, items=needed_items)
    return grouped, model, gate, device_budget, host, request, claim, profile, jobs


class GroupedWorkerTests(unittest.TestCase):
    def test_one_whole_batch_lease_and_host_pin_through_split(self):
        grouped, model, gate, device_budget, host, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        original_finish = Microbatch.finish

        def verify_handoff(plan, logits, *, launch_id):
            self.assertEqual(device_budget.snapshot().active_leases, 1)
            with self.assertRaises(HostBusy):
                claim.release()
            return original_finish(plan, logits, launch_id=launch_id)

        with patch.object(Microbatch, 'finish', verify_handoff):
            outcome = grouped.execute(group, launch)
        self.assertEqual(model.calls, 1)
        self.assertEqual([r.status for r in outcome.results], ['ok', 'ok'])
        self.assertTrue(torch.equal(outcome.results[0].logits[..., 0], jobs[0].inputs['input_ids'].float()))
        self.assertTrue(torch.equal(outcome.results[1].logits[..., 0], jobs[1].inputs['input_ids'].float()))
        self.assertEqual(device_budget.snapshot().active_leases, 0)
        self.assertFalse(gate.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        del outcome, launch, jobs
        group.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        request.close()

    def test_insufficient_host_claim_fails_before_plan_creation(self):
        grouped, model, _, _, host, request, claim, _, jobs = setup_group()
        needed_bytes, needed_items = grouped.required_host_capacity('batch-v1')
        claim.release()
        small = request.reserve('broker', 'group-envelope', needed_bytes - 1, items=needed_items)
        with patch.object(Microbatch, '__init__', side_effect=AssertionError('must not allocate')):
            with self.assertRaises(ValueError):
                grouped.prepare('batch-v1', jobs, request, small, 'group-small')
        self.assertIsNone(small.info().binding)
        self.assertEqual(model.calls, 0)
        small.release()
        request.close()

    def test_one_item_short_fails_before_plan_creation(self):
        grouped, model, _, _, _, request, claim, _, jobs = setup_group()
        needed_bytes, needed_items = grouped.required_host_capacity('batch-v1')
        claim.release()
        small = request.reserve('broker', 'group-envelope', needed_bytes, items=needed_items - 1)
        with patch.object(Microbatch, '__init__', side_effect=AssertionError('must not allocate')):
            with self.assertRaises(ValueError):
                grouped.prepare('batch-v1', jobs, request, small, 'group-small')
        self.assertIsNone(small.info().binding)
        self.assertEqual(grouped.active_groups(), ())
        self.assertEqual(model.calls, 0)
        small.release()
        request.close()

    def test_view_with_oversized_backing_storage_is_rejected(self):
        grouped, model, _, _, host, request, claim, _, jobs = setup_group()
        values = torch.zeros((100, 3), dtype=torch.int64)
        inputs = dict(jobs[0].inputs)
        inputs['input_ids'] = values[:1]
        jobs[0] = replace(jobs[0], inputs=inputs)
        with self.assertRaisesRegex(ValueError, 'exact contiguous CPU storage'):
            grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        self.assertEqual(model.calls, 0)
        self.assertEqual(grouped.active_groups(), ())
        self.assertGreater(host.snapshot().used_bytes, 0)
        claim.release()
        request.close()

    def test_same_claim_cannot_back_two_batches(self):
        grouped, _, _, _, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        with self.assertRaises((ValueError, HostBusy)):
            grouped.prepare('batch-v1', jobs, request, claim, 'group-b')
        launch = group.launch()
        outcome = grouped.execute(group, launch)
        with self.assertRaises((ValueError, HostBusy)):
            grouped.prepare('batch-v1', jobs, request, claim, 'group-b')
        del outcome, launch, jobs
        group.dispose()
        request.close()

    def test_cancel_one_member_during_forward_preserves_other_result(self):
        entered, proceed = threading.Event(), threading.Event()
        grouped, model, gate, device_budget, _, request, claim, _, jobs = setup_group()
        original = model.forward

        def blocking_forward(input_ids, attention_mask):
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError('test timed out')
            return original(input_ids, attention_mask)

        model.forward = blocking_forward
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        outcomes, errors = [], []

        def run():
            try:
                outcomes.append(grouped.execute(group, launch))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(device_budget.snapshot().active_leases, 1)
            self.assertTrue(group.cancel('job-1'))
        finally:
            proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([r.status for r in outcomes[0].results], ['cancelled', 'ok'])
        self.assertEqual(model.calls, 1)
        self.assertFalse(gate.quarantined)
        outcomes.clear()
        del launch, jobs
        group.dispose()
        request.close()

    def test_cancelled_host_request_never_starts_a_new_forward(self):
        for after_launch in (False, True):
            with self.subTest(after_launch=after_launch):
                grouped, model, gate, device_budget, host, request, claim, _, jobs = setup_group()
                group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
                launch = group.launch() if after_launch else None
                request.cancel()
                if after_launch:
                    with self.assertRaises(HostRequestCancelled):
                        grouped.execute(group, launch)
                else:
                    with self.assertRaises(HostRequestCancelled):
                        group.launch()
                    group.abort_unexecuted(HostRequestCancelled('cancelled'))
                self.assertEqual(model.calls, 0)
                self.assertFalse(gate.quarantined)
                self.assertEqual(device_budget.snapshot().active_leases, 0)
                del launch, jobs
                group.dispose()
                request.close()
                self.assertEqual(host.snapshot().used_bytes, 0)

    def test_clone_does_not_hold_host_ledger_lock(self):
        grouped, _, _, _, host, request, claim, _, jobs = setup_group()
        entered, resume, inspected = (threading.Event() for _ in range(3))
        groups, errors = [], []
        original = Microbatch.__init__

        def blocked_init(plan, *args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test timed out')
            return original(plan, *args, **kwargs)

        def prepare():
            try:
                groups.append(grouped.prepare('batch-v1', jobs, request, claim, 'group-a'))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(Microbatch, '__init__', blocked_init):
            thread = threading.Thread(target=prepare)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                pending = grouped.active_groups()
                self.assertEqual(len(pending), 1)
                with self.assertRaises(HostBusy):
                    pending[0].dispose()
                self.assertGreater(host.snapshot().used_bytes, 0)
                inspect_thread = threading.Thread(target=lambda: (host.snapshot(), inspected.set()))
                inspect_thread.start()
                self.assertTrue(inspected.wait(2), 'tensor copy blocked the host ledger')
                inspect_thread.join(5)
            finally:
                resume.set()
                thread.join(5)
        self.assertEqual(errors, [])
        groups[0].abort_unexecuted(RuntimeError('no launch'))
        groups[0].dispose()
        request.close()

    def test_request_construction_interrupt_does_not_strand_execution(self):
        grouped, model, _, budget, host, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        with patch('looking_glass.grouped_worker.ForwardRequest', side_effect=KeyboardInterrupt('injected')):
            with self.assertRaises(KeyboardInterrupt):
                grouped.execute(group, launch)
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)
        del launch, jobs
        group.dispose()
        request.close()
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_disposal_can_retry_after_operation_finish_failure(self):
        grouped, _, _, _, host, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        outcome = grouped.execute(group, launch)
        del outcome, launch, jobs
        with patch.object(HostOperation, 'finish', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                group.dispose()
        self.assertGreater(host.snapshot().used_bytes, 0)
        group.dispose()
        request.close()
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_disposal_retry_removes_only_its_exact_registry_entry(self):
        grouped, _, _, _, host, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        group.abort_unexecuted(RuntimeError('no launch'))
        source, first = inspect.getsourcelines(GroupPlan.dispose)
        target = first + max(i for i, line in enumerate(source)
                             if line.strip() == 'if self._owner._groups.get(self.group_id) is self:')
        fired = []

        def trace(frame, event, arg):
            if frame.f_code is GroupPlan.dispose.__code__ and event == 'line' and frame.f_lineno == target:
                sys.settrace(None)
                fired.append(True)
                raise KeyboardInterrupt('registry cleanup interrupted')
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                group.dispose()
        finally:
            sys.settrace(None)
        self.assertEqual(fired, [True])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertEqual(grouped.active_groups(), (group,))
        self.assertFalse(group.dispose())
        self.assertEqual(grouped.active_groups(), ())
        request.close()

    def test_mutated_launch_contents_fail_before_forward(self):
        grouped, model, gate, budget, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        # NumPy writes bypass Tensor._version; the independent digest still
        # catches this cooperative-owner mistake before model execution.
        launch.inputs['input_ids'].numpy()[0, 0] = 99
        with self.assertRaisesRegex(ValueError, 'snapshot differs'):
            grouped.execute(group, launch)
        self.assertEqual(model.calls, 0)
        self.assertFalse(gate.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)
        del launch, jobs
        group.dispose()
        request.close()

    def test_input_binding_does_not_require_numpy(self):
        grouped, model, _, _, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        with patch.object(torch.Tensor, 'numpy', side_effect=RuntimeError('NumPy unavailable')):
            launch = group.launch()
            outcome = grouped.execute(group, launch)
        self.assertEqual(model.calls, 1)
        self.assertEqual([result.status for result in outcome.results], ['ok', 'ok'])
        del outcome, launch, jobs
        group.dispose()
        request.close()

    def test_all_cancelled_before_launch_does_not_acquire_a_device_lease(self):
        grouped, model, gate, budget, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        for job in jobs:
            self.assertTrue(group.cancel(job.job_id))
        self.assertIsNone(group.launch())
        outcome = group.outcome()
        self.assertEqual([result.status for result in outcome.results], ['cancelled', 'cancelled'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)
        self.assertFalse(gate.quarantined)
        del outcome, jobs
        group.dispose()
        request.close()

    def test_all_cancelled_while_waiting_never_acquire_or_run(self):
        grouped, model, gate, budget, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        outcomes, errors = [], []

        def run():
            try:
                outcomes.append(grouped.execute(group, launch))
            except BaseException as exc:
                errors.append(exc)

        gate._lock.acquire()
        held = True
        thread = threading.Thread(target=run)
        try:
            with patch.object(budget, 'try_acquire', side_effect=AssertionError('must not acquire')):
                thread.start()
                for job in jobs:
                    self.assertTrue(group.cancel(job.job_id))
                gate._lock.release()
                held = False
                thread.join(5)
        finally:
            if held:
                gate._lock.release()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([r.status for r in outcomes[0].results], ['cancelled', 'cancelled'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(budget.snapshot().active_leases, 0)
        self.assertFalse(gate.quarantined)
        outcomes.clear()
        del launch, jobs
        group.dispose()
        request.close()

    def test_host_request_cancelled_during_forward_suppresses_all_rows(self):
        entered, proceed = threading.Event(), threading.Event()
        grouped, model, gate, budget, host, request, claim, _, jobs = setup_group()
        original = model.forward

        def blocking_forward(input_ids, attention_mask):
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError('test timed out')
            return original(input_ids, attention_mask)

        model.forward = blocking_forward
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        outcomes, errors = [], []

        def run():
            try:
                outcomes.append(grouped.execute(group, launch))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            request.cancel()
            self.assertEqual(budget.snapshot().active_leases, 1)
        finally:
            proceed.set()
            thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual([result.status for result in outcomes[0].results], ['cancelled', 'cancelled'])
        self.assertFalse(gate.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 0)
        self.assertGreater(host.snapshot().used_bytes, 0)
        outcomes.clear()
        del launch, jobs
        group.dispose()
        request.close()

    def test_late_host_cancellation_requires_broker_publication_check(self):
        grouped, model, _, _, _, request, claim, _, jobs = setup_group()
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        observed, resume = threading.Event(), threading.Event()
        outcomes, errors = [], []
        original = group._request_cancelled

        def observed_check():
            cancelled = original()
            if model.calls:
                observed.set()
                if not resume.wait(5):
                    raise RuntimeError('test timed out')
            return cancelled

        def run():
            try:
                outcomes.append(grouped.execute(group, launch))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(group, '_request_cancelled', observed_check):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(observed.wait(5))
                request.cancel()
            finally:
                resume.set()
                thread.join(5)
        self.assertEqual(errors, [])
        # These are private plan results; the broker must still suppress them
        # atomically if cancellation won its publication decision.
        self.assertEqual([result.status for result in outcomes[0].results], ['ok', 'ok'])
        outcomes.clear()
        del launch, jobs
        group.dispose()
        request.close()

    def test_postlaunch_oom_quarantines_the_batch_even_if_one_member_cancelled(self):
        entered, proceed = threading.Event(), threading.Event()
        grouped, model, gate, budget, host, request, claim, _, jobs = setup_group()

        def failing_forward(input_ids, attention_mask):
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError('test timed out')
            raise torch.OutOfMemoryError('injected batch OOM')

        model.forward = failing_forward
        group = grouped.prepare('batch-v1', jobs, request, claim, 'group-a')
        launch = group.launch()
        errors = []

        def run():
            try:
                grouped.execute(group, launch)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(group.cancel('job-1'))
        finally:
            proceed.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], WorkerOutOfMemory)
        self.assertTrue(gate.quarantined)
        self.assertEqual(budget.snapshot().active_leases, 1)
        self.assertGreater(host.snapshot().used_bytes, 0)
        private = group.outcome()
        self.assertTrue(private.quarantine_required)
        self.assertEqual([r.status for r in private.results], ['cancelled', 'failed'])
        with self.assertRaises(HostBusy):
            group.dispose()


if __name__ == '__main__':
    unittest.main()
