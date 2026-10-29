"""CPU coordinator tests using the actual grouped/resident worker chain."""

import threading
import unittest
import gc
import inspect
import sys
import weakref
from time import perf_counter_ns
from unittest.mock import patch, PropertyMock

import torch
from torch import nn

from looking_glass import batch_validation
from looking_glass.certified_batch_policy import CertifiedBatchPolicy, runtime_fingerprint
from looking_glass.host_budget import HostBudget, HostBusy
from looking_glass.microbatch import BatchProfile, ForwardContract
from looking_glass.resident_worker import ForwardRequest, WorkerOutOfMemory, ResidentWorker
from looking_glass.grouped_broker_state import CohortState
from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
from looking_glass.grouped_worker import GroupedWorker
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget


class ObservedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.call_times = []
        self.entered, self.proceed = None, None
        self.fail = False

    def forward(self, input_ids):
        self.call_times.append(perf_counter_ns())
        self.calls.append(tuple(input_ids.shape))
        if self.entered is not None:
            self.entered.set()
            if not self.proceed.wait(3):
                raise RuntimeError('test timed out')
        if self.fail:
            raise torch.OutOfMemoryError('simulated worker OOM')
        return {'logits': input_ids.float().unsqueeze(-1)}


def fixture(host=None, *, max_pending=4, observation=None, workload_id='whole-2'):
    from looking_glass.grouped_broker import GroupedBroker, SnapshotReservation, ValidationReservation
    runtime = {'schema': 'test-cpu-v1', 'execution': 'eager'}
    contract = ForwardContract('a' * 64, 'checkpoint', 'adapter', 'cpu', torch.float32,
                               runtime_id='sha256:' + runtime_fingerprint(runtime), seed=17,
                               deterministic=True, batch_separable=True, mode='eval',
                               rng_policy='none', cache_policy='disabled', mutable_state_policy='none',
                               tracing=False, intervention=False)
    profile = BatchProfile('profile', 'cert', contract, (('input_ids', torch.int64),), (1, 3),
                           torch.float32, 1, 0., 0., (2,), 2, 1, 2, 6, 48, 24, 72)
    identity = {'model_fingerprint': 'a' * 64, 'checkpoint_sha256': 'b' * 64,
                'adapter_sha256': 'c' * 64, 'backend': 'cpu-eager'}
    binding = batch_validation.binding_from_profile(profile, **identity)
    groups = [{'group_id': f'g-{g}', 'cases': [
        {'case_id': f'{g}-{i}', 'owner': f'team-{i}',
         'inputs': {'input_ids': torch.tensor([[10 * g + i, 10 * g + i + 1, 10 * g + i + 2]])}}
        for i in (1, 2)]} for g in (1, 2)]
    discovery = {'input_ids': torch.tensor([[90, 91, 92]])}
    protocol = batch_validation.freeze_protocol(binding, groups,
        discovery_row_hashes=batch_validation.input_row_digests(binding, discovery))
    def forward(inputs, *, request_id, seed):
        return inputs['input_ids'].float().unsqueeze(-1)
    report = batch_validation.run_validation(protocol, binding, groups, forward,
                                            expected_protocol_sha256=protocol['protocol_sha256'])
    policy = CertifiedBatchPolicy(profile, protocol, report,
        expected_protocol_sha256=protocol['protocol_sha256'], evidence_sha256=report['evidence_sha256'],
        minimum_groups=2, max_atol=0., max_rtol=0., runtime_manifest=runtime, **identity)
    if host is None:
        host = HostBudget(max_bytes=4096, max_items=64, max_requests=8, max_operations=8)
    worker = ObservedModel().eval()
    resident = ResidentWorker('a' * 64, worker, device='cpu', max_batch_size=2,
                              max_sequence_length=3, max_input_bytes=96, max_output_bytes=256)
    device_budget = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                                 max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    gate = DeviceAdmission(device_budget, device='cpu', max_estimated_incremental_bytes=4096)
    workload = WorkloadProfile(workload_id, 'a' * 64, 'cpu', (2, 3),
                               (('input_ids', 'torch.int64'),), 256, 1024)
    budgeted = BudgetedWorker(resident, gate, (workload,))
    grouped = GroupedWorker(budgeted, (profile,), host_budget=host,
                            finite_scratch_bytes={'profile': 128},
                            finite_scratch_items={'profile': 4})
    reservation = ValidationReservation(policy.compatibility_key, 2, 'd' * 64, 256, 4)
    snapshot_reservation = SnapshotReservation(policy.compatibility_key, 'e' * 64, 128, 4)
    options = {} if observation is None else {'observation': observation}
    broker = GroupedBroker(grouped, profile, policy, host_budget=host,
        identity=identity, collect_runtime=lambda: dict(runtime), validation_reservations=(reservation,),
        snapshot_reservation=snapshot_reservation,
        max_pending=max_pending, max_pending_per_team=2, max_coalesce_wait=0.05, max_lifetime_jobs=20,
        **options)
    return broker, worker, host


def request(request_id, row):
    return ForwardRequest(request_id, {'input_ids': torch.tensor([row])}, seed=17)


def receive(ticket, timeout):
    result = ticket.wait(timeout)
    ticket.confirm(result)
    return result


class GroupedBrokerTests(unittest.TestCase):
    def test_submit_return_frame_does_not_retain_retired_input_snapshot(self):
        from looking_glass.grouped_broker import GroupedBroker
        broker, _, host = fixture()
        entered, resume = threading.Event(), threading.Event()
        failures = []
        lines, first_line = inspect.getsourcelines(GroupedBroker.submit)
        return_line = first_line + next(i for i, line in enumerate(lines) if line.strip() == 'return handle')

        def submit():
            def trace(frame, event, arg):
                if frame.f_code is GroupedBroker.submit.__code__ and event == 'line' and frame.f_lineno == return_line:
                    entered.set()
                    if not resume.wait(5):
                        raise RuntimeError('test timed out')
                return trace
            sys.settrace(trace)
            try:
                broker.submit('team-a', request('a', [1, 2, 3]))
            except BaseException as exc:
                failures.append(exc)
            finally:
                sys.settrace(None)

        thread = threading.Thread(target=submit)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            first = broker.lookup('team-a', 'a')
            tensor_ref = weakref.ref(first._job.snapshot.inputs['input_ids'])
            second = broker.submit('team-b', request('b', [4, 5, 6]))
            broker.run_once(1)
            for handle in (first, second):
                ticket = handle.delivery()
                result = receive(ticket, 1)
                del result
                ticket.release()
                handle.dispose()
            gc.collect()
            self.assertEqual(host.snapshot().used_bytes, 0)
            self.assertIsNone(tensor_ref())
        finally:
            resume.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(broker.close())

    def test_constructor_rejects_every_insufficient_headroom_without_reservations(self):
        exact = dict(max_bytes=2784, max_items=52, max_requests=6, max_operations=2)
        for key in exact:
            with self.subTest(short=key):
                caps = dict(exact)
                caps[key] -= 1
                host = HostBudget(**caps)
                with self.assertRaisesRegex(ValueError, 'host ledger cannot cover'):
                    fixture(host)
                snapshot = host.snapshot()
                self.assertEqual((snapshot.used_bytes, snapshot.used_items), (0, 0))
                self.assertEqual((snapshot.requests, snapshot.operations), ((), ()))

    def test_retained_cohorts_fit_exact_headroom_and_ledger_is_exclusive(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        host = HostBudget(max_bytes=2784, max_items=52, max_requests=6, max_operations=2)
        broker, worker, _ = fixture(host)
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            fixture(host)
        handles = [broker.submit(team, request(f'r{i}', [i, i + 1, i + 2]))
                   for i, team in enumerate(('team-a', 'team-b', 'team-a', 'team-b'))]
        self.assertTrue(broker.run_once(1))
        self.assertTrue(broker.run_once(1))
        self.assertEqual(worker.calls, [(2, 3), (2, 3)])
        with self.assertRaises(GroupedBrokerUnavailable):
            broker.submit('team-c', request('overflow', [7, 8, 9]))
        self.assertGreater(host.snapshot().used_bytes, 0)
        with self.assertRaises(HostBusy):
            broker.close()
        for handle in handles:
            ticket = handle.delivery()
            output = receive(ticket, 1)
            self.assertEqual(output.status, 'ok')
            del output
            ticket.release()
            handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        self.assertFalse(broker.close())
        replacement, _, _ = fixture(host)
        self.assertTrue(replacement.close())

    def test_active_delivery_limit_and_exact_ticket_ownership(self):
        from looking_glass.grouped_broker import GroupedDelivery
        broker, _, _ = fixture()
        handle = broker.submit('team-a', request('req', [1, 2, 3]))
        tickets = [handle.delivery() for _ in range(4)]
        with self.assertRaises(HostBusy):
            handle.delivery()
        forged = GroupedDelivery(broker, handle._job)
        with self.assertRaises(ValueError):
            forged.release()
        for ticket in tickets:
            with self.assertRaises(TimeoutError):
                receive(ticket, 0)
        self.assertEqual(set(handle.active_deliveries()), set(tickets))
        for ticket in tickets:
            self.assertTrue(ticket.release())
            self.assertFalse(ticket.release())
        self.assertTrue(handle.cancel())
        handle.dispose()
        self.assertTrue(broker.close())

    def test_cancel_after_retirement_is_nonfatal(self):
        broker, model, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            del result
            ticket.release()
            handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertFalse(first.cancel())
        self.assertFalse(second.cancel())
        self.assertFalse(broker.quarantined)
        third = broker.submit('team-a', request('c', [7, 8, 9]))
        fourth = broker.submit('team-b', request('d', [10, 11, 12]))
        broker.run_once(1)
        self.assertEqual(model.calls, [(2, 3), (2, 3)])
        for handle in (third, fourth):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            del result
            ticket.release()
            handle.dispose()
        self.assertTrue(broker.close())

    def test_delivery_construction_failure_quarantines_with_claims_retained(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        before = host.snapshot().used_bytes
        with patch('looking_glass.grouped_broker.GroupedResult', side_effect=MemoryError('private logits')):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                receive(ticket, 1)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        ticket.release()
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_interrupted_delivery_return_requires_explicit_abandon_attestation(self):
        from looking_glass.grouped_broker import GroupedBroker
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        tensor_ref = weakref.ref(first._job.group.state.publication().members[0].logits)
        ticket = first.delivery()
        errors = []

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._wait.__code__ and event == 'return':
                raise KeyboardInterrupt('interrupted delivery return')
            return trace

        sys.settrace(trace)
        try:
            try:
                ticket.wait(1)
            except KeyboardInterrupt as exc:
                errors.append(exc)
        finally:
            sys.settrace(None)
        self.assertEqual(len(errors), 1)
        with self.assertRaises(HostBusy):
            ticket.release()
        other = second.delivery()
        result = receive(other, 1)
        del result
        other.release()
        second.dispose()
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertIsNotNone(tensor_ref())
        errors.clear()
        gc.collect()
        self.assertTrue(ticket.abandon())
        self.assertFalse(ticket.abandon())
        ticket.release()
        first.dispose()
        gc.collect()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertIsNone(tensor_ref())
        self.assertTrue(broker.close())

    def test_timeout_during_running_group_is_retryable_without_quarantine(self):
        broker, model, host = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        failures = []
        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=dispatch)
        thread.start()
        ticket = first.delivery()
        try:
            self.assertTrue(model.entered.wait(5))
            with self.assertRaises(TimeoutError):
                ticket.wait(0)
            self.assertFalse(broker.quarantined)
            self.assertGreater(host.snapshot().used_bytes, 0)
        finally:
            model.proceed.set()
            thread.join(5)
        self.assertEqual(failures, [])
        with patch.object(CohortState, 'wait', side_effect=AssertionError('nested handoff must not run')):
            result = receive(ticket, 1)
        del result
        ticket.release()
        first.dispose()
        other = second.delivery()
        result = receive(other, 1)
        del result
        other.release()
        second.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())

    def test_timeout_injected_after_publication_is_quarantined(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        ticket = first.delivery()
        lines, first_line = inspect.getsourcelines(GroupedBroker._wait)
        target = first_line + next(i for i, line in enumerate(lines)
                                   if line.strip() == 'state._check_state(publication)')

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._wait.__code__ and event == 'line' and frame.f_lineno == target:
                raise TimeoutError('injected after publication')
            return trace
        sys.settrace(trace)
        try:
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                ticket.wait(1)
        finally:
            sys.settrace(None)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        ticket.release()
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_blocked_cohort_validation_does_not_block_unrelated_submit(self):
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        entered, resume, submitted = (threading.Event() for _ in range(3))
        failures, cancellations, disposals, later = [], [], [], []
        original = CohortState._validate_outcome

        def validation(state, outcome):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test timed out')
            return original(state, outcome)

        def dispatch():
            try:
                broker.run_once(1)
            except BaseException as exc:
                failures.append(exc)

        def cancel():
            try:
                cancellations.append(first.cancel())
            except BaseException as exc:
                failures.append(exc)

        def dispose():
            try:
                disposals.append(second.dispose())
            except BaseException as exc:
                failures.append(exc)

        def submit():
            try:
                later.append(broker.submit('team-c', request('c', [7, 8, 9])))
            except BaseException as exc:
                failures.append(exc)
            finally:
                submitted.set()

        with patch.object(CohortState, '_validate_outcome', validation):
            dispatch_thread = threading.Thread(target=dispatch)
            dispatch_thread.start()
            self.assertTrue(entered.wait(5))
            cancel_thread = threading.Thread(target=cancel)
            dispose_thread = threading.Thread(target=dispose)
            submit_thread = threading.Thread(target=submit)
            cancel_thread.start()
            dispose_thread.start()
            submit_thread.start()
            try:
                self.assertTrue(submitted.wait(2))
                self.assertFalse(broker.quarantined)
            finally:
                resume.set()
                for thread in (dispatch_thread, cancel_thread, dispose_thread, submit_thread):
                    thread.join(5)
        self.assertEqual(failures, [])
        self.assertEqual(cancellations, [False])
        self.assertEqual(disposals, [True])
        ticket = first.delivery()
        result = receive(ticket, 1)
        del result
        ticket.release()
        first.dispose()
        self.assertTrue(later[0].cancel())
        later[0].dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())

    def test_cancel_failure_blocks_retirement_before_quarantine_is_latched(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            del result
            ticket.release()
        state = first._job.group.state
        entered, resume = threading.Event(), threading.Event()
        failures = []
        original_fail = broker._fail_closed

        def bad_cancel(job_id):
            publication = state.publication()
            self.assertEqual(len(publication.members), 2)
            raise MemoryError('failure after obtaining publication')

        def delayed_fail(exc, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test timed out')
            original_fail(exc, **kwargs)

        def cancel():
            try:
                first.cancel()
            except BaseException as exc:
                failures.append(exc)

        with patch.object(state, 'cancel', bad_cancel), patch.object(broker, '_fail_closed', delayed_fail):
            thread = threading.Thread(target=cancel)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                second.dispose()
                with self.assertRaises(HostBusy):
                    first.dispose()
                self.assertGreater(host.snapshot().used_bytes, 0)
            finally:
                resume.set()
                thread.join(5)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], GroupedBrokerUnavailable)
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_disposal_retry_finishes_interrupted_retirement_handoff(self):
        from looking_glass.grouped_broker import GroupedBroker
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            del result
            ticket.release()
        first.dispose()
        lines, start = inspect.getsourcelines(GroupedBroker._dispose)
        target = start + [i for i, line in enumerate(lines)
                          if line.strip() == 'self._maybe_retire(group)'][-1]

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._dispose.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('interrupted before retirement')
            return trace
        sys.settrace(trace)
        try:
            with self.assertRaises(KeyboardInterrupt):
                second.dispose()
        finally:
            sys.settrace(None)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertFalse(second.dispose())
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertFalse(broker.quarantined)
        self.assertTrue(broker.close())

    def test_disposal_marker_interruption_quarantines_before_retirement(self):
        from looking_glass.grouped_broker import GroupedBroker, GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        for handle in (first, second):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            del result
            ticket.release()
        first.dispose()
        lines, start = inspect.getsourcelines(GroupedBroker._dispose)
        target = start + next(i for i, line in enumerate(lines)
                              if line.strip() == 'if expected_error is None and group is not None and not already_disposed:')

        def trace(frame, event, arg):
            if frame.f_code is GroupedBroker._dispose.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('interrupted after disposal pin')
            return trace
        sys.settrace(trace)
        try:
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                second.dispose()
        finally:
            sys.settrace(None)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()

    def test_batch_matches_separate_serial_worker_for_distinct_held_out_inputs(self):
        broker, model, host = fixture()
        reference = ResidentWorker('a' * 64, ObservedModel().eval(), device='cpu', max_batch_size=2,
                                   max_sequence_length=3, max_input_bytes=96, max_output_bytes=256)
        rows = ([101, 22, 78], [5, 88, 1], [9, 7, 35], [99, 3, 11])
        for index in range(0, len(rows), 2):
            inputs = [request(f'ref-{i}', rows[i]) for i in (index, index + 1)]
            expected = [reference.execute(item).logits for item in inputs]
            handles = [broker.submit(team, request(f'case-{i}', rows[i]))
                       for team, i in zip(('owner-x', 'owner-y'), (index, index + 1))]
            broker.run_once(1)
            tickets = [handle.delivery() for handle in handles]
            outputs = [receive(ticket, 1) for ticket in tickets]
            self.assertTrue(all(torch.equal(output.logits, serial) for output, serial in zip(outputs, expected)))
            del outputs, expected
            for ticket in tickets:
                ticket.release()
            for handle in handles:
                handle.dispose()
        self.assertEqual(model.calls, [(2, 3), (2, 3)])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        reference.close()

    def test_queue_cancel_failure_after_removal_latches_quarantine(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host = fixture()
        handle = broker.submit('team-a', request('req', [1, 2, 3]))
        before = host.snapshot().used_bytes
        original = broker._queue.cancel

        def remove_then_fail(job_id):
            self.assertTrue(original(job_id))
            raise MemoryError('private queue failure')

        with patch.object(broker._queue, 'cancel', side_effect=remove_then_fail):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                handle.cancel()
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_submission_error_does_not_expose_private_exception_context(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, worker, host = fixture()
        private = RuntimeError('private input sentinel')
        with patch.object(broker._worker, 'snapshot_request', side_effect=private):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                broker.submit('team-a', request('req', [1, 2, 3]))
        self.assertIsNone(caught.exception.__context__)
        self.assertIs(broker._failure, private)
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)

    def test_prelaunch_cancel_retirement_after_last_dispose(self):
        broker, worker, host = fixture()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        entered, resume_commit, published, resume_finish = (threading.Event() for _ in range(4))
        errors = []
        original_commit = CohortState.try_commit_launch
        original_finish = CohortState.finish_cancelled_before_launch

        def blocked_commit(state):
            entered.set()
            if not resume_commit.wait(5):
                raise RuntimeError('commit test timed out')
            return original_commit(state)

        def blocked_finish(state):
            result = original_finish(state)
            published.set()
            if not resume_finish.wait(5):
                raise RuntimeError('finish test timed out')
            return result

        def dispatch():
            try:
                broker.run_once(timeout=1)
            except BaseException as exc:
                errors.append(exc)

        with patch.object(CohortState, 'try_commit_launch', blocked_commit):
            with patch.object(CohortState, 'finish_cancelled_before_launch', blocked_finish):
                thread = threading.Thread(target=dispatch)
                thread.start()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertTrue(first.cancel())
                    resume_commit.set()
                    self.assertTrue(published.wait(5))
                    first_ticket, second_ticket = first.delivery(), second.delivery()
                    left, right = receive(first_ticket, 1), receive(second_ticket, 1)
                    self.assertEqual((left.status, right.status), ('cancelled', 'failed'))
                    del left, right
                    first_ticket.release()
                    second_ticket.release()
                    first.dispose()
                    second.dispose()
                    self.assertGreater(host.snapshot().used_bytes, 0)
                finally:
                    resume_commit.set()
                    resume_finish.set()
                    thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(worker.calls, [])
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_waiting_delivery_blocks_release_and_disposal(self):
        from looking_glass.grouped_broker import GroupedResult
        broker, _, host = fixture()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        broker.run_once(timeout=1)
        ticket = first.delivery()
        other = second.delivery()
        other_result = receive(other, 1)
        del other_result
        other.release()
        second.dispose()
        entered, resume, delivered, release = (threading.Event() for _ in range(4))
        outcomes, errors = [], []

        def blocked_result(*args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test timed out')
            return GroupedResult(*args, **kwargs)

        def wait():
            try:
                outcomes.append(receive(ticket, 1))
                delivered.set()
                if not release.wait(5):
                    raise RuntimeError('test timed out')
                ticket.release()
            except BaseException as exc:
                errors.append(exc)

        with patch('looking_glass.grouped_broker.GroupedResult', blocked_result):
            thread = threading.Thread(target=wait)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(HostBusy):
                    ticket.release()
                with self.assertRaises(HostBusy):
                    first.dispose()
                self.assertGreater(host.snapshot().used_bytes, 0)
            finally:
                resume.set()
                self.assertTrue(delivered.wait(5))
                self.assertEqual(outcomes[0].status, 'ok')
                with self.assertRaises(HostBusy):
                    ticket.release()
                outcomes.clear()
                release.set()
                thread.join(5)
        self.assertEqual(errors, [])
        first.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_each_delivery_requires_separate_release_before_disposal(self):
        broker, _, host = fixture()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        broker.run_once(timeout=1)
        a, b, other = first.delivery(), first.delivery(), second.delivery()
        x, y, z = receive(a, 1), receive(b, 1), receive(other, 1)
        del x, z
        a.release()
        other.release()
        second.dispose()
        with self.assertRaises(HostBusy):
            first.dispose()
        self.assertGreater(host.snapshot().used_bytes, 0)
        del y
        b.release()
        first.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        with self.assertRaises(Exception):
            receive(b, 1)

    def test_quarantine_health_does_not_wait_for_blocked_cohort_lock(self):
        broker, _, _ = fixture()
        broker.submit('team-a', request('req-a', [1, 2, 3]))
        broker.submit('team-b', request('req-b', [4, 5, 6]))
        group = broker._claim(1)
        entered, resume, inspected = (threading.Event() for _ in range(3))
        seen = []

        def blocked_unknown():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test timed out')

        with patch.object(group.state, 'mark_unknown', blocked_unknown):
            reporter = threading.Thread(target=lambda: broker._fail_closed(RuntimeError('test')))
            reporter.start()
            try:
                self.assertTrue(entered.wait(5))
                reader = threading.Thread(target=lambda: (seen.append(broker.quarantined), inspected.set()))
                reader.start()
                self.assertTrue(inspected.wait(2))
                reader.join(5)
            finally:
                resume.set()
                reporter.join(5)
        self.assertEqual(seen, [True])

    def test_external_device_quarantine_blocks_admission_and_disposal(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        before = host.snapshot().used_bytes
        broker._worker._worker._admission._latch()
        self.assertTrue(broker.quarantined)
        with self.assertRaises(GroupedBrokerUnavailable):
            broker.submit('team-c', request('c', [7, 8, 9]))
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            broker.close()
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_healthy_active_device_gate_does_not_block_or_quarantine_health(self):
        broker, _, _ = fixture()
        gate = broker._worker._worker._admission._lock
        gate.acquire()
        finished = threading.Event()
        observed = []
        reader = threading.Thread(target=lambda: (observed.append(broker.quarantined), finished.set()))
        reader.start()
        try:
            self.assertTrue(finished.wait(2))
            self.assertEqual(observed, [False])
        finally:
            gate.release()
            reader.join(5)
        self.assertTrue(broker.close())

    def test_failed_device_health_read_stays_quarantined(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, _, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        broker.run_once(1)
        before = host.snapshot().used_bytes
        with patch.object(BudgetedWorker, 'quarantined', new_callable=PropertyMock,
                          side_effect=[MemoryError('health read'), False]) as health:
            self.assertTrue(broker.quarantined)
            self.assertTrue(broker.quarantined)
            self.assertEqual(health.call_count, 1)
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()
        self.assertEqual(host.snapshot().used_bytes, before)

    def test_failed_worker_status_read_quarantines_submit(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, worker, host = fixture()
        with patch.object(BudgetedWorker, 'failed', new_callable=PropertyMock,
                          side_effect=[MemoryError('private health failure'), False]) as health:
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                broker.submit('team-a', request('a', [1, 2, 3]))
            self.assertIsNone(caught.exception.__context__)
            self.assertTrue(broker.quarantined)
            self.assertTrue(broker.failed)
            self.assertEqual(health.call_count, 1)
        self.assertEqual(worker.calls, [])
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_failed_worker_status_read_during_dispatch_keeps_staged_claims(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        broker, worker, host = fixture()
        first = broker.submit('team-a', request('a', [1, 2, 3]))
        second = broker.submit('team-b', request('b', [4, 5, 6]))
        before = host.snapshot().used_bytes
        with patch.object(BudgetedWorker, 'failed', new_callable=PropertyMock,
                          side_effect=MemoryError('private health failure')):
            with self.assertRaises(GroupedBrokerUnavailable) as caught:
                broker.run_once(1)
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(broker.quarantined)
        self.assertEqual(worker.calls, [])
        self.assertEqual(host.snapshot().used_bytes, before)
        with self.assertRaises(GroupedBrokerUnavailable):
            first.dispose()
        with self.assertRaises(GroupedBrokerUnavailable):
            second.dispose()

    def test_two_teams_share_one_model_call_and_keep_host_custody_until_disposal(self):
        broker, worker, host = fixture()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        self.assertTrue(broker.run_once(timeout=1))
        first_delivery, second_delivery = first.delivery(), second.delivery()
        left, right = receive(first_delivery, 1), receive(second_delivery, 1)
        self.assertEqual(worker.calls, [(2, 3)])
        self.assertEqual((left.team_id, left.request_id, left.status), ('team-a', 'req-a', 'ok'))
        self.assertEqual((right.team_id, right.request_id, right.status), ('team-b', 'req-b', 'ok'))
        self.assertTrue(torch.equal(left.logits, torch.tensor([[[1.], [2.], [3.]]])))
        self.assertTrue(torch.equal(right.logits, torch.tensor([[[4.], [5.], [6.]]])))
        self.assertGreater(host.snapshot().used_bytes, 0)
        del left, right
        first_delivery.release()
        second_delivery.release()
        first.dispose()
        self.assertGreater(host.snapshot().used_bytes, 0)
        second.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_cancel_during_forward_preserves_peer_and_suppresses_cancelled_output(self):
        broker, worker, _ = fixture()
        worker.entered, worker.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        errors = []
        def dispatch():
            try:
                broker.run_once(timeout=1)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=dispatch, daemon=True)
        thread.start()
        try:
            self.assertTrue(worker.entered.wait(3))
            self.assertTrue(first.cancel())
        finally:
            worker.proceed.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        first_delivery, second_delivery = first.delivery(), second.delivery()
        left, right = receive(first_delivery, 1), receive(second_delivery, 1)
        self.assertEqual((left.status, right.status), ('cancelled', 'ok'))
        self.assertTrue(left.cancel_requested)
        self.assertIsNone(left.logits)
        self.assertTrue(torch.equal(right.logits, torch.tensor([[[4.], [5.], [6.]]])))
        self.assertEqual(worker.calls, [(2, 3)])
        del left, right
        first_delivery.release()
        second_delivery.release()
        first.dispose()
        second.dispose()

    def test_whole_batch_oom_dominates_cancel_and_quarantines(self):
        broker, worker, host = fixture()
        worker.fail = True
        worker.entered, worker.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-b', request('req-b', [4, 5, 6]))
        thread = threading.Thread(target=lambda: broker.run_once(timeout=1), daemon=True)
        thread.start()
        try:
            self.assertTrue(worker.entered.wait(3))
            self.assertTrue(first.cancel())
            self.assertTrue(second.cancel())
        finally:
            worker.proceed.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        first_delivery, second_delivery = first.delivery(), second.delivery()
        left, right = receive(first_delivery, 1), receive(second_delivery, 1)
        self.assertEqual((left.status, right.status), ('failed', 'failed'))
        self.assertTrue(left.cancel_requested and right.cancel_requested)
        self.assertEqual((left.error_code, right.error_code), ('out_of_memory', 'out_of_memory'))
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        del left, right
        first_delivery.release()
        second_delivery.release()

    def test_pending_cancel_and_unserviceable_head_advance_without_reordering(self):
        broker, worker, host = fixture()
        cancel = broker.submit('team-a', request('cancel', [1, 2, 3]))
        self.assertTrue(cancel.cancel())
        cancel_delivery = cancel.delivery()
        self.assertEqual(receive(cancel_delivery, 1).status, 'cancelled')
        cancel_delivery.release()
        cancel.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        first = broker.submit('team-a', request('req-a', [1, 2, 3]))
        second = broker.submit('team-a', request('req-b', [4, 5, 6]))
        self.assertTrue(broker.run_once(timeout=1))
        first_delivery = first.delivery()
        self.assertEqual(receive(first_delivery, 1).error_code, 'no_compatible_cohort')
        first_delivery.release()
        first.dispose()
        third = broker.submit('team-b', request('req-c', [7, 8, 9]))
        self.assertTrue(broker.run_once(timeout=1))
        second_delivery, third_delivery = second.delivery(), third.delivery()
        result2, result3 = receive(second_delivery, 1), receive(third_delivery, 1)
        self.assertEqual((result2.status, result3.status), ('ok', 'ok'))
        self.assertEqual(worker.calls, [(2, 3)])
        del result2, result3
        second_delivery.release()
        third_delivery.release()
        second.dispose()
        third.dispose()

    def test_coalescing_deadline_uses_elapsed_high_resolution_clock(self):
        broker, worker, host = fixture()
        with patch('looking_glass.grouped_broker.perf_counter', return_value=100.0):
            handle = broker.submit('team-a', request('solo', [1, 2, 3]))
        with patch('looking_glass.grouped_broker.perf_counter', return_value=100.051):
            self.assertTrue(broker.run_once(timeout=0))
        ticket = handle.delivery()
        result = receive(ticket, 1)
        self.assertEqual(result.error_code, 'no_compatible_cohort')
        self.assertEqual(worker.calls, [])
        del result
        ticket.release()
        handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())


if __name__ == '__main__':
    unittest.main()
