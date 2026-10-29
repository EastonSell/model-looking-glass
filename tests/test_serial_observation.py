"""Matched serial observation tests against the real CPU broker/worker chain."""

import threading
import unittest
from unittest.mock import patch
from hashlib import sha256
from time import perf_counter_ns

import torch

from test_grouped_broker import ObservedModel, request
from looking_glass.broker import Broker, BrokerUnavailable, RequestHandle, RequestState
from looking_glass.budgeted_worker import BudgetedWorker, DeviceAdmission, WorkloadProfile
from looking_glass.memory_budget import CudaMemorySnapshot, GIB, MemoryBudget
from looking_glass.resident_worker import InvalidRequest, ResidentWorker


def fixture(*, max_events=128, profile_id='c' * 64):
    from looking_glass.batched_recorder import BatchedEventRecorder
    from looking_glass.serial_observation import SerialObservation, SerialObservedWorker

    recorder = BatchedEventRecorder(max_events=max_events)
    observation = SerialObservation(recorder)
    model = ObservedModel().eval()
    resident = ResidentWorker('a' * 64, model, device='cpu', max_batch_size=1,
                              max_sequence_length=3, max_input_bytes=24, max_output_bytes=64)
    memory = MemoryBudget(lambda: CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0),
                          max_gpu_trace_bytes=0, max_host_trace_bytes=0)
    admission = DeviceAdmission(memory, device='cpu', max_estimated_incremental_bytes=4096)
    profile = WorkloadProfile(profile_id, 'a' * 64, 'cpu', (1, 3),
                              (('input_ids', 'torch.int64'),), 64, 1024)
    budgeted = BudgetedWorker(resident, admission, (profile,))
    proxy = SerialObservedWorker(budgeted, observation)
    broker = Broker(proxy, max_pending=4, max_pending_per_team=2)
    proxy.bind_broker(broker)
    return broker, model, observation, recorder


class SerialObservationTests(unittest.TestCase):
    def test_invalid_event_does_not_run_caller_equality_hooks(self):
        from looking_glass.batched_recorder import BatchedEventRecorder
        from looking_glass.serial_observation import SerialObservation

        comparisons = []

        class BadKey:
            def __hash__(self):
                return hash('kind')

            def __eq__(self, other):
                comparisons.append(other)
                raise AssertionError('caller equality hook ran')

        class BadKind:
            def __eq__(self, other):
                comparisons.append(other)
                raise AssertionError('caller equality hook ran')

        observation = SerialObservation(BatchedEventRecorder())
        self.assertFalse(observation.record({BadKey(): 'window_end'}))
        self.assertFalse(observation.record({'kind': BadKind()}))
        self.assertEqual(comparisons, [])
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_invalid_request_does_not_run_caller_attribute_hooks(self):
        class NotARequest:
            def __getattribute__(self, name):
                if name == 'request_id':
                    raise AssertionError('caller attribute hook ran')
                return object.__getattribute__(self, name)

        broker, model, observation, _ = fixture()
        try:
            with self.assertRaises(InvalidRequest):
                observation.submit('team-a', NotARequest())
        finally:
            broker.close()
        self.assertEqual(model.calls, [])
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_real_serial_calls_preserve_profile_identity_and_boundaries(self):
        broker, model, observation, _ = fixture()
        try:
            handles = [observation.submit(team, request(job, row)) for team, job, row in (
                ('team-a', 'job-a', [1, 2, 3]), ('team-b', 'job-b', [4, 5, 6]))]
            outcomes = [observation.wait(handle, 3) for handle in handles]
            self.assertEqual([o.state for o in outcomes], [RequestState.SUCCEEDED] * 2)
            self.assertEqual(model.calls, [(1, 3), (1, 3)])
        finally:
            broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        events = report['events']
        ready = {e['job_id']: e for e in events if e['kind'] == 'ready'}
        selected = {e['job_id']: e for e in events if e['kind'] == 'selected'}
        terminal = {e['job_id']: e for e in events if e['kind'] == 'broker_terminal'}
        entries = {e['members'][0]['job_id']: e for e in events if e['kind'] == 'forward_entry'}
        finishes = {e['call_id']: e for e in events if e['kind'] == 'forward_finished'}
        self.assertEqual(set(entries), {'job-a', 'job-b'})
        for index, (handle, outcome) in enumerate(zip(handles, outcomes)):
            job = handle.request_id
            self.assertEqual((ready[job]['time_ns'], selected[job]['time_ns'], terminal[job]['time_ns']),
                             (outcome.ready_ns, outcome.started_ns, outcome.finished_ns))
            self.assertEqual((ready[job]['broker_id'], ready[job]['broker_job_id']),
                             (broker.broker_id, handle.job_id))
            self.assertIs(type(ready[job]['broker_job_id']), int)
            entry = entries[job]
            self.assertEqual((entry['call_id'], entry['group_id']),
                             (handle._worker_request_id, handle._worker_request_id))
            self.assertEqual(entry['profile_sha256'], 'c' * 64)
            self.assertEqual(entry['input_shape'], [1, 3])
            self.assertEqual(entry['members'], [{'job_id': job, 'team_id': handle.team_id,
                                                'broker_id': broker.broker_id,
                                                'broker_job_id': handle.job_id, 'row_offset': 0}])
            finish = finishes[entry['call_id']]
            self.assertLessEqual(selected[job]['time_ns'], entry['time_ns'])
            self.assertLessEqual(entry['time_ns'], model.call_times[index])
            self.assertLessEqual(model.call_times[index], finish['time_ns'])
            self.assertLessEqual(finish['time_ns'], terminal[job]['time_ns'])
            self.assertEqual((finish['state'], finish['synchronized'], finish['error']),
                             ('succeeded', True, None))

    def test_pending_cancel_has_no_selection_or_model_call(self):
        broker, model, observation, _ = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = observation.submit('team-a', request('job-a', [1, 2, 3]))
        try:
            self.assertTrue(model.entered.wait(3))
            second = observation.submit('team-b', request('job-b', [4, 5, 6]))
            self.assertTrue(second.cancel())
            cancelled = observation.wait(second, 3)
            self.assertEqual(cancelled.state, RequestState.CANCELLED)
        finally:
            model.proceed.set()
        outcome = observation.wait(first, 3)
        self.assertEqual(outcome.state, RequestState.SUCCEEDED)
        broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual(model.calls, [(1, 3)])
        kinds = [e['kind'] for e in report['events'] if e.get('job_id') == 'job-b']
        self.assertEqual(kinds, ['ready', 'broker_terminal'])

    def test_late_cancel_keeps_real_completed_model_observation(self):
        broker, model, observation, _ = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
        try:
            self.assertTrue(model.entered.wait(3))
            self.assertTrue(handle.cancel())
        finally:
            model.proceed.set()
        outcome = observation.wait(handle, 3)
        self.assertEqual(outcome.state, RequestState.CANCELLED)
        broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual(model.calls, [(1, 3)])
        self.assertEqual(next(e for e in report['events'] if e['kind'] == 'forward_finished')['state'],
                         'succeeded')
        self.assertEqual(next(e for e in report['events'] if e['kind'] == 'broker_terminal')['state'],
                         'cancelled')

    def test_telemetry_failure_does_not_change_serial_result(self):
        broker, model, observation, recorder = fixture()
        try:
            with patch.object(recorder, 'record', side_effect=MemoryError('observer failure')):
                handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
                result = observation.wait(handle, 3)
            self.assertEqual(result.state, RequestState.SUCCEEDED)
            self.assertEqual(model.calls, [(1, 3)])
        finally:
            broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_read_only_identity_probe_failure_does_not_change_serial_result(self):
        from looking_glass.serial_observation import SerialObservedWorker

        broker, model, observation, _ = fixture()
        try:
            with patch.object(SerialObservedWorker, '_identity', side_effect=RuntimeError('probe failed')):
                handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
                result = observation.wait(handle, 3)
            self.assertEqual(result.state, RequestState.SUCCEEDED)
            self.assertEqual(model.calls, [(1, 3)])
        finally:
            broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_observer_wait_lock_failure_preserves_real_result(self):
        class BrokenLock:
            def __enter__(self):
                raise RuntimeError('metadata lock failed')

            def __exit__(self, *args):
                return False

        broker, _, observation, _ = fixture()
        try:
            handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
            self.assertEqual(handle.wait(3).state, RequestState.SUCCEEDED)
            with patch.object(observation, '_registration_lock', BrokenLock()):
                result = observation.wait(handle, 3)
            self.assertEqual(result.state, RequestState.SUCCEEDED)
        finally:
            broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_model_entry_interrupt_is_not_swallowed_or_reported_as_a_finish(self):
        from looking_glass.grouped_observation import _SerialForwardScope

        broker, model, observation, _ = fixture()
        original = _SerialForwardScope.enter

        def interrupt_after_entry(scope, worker, request, inputs):
            original(scope, worker, request, inputs)
            raise KeyboardInterrupt()

        with patch.object(_SerialForwardScope, 'enter', new=interrupt_after_entry):
            handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
            try:
                outcome = observation.wait(handle, 3)
            except BrokerUnavailable:
                outcome = None
            broker._thread.join(3)
        self.assertFalse(broker._thread.is_alive())
        self.assertTrue(broker.quarantined)
        self.assertTrue(outcome is None or outcome.state is RequestState.FAILED)
        self.assertEqual(model.calls, [])
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertNotIn('forward_finished', [e['kind'] for e in report['events']])

    def test_unknown_profile_label_does_not_fabricate_a_digest(self):
        broker, model, observation, _ = fixture(profile_id='descriptive-label')
        try:
            handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
            self.assertEqual(observation.wait(handle, 3).state, RequestState.SUCCEEDED)
        finally:
            broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertNotIn('forward_entry', [e['kind'] for e in report['events']])
        self.assertEqual(model.calls, [(1, 3)])

    def test_missing_terminal_capture_invalidates_the_recording(self):
        broker, _, observation, _ = fixture()
        handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
        self.assertEqual(handle.wait(3).state, RequestState.SUCCEEDED)
        broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_unobserved_direct_job_cancelled_before_execution_invalidates_cut(self):
        broker, model, observation, _ = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = observation.submit('team-a', request('job-a', [1, 2, 3]))
        try:
            self.assertTrue(model.entered.wait(3))
            hidden = broker.submit('team-b', request('hidden', [4, 5, 6]))
            self.assertTrue(hidden.cancel())
            self.assertEqual(hidden.wait(3).state, RequestState.CANCELLED)
        finally:
            model.proceed.set()
        self.assertEqual(observation.wait(first, 3).state, RequestState.SUCCEEDED)
        observation.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
        self.assertTrue(observation._faulted)
        broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertEqual(model.calls, [(1, 3)])

    def test_model_may_finish_before_submit_registration_returns(self):
        broker, model, observation, _ = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        model.proceed.set()
        registration_entered, registration_continue = threading.Event(), threading.Event()
        original = observation._register
        returned, errors = [], []

        def blocked_register(*args):
            registration_entered.set()
            if not registration_continue.wait(3):
                raise RuntimeError('test timed out')
            return original(*args)

        def submit():
            try:
                returned.append(observation.submit('team-a', request('job-a', [1, 2, 3])))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=submit)
        try:
            with patch.object(observation, '_register', side_effect=blocked_register):
                thread.start()
                self.assertTrue(registration_entered.wait(3))
                self.assertTrue(model.entered.wait(3))
                self.assertEqual(model.calls, [(1, 3)])
                registration_continue.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(observation.wait(returned[0], 3).state, RequestState.SUCCEEDED)
        finally:
            registration_continue.set()
            thread.join(3)
            broker.close()
        self.assertTrue(observation.seal(producers_stopped=True)['events_complete'])

    def test_timeout_keeps_outcome_real_and_invalidates_evidence(self):
        broker, model, observation, _ = fixture()
        model.entered, model.proceed = threading.Event(), threading.Event()
        handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
        try:
            self.assertTrue(model.entered.wait(3))
            with self.assertRaises(TimeoutError):
                observation.wait(handle, 0)
        finally:
            model.proceed.set()
        self.assertEqual(observation.wait(handle, 3).state, RequestState.SUCCEEDED)
        broker.close()
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        terminals = [e for e in report['events'] if e['kind'] == 'broker_terminal']
        self.assertEqual([e['state'] for e in terminals], ['succeeded'])

    def test_window_end_during_terminal_copy_invalidates_evidence(self):
        broker, _, observation, recorder = fixture()
        handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
        self.assertEqual(handle.wait(3).state, RequestState.SUCCEEDED)
        copying, resume = threading.Event(), threading.Event()
        original = recorder.record
        outcomes, errors = [], []

        def blocked_record(event):
            if event.get('kind') == 'broker_terminal':
                copying.set()
                if not resume.wait(3):
                    raise RuntimeError('test timed out')
            return original(event)

        def wait():
            try:
                outcomes.append(observation.wait(handle, 3))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=wait)
        try:
            with patch.object(recorder, 'record', side_effect=blocked_record):
                thread.start()
                self.assertTrue(copying.wait(3))
                observation.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(outcomes[0].state, RequestState.SUCCEEDED)
        finally:
            resume.set()
            thread.join(3)
            broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_forged_same_broker_handle_cannot_receive_registered_result(self):
        broker, _, observation, _ = fixture()
        handle = observation.submit('team-a', request('job-a', [1, 2, 3]))
        forged = RequestHandle(broker, handle.job_id, handle.team_id, handle.submitted_ns)
        try:
            with self.assertRaisesRegex(ValueError, 'registered submission'):
                observation.wait(forged, 0)
            self.assertEqual(observation.wait(handle, 3).state, RequestState.SUCCEEDED)
        finally:
            broker.close()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_repeated_logical_ids_keep_results_but_invalidate_evidence(self):
        broker, model, observation, _ = fixture()
        try:
            handles = [observation.submit(team, request('same-id', row)) for team, row in (
                ('team-a', [1, 2, 3]), ('team-b', [4, 5, 6]))]
            self.assertEqual([observation.wait(handle, 3).state for handle in handles],
                             [RequestState.SUCCEEDED] * 2)
        finally:
            broker.close()
        self.assertEqual(model.calls, [(1, 3), (1, 3)])
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_real_serial_and_grouped_events_assemble_with_matching_outputs(self):
        # The fixtures use distinct model instances and a synthetic shared
        # owner label. This checks schema/output consistency only; it proves
        # neither resident-model continuity nor comparative performance.
        from looking_glass.batched_events import assemble_window
        from looking_glass.batched_recorder import BatchedEventRecorder
        from looking_glass.batched_throughput import VERSION, analyze_pair, freeze_protocol
        from looking_glass.grouped_observation import GroupedObservation
        from looking_glass.throughput_benchmark import canonical_digest
        from test_grouped_broker import fixture as grouped_fixture, receive

        requests = [request('job-a', [1, 2, 3]), request('job-b', [4, 5, 6])]
        teams = ['team-a', 'team-b']
        jobs = []
        for req, team in zip(requests, teams):
            input_hash = canonical_digest(req.inputs['input_ids'])
            request_hash = sha256((req.request_id + ':' + team + ':17:' + input_hash).encode()).hexdigest()
            jobs.append({'job_id': req.request_id, 'team_id': team, 'input_sha256': input_hash,
                         'request_sha256': request_hash, 'seed': 17})
        protocol, digest = freeze_protocol({
            'version': VERSION, 'model_sha256': 'a' * 64, 'checkpoint_sha256': 'f' * 64,
            'source_sha256': '9' * 64, 'runtime_sha256': 'b' * 64, 'capture_policy': 'none',
            'serial_policy_sha256': '8' * 64, 'grouped_policy_sha256': '7' * 64,
            'profile_sha256_by_rows': {'1': 'c' * 64, '2': 'd' * 64},
            'heldout_evidence_sha256': 'e' * 64, 'input_shape': [1, 3],
            'output_shape': [1, 3, 1], 'output_dtype': 'float32', 'output_cap_bytes': 12,
            'atol': 0., 'rtol': 0., 'jobs': jobs})

        def driver_attempt(obs, job):
            obs.record({'kind': 'attempt', 'time_ns': perf_counter_ns(), 'job_id': job['job_id'],
                        'team_id': job['team_id'], 'input_sha256': job['input_sha256'],
                        'request_sha256': job['request_sha256']})

        def driver_return(obs, job, value):
            obs.record({'kind': 'returned', 'time_ns': perf_counter_ns(), 'job_id': job['job_id'],
                        'state': 'succeeded', 'error': None, 'output_sha256': canonical_digest(value),
                        'cancellation_requested': False, 'cancellation_requested_ns': None,
                        'cancellation_accepted': None, 'cancellation_decision_ns': None})

        broker, serial_model, serial, _ = fixture()
        serial.record({'kind': 'window_start', 'time_ns': perf_counter_ns()})
        serial_outputs = {}
        for req, job in zip(requests, jobs):
            driver_attempt(serial, job)
            handle = serial.submit(job['team_id'], req)
            outcome = serial.wait(handle, 3)
            self.assertEqual(outcome.state, RequestState.SUCCEEDED)
            serial_outputs[job['job_id']] = outcome.result.logits.detach().clone()
            driver_return(serial, job, outcome.result.logits)
        broker.close()
        serial.record({'kind': 'owner_drain', 'time_ns': perf_counter_ns(), 'drained': True})
        serial.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
        serial_report = serial.seal(producers_stopped=True)

        grouped = GroupedObservation(BatchedEventRecorder())
        grouped_broker, grouped_model, host = grouped_fixture(observation=grouped, workload_id='d' * 64)
        grouped.record({'kind': 'window_start', 'time_ns': perf_counter_ns()})
        handles = []
        for req, job in zip(requests, jobs):
            driver_attempt(grouped, job)
            handles.append(grouped_broker.submit(job['team_id'], req))
        grouped_broker.run_once(1)
        grouped_outputs = {}
        for job, handle in zip(jobs, handles):
            ticket = handle.delivery()
            result = receive(ticket, 3)
            self.assertEqual(result.status, 'ok')
            grouped_outputs[job['job_id']] = result.logits.detach().clone()
            driver_return(grouped, job, result.logits)
            del result
            ticket.release()
            handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(grouped_broker.close())
        grouped.record({'kind': 'owner_drain', 'time_ns': perf_counter_ns(), 'drained': True})
        grouped.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
        grouped_report = grouped.seal(producers_stopped=True)

        def context(mode, report):
            return {'mode': mode, 'window_id': mode + '-cpu-test', 'owner_session_id': 'cpu-test',
                    'model_sha256': protocol['model_sha256'], 'runtime_sha256': protocol['runtime_sha256'],
                    'policy_sha256': protocol[mode + '_policy_sha256'], 'clock': 'perf_counter_ns',
                    'clock_resolution_ns': 100, 'events_complete': report['events_complete']}

        serial_window = assemble_window(protocol, expected_protocol_sha256=digest,
                                        context=context('serial', serial_report), events=serial_report['events'])
        grouped_window = assemble_window(protocol, expected_protocol_sha256=digest,
                                         context=context('grouped', grouped_report), events=grouped_report['events'])
        report = analyze_pair(protocol, serial_window, grouped_window, expected_protocol_sha256=digest,
                              serial_outputs=serial_outputs, grouped_outputs=grouped_outputs)
        self.assertTrue(report['measured_pair_consistent'])
        self.assertEqual(serial_model.calls, [(1, 3), (1, 3)])
        self.assertEqual(grouped_model.calls, [(2, 3)])


if __name__ == '__main__':
    unittest.main()
