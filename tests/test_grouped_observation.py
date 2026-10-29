"""Real grouped-worker observations, exercised without a CUDA device."""

import threading
import unittest
from unittest.mock import patch
import hashlib
import inspect
import sys
from time import perf_counter_ns

from test_grouped_broker import fixture, receive, request


def observed_stack(*, workload_id='d' * 64, max_events=128):
    from looking_glass.batched_recorder import BatchedEventRecorder
    from looking_glass.grouped_observation import GroupedObservation

    recorder = BatchedEventRecorder(max_events=max_events)
    observation = GroupedObservation(recorder)
    broker, model, host = fixture(observation=observation, workload_id=workload_id)
    return broker, model, host, observation, recorder


def dispose(handle):
    ticket = handle.delivery()
    result = receive(ticket, 1)
    status = result.status
    del result
    ticket.release()
    handle.dispose()
    return status


class GroupedObservationTests(unittest.TestCase):
    def test_real_model_call_captures_exact_profile_order_and_boundaries(self):
        broker, model, host, observation, _ = observed_stack()
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        self.assertTrue(broker.run_once(1))
        self.assertEqual([dispose(h) for h in (first, second)], ['ok', 'ok'])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())

        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        events = report['events']
        by_kind = {kind: [e for e in events if e['kind'] == kind] for kind in
                   ('ready', 'selected', 'forward_entry', 'forward_finished', 'broker_terminal')}
        self.assertEqual([e['job_id'] for e in by_kind['ready']], ['job-a', 'job-b'])
        self.assertEqual([e['job_id'] for e in by_kind['selected']], ['job-a', 'job-b'])
        self.assertEqual(len(by_kind['forward_entry']), 1)
        self.assertEqual(len(by_kind['forward_finished']), 1)
        self.assertEqual(model.calls, [(2, 3)])
        entry, finish = by_kind['forward_entry'][0], by_kind['forward_finished'][0]
        self.assertEqual(entry['profile_sha256'], 'd' * 64)
        self.assertEqual(entry['input_shape'], [2, 3])
        self.assertEqual([(m['job_id'], m['team_id'], m['row_offset']) for m in entry['members']],
                         [('job-a', 'team-a', 0), ('job-b', 'team-b', 1)])
        self.assertEqual([m['broker_job_id'] for m in entry['members']],
                         [e['broker_job_id'] for e in by_kind['ready']])
        self.assertEqual({m['broker_id'] for m in entry['members']}, {broker._broker_id})
        self.assertLessEqual(max(e['time_ns'] for e in by_kind['ready']),
                             min(e['time_ns'] for e in by_kind['selected']))
        self.assertLessEqual(max(e['time_ns'] for e in by_kind['selected']), entry['time_ns'])
        self.assertLessEqual(entry['time_ns'], model.call_times[0])
        self.assertLessEqual(model.call_times[0], finish['time_ns'])
        self.assertEqual((finish['call_id'], finish['state'], finish['synchronized'], finish['error']),
                         (entry['call_id'], 'succeeded', True, None))
        self.assertEqual([e['state'] for e in by_kind['broker_terminal']], ['succeeded', 'succeeded'])
        self.assertTrue(all(e['time_ns'] >= finish['time_ns'] for e in by_kind['broker_terminal']))

    def test_queued_cancel_has_no_fabricated_selection_or_forward(self):
        broker, model, host, observation, _ = observed_stack()
        handle = broker.submit('team-a', request('job-a', [1, 2, 3]))
        self.assertTrue(handle.cancel())
        self.assertEqual(dispose(handle), 'cancelled')
        self.assertTrue(broker.close())
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual([e['kind'] for e in report['events']], ['ready', 'broker_terminal'])
        self.assertEqual(report['events'][1]['state'], 'cancelled')
        self.assertEqual(model.calls, [])
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_cancel_after_selection_before_launch_is_not_a_model_call(self):
        from looking_glass.grouped_broker_state import CohortState

        broker, model, host, observation, _ = observed_stack()
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        reached, proceed = threading.Event(), threading.Event()
        original = CohortState.try_commit_launch

        def block(state):
            reached.set()
            if not proceed.wait(3):
                raise RuntimeError('test timed out')
            return original(state)

        with patch.object(CohortState, 'try_commit_launch', block):
            runner = threading.Thread(target=broker.run_once, args=(1,))
            runner.start()
            self.assertTrue(reached.wait(3))
            self.assertTrue(first.cancel())
            proceed.set()
            runner.join(3)
        self.assertFalse(runner.is_alive())
        self.assertEqual([dispose(h) for h in (first, second)], ['cancelled', 'failed'])
        self.assertTrue(broker.close())
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual([e['kind'] for e in report['events']].count('selected'), 2)
        self.assertNotIn('forward_entry', [e['kind'] for e in report['events']])
        self.assertEqual([e['state'] for e in report['events'] if e['kind'] == 'broker_terminal'],
                         ['cancelled', 'failed'])
        self.assertEqual(model.calls, [])
        self.assertEqual(host.snapshot().used_bytes, 0)

    def test_recording_failure_does_not_change_results_or_custody(self):
        broker, model, host, observation, recorder = observed_stack()
        with patch.object(recorder, 'record', side_effect=MemoryError('telemetry buffer')):
            first = broker.submit('team-a', request('job-a', [1, 2, 3]))
            second = broker.submit('team-b', request('job-b', [4, 5, 6]))
            self.assertTrue(broker.run_once(1))
        self.assertEqual([dispose(h) for h in (first, second)], ['ok', 'ok'])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_unusable_profile_identifier_invalidates_evidence_without_faking_hash(self):
        broker, model, host, observation, _ = observed_stack(workload_id='descriptive-label')
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        broker.run_once(1)
        self.assertEqual([dispose(h) for h in (first, second)], ['ok', 'ok'])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertNotIn('forward_entry', [event['kind'] for event in report['events']])

    def test_late_cancellation_preserves_peer_and_observed_member_states(self):
        broker, model, host, observation, _ = observed_stack()
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        runner = threading.Thread(target=broker.run_once, args=(1,))
        runner.start()
        try:
            self.assertTrue(model.entered.wait(3))
            group = next(iter(broker._groups.values()))
            launch_id, group_id, launch_jobs = group.launch.launch_id, group.group_id, group.launch.job_ids
            self.assertTrue(first.cancel())
        finally:
            model.proceed.set()
            runner.join(3)
        self.assertFalse(runner.is_alive())
        self.assertEqual([dispose(h) for h in (first, second)], ['cancelled', 'ok'])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        terminal = {e['job_id']: e['state'] for e in report['events'] if e['kind'] == 'broker_terminal'}
        self.assertEqual(terminal, {'job-a': 'cancelled', 'job-b': 'succeeded'})
        entry = next(e for e in report['events'] if e['kind'] == 'forward_entry')
        self.assertEqual((entry['call_id'], entry['group_id']), (launch_id, group_id))
        self.assertEqual(tuple(m['broker_job_id'] for m in entry['members']), launch_jobs)
        self.assertEqual(model.calls, [(2, 3)])

    def test_oom_dominates_cancellation_and_forward_remains_uncertain(self):
        broker, model, host, observation, _ = observed_stack()
        model.fail = True
        model.entered, model.proceed = threading.Event(), threading.Event()
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        runner = threading.Thread(target=broker.run_once, args=(1,))
        runner.start()
        try:
            self.assertTrue(model.entered.wait(3))
            self.assertTrue(first.cancel())
        finally:
            model.proceed.set()
            runner.join(3)
        self.assertFalse(runner.is_alive())
        first_ticket, second_ticket = first.delivery(), second.delivery()
        left, right = receive(first_ticket, 1), receive(second_ticket, 1)
        self.assertEqual((left.status, right.status), ('failed', 'failed'))
        self.assertTrue(left.cancel_requested)
        self.assertEqual((left.error_code, right.error_code), ('out_of_memory', 'out_of_memory'))
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        del left, right
        first_ticket.release()
        second_ticket.release()
        report = observation.seal(producers_stopped=True)
        self.assertTrue(report['events_complete'])
        self.assertEqual([e['kind'] for e in report['events']].count('forward_entry'), 1)
        self.assertEqual([e['kind'] for e in report['events']].count('forward_uncertain'), 1)
        self.assertNotIn('forward_finished', [e['kind'] for e in report['events']])
        self.assertEqual([e['state'] for e in report['events'] if e['kind'] == 'broker_terminal'],
                         ['failed', 'failed'])

    def test_duplicate_logical_ids_invalidate_evidence_but_not_real_jobs(self):
        broker, model, host, observation, _ = observed_stack()
        first = broker.submit('team-a', request('same-id', [1, 2, 3]))
        second = broker.submit('team-b', request('same-id', [4, 5, 6]))
        broker.run_once(1)
        self.assertEqual([dispose(h) for h in (first, second)], ['ok', 'ok'])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_overflow_invalidates_only_evidence(self):
        broker, model, host, observation, _ = observed_stack(max_events=1)
        first = broker.submit('team-a', request('job-a', [1, 2, 3]))
        second = broker.submit('team-b', request('job-b', [4, 5, 6]))
        broker.run_once(1)
        self.assertEqual([dispose(h) for h in (first, second)], ['ok', 'ok'])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertGreater(report['lost_capacity'], 0)

    def test_control_interrupt_is_not_swallowed_by_the_observer(self):
        from looking_glass.batched_recorder import BatchedEventRecorder
        from looking_glass.grouped_observation import GroupedObservation

        recorder = BatchedEventRecorder()
        observation = GroupedObservation(recorder)
        with patch.object(recorder, 'record', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                observation.record({'kind': 'window_start', 'time_ns': 1})
        with patch('looking_glass.grouped_observation.perf_counter_ns', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                observation._stamp()
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_inflight_clock_interrupt_quarantines_and_never_invents_a_forward(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.grouped_observation import _ForwardScope

        broker, model, host, observation, _ = observed_stack()
        broker.submit('team-a', request('job-a', [1, 2, 3]))
        broker.submit('team-b', request('job-b', [4, 5, 6]))

        def clock():
            if any(frame.function == 'enter' and frame.frame.f_code is _ForwardScope.enter.__code__
                   for frame in inspect.stack()):
                raise KeyboardInterrupt()
            return perf_counter_ns()

        with patch('looking_glass.grouped_observation.perf_counter_ns', side_effect=clock):
            with self.assertRaises(GroupedBrokerUnavailable):
                broker.run_once(1)
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(model.calls, [])
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertNotIn('forward_finished', [e['kind'] for e in report['events']])

    def test_cancel_observer_interrupt_quarantines_and_retains_the_snapshot_claim(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable

        broker, model, host, observation, recorder = observed_stack()
        handle = broker.submit('team-a', request('job-a', [1, 2, 3]))
        with patch.object(recorder, 'record', side_effect=KeyboardInterrupt):
            with self.assertRaises(GroupedBrokerUnavailable):
                handle.cancel()
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(model.calls, [])
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_lost_forward_context_return_is_cleared_before_quarantine(self):
        from looking_glass import grouped_observation
        from looking_glass.grouped_broker import GroupedBrokerUnavailable

        broker, model, host, observation, _ = observed_stack()
        broker.submit('team-a', request('job-a', [1, 2, 3]))
        broker.submit('team-b', request('job-b', [4, 5, 6]))

        def trace(frame, event, arg):
            if frame.f_code is type(observation)._start_forward.__code__ and event == 'return':
                raise KeyboardInterrupt()
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(GroupedBrokerUnavailable):
                broker.run_once(1)
        finally:
            sys.settrace(None)
        self.assertIsNone(getattr(grouped_observation._local, 'forward', None))
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(model.calls, [])
        self.assertFalse(observation.seal(producers_stopped=True)['events_complete'])

    def test_interruption_after_entry_stamp_before_model_call_is_not_a_finish(self):
        from looking_glass.grouped_broker import GroupedBrokerUnavailable
        from looking_glass.resident_worker import ResidentWorker

        broker, model, host, observation, _ = observed_stack()
        broker.submit('team-a', request('job-a', [1, 2, 3]))
        broker.submit('team-b', request('job-b', [4, 5, 6]))
        lines, first_line = inspect.getsourcelines(ResidentWorker._perform_with_custody)
        call_line = first_line + next(i for i, line in enumerate(lines)
                                      if line.strip() == 'output = self._model(**device_inputs)')

        def trace(frame, event, arg):
            if (frame.f_code is ResidentWorker._perform_with_custody.__code__
                    and event == 'line' and frame.f_lineno == call_line):
                raise KeyboardInterrupt()
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(GroupedBrokerUnavailable):
                broker.run_once(1)
        finally:
            sys.settrace(None)
        self.assertTrue(broker.quarantined)
        self.assertGreater(host.snapshot().used_bytes, 0)
        self.assertEqual(model.calls, [])
        report = observation.seal(producers_stopped=True)
        self.assertFalse(report['events_complete'])
        self.assertNotIn('forward_finished', [e['kind'] for e in report['events']])

    def test_unobserved_resident_does_not_swallow_control_interrupt(self):
        from looking_glass.resident_worker import ResidentWorker
        from test_grouped_broker import ObservedModel

        class BadLocal:
            @property
            def forward(self):
                raise KeyboardInterrupt()

        worker = ResidentWorker('a' * 64, ObservedModel().eval(), device='cpu',
                                max_batch_size=1, max_sequence_length=3,
                                max_input_bytes=24, max_output_bytes=64)
        with patch('looking_glass.grouped_observation._local', BadLocal()):
            with self.assertRaises(KeyboardInterrupt):
                worker.execute(request('job', [1, 2, 3]))

    def test_real_raw_observations_assemble_into_an_analyzer_window(self):
        from looking_glass.batched_events import assemble_window
        from looking_glass.batched_throughput import VERSION, freeze_protocol
        from looking_glass.throughput_benchmark import canonical_digest

        broker, model, host, observation, _ = observed_stack()
        requests = [request('job-a', [1, 2, 3]), request('job-b', [4, 5, 6])]
        teams = ['team-a', 'team-b']
        jobs = []
        for req, team in zip(requests, teams):
            input_hash = canonical_digest(req.inputs['input_ids'])
            request_hash = hashlib.sha256(
                (req.request_id + ':' + team + ':17:' + input_hash).encode()).hexdigest()
            jobs.append({'job_id': req.request_id, 'team_id': team,
                         'input_sha256': input_hash, 'request_sha256': request_hash, 'seed': 17})
        protocol, digest = freeze_protocol({
            'version': VERSION, 'model_sha256': 'a' * 64, 'checkpoint_sha256': 'f' * 64,
            'source_sha256': '9' * 64, 'runtime_sha256': 'b' * 64, 'capture_policy': 'none',
            'serial_policy_sha256': '8' * 64, 'grouped_policy_sha256': '7' * 64,
            'profile_sha256_by_rows': {'1': 'c' * 64, '2': 'd' * 64},
            'heldout_evidence_sha256': 'e' * 64, 'input_shape': [1, 3],
            'output_shape': [1, 3, 1], 'output_dtype': 'float32', 'output_cap_bytes': 12,
            'atol': 0., 'rtol': 0., 'jobs': jobs})
        observation.record({'kind': 'window_start', 'time_ns': perf_counter_ns()})
        handles = []
        for job, req in zip(jobs, requests):
            observation.record({'kind': 'attempt', 'time_ns': perf_counter_ns(),
                                'job_id': job['job_id'], 'team_id': job['team_id'],
                                'input_sha256': job['input_sha256'], 'request_sha256': job['request_sha256']})
            handles.append(broker.submit(job['team_id'], req))
        broker.run_once(1)
        for job, handle in zip(jobs, handles):
            ticket = handle.delivery()
            result = receive(ticket, 1)
            output_digest = canonical_digest(result.logits)
            observation.record({'kind': 'returned', 'time_ns': perf_counter_ns(),
                                'job_id': job['job_id'], 'state': 'succeeded', 'error': None,
                                'output_sha256': output_digest, 'cancellation_requested': False,
                                'cancellation_requested_ns': None, 'cancellation_accepted': None,
                                'cancellation_decision_ns': None})
            del result
            ticket.release()
            handle.dispose()
        self.assertEqual(host.snapshot().used_bytes, 0)
        self.assertTrue(broker.close())
        observation.record({'kind': 'owner_drain', 'time_ns': perf_counter_ns(), 'drained': True})
        observation.record({'kind': 'window_end', 'time_ns': perf_counter_ns()})
        report = observation.seal(producers_stopped=True)
        context = {'mode': 'grouped', 'window_id': 'cpu-test', 'owner_session_id': 'cpu-owner',
                   'model_sha256': protocol['model_sha256'], 'runtime_sha256': protocol['runtime_sha256'],
                   'policy_sha256': protocol['grouped_policy_sha256'], 'clock': 'perf_counter_ns',
                   'clock_resolution_ns': 100, 'events_complete': report['events_complete']}
        window = assemble_window(protocol, expected_protocol_sha256=digest, context=context,
                                 events=report['events'])
        self.assertEqual(model.calls, [(2, 3)])
        self.assertEqual(len(window['calls']), 1)
        self.assertEqual(window['calls'][0]['input_shape'], [2, 3])
        self.assertEqual([row['state'] for row in window['attempts']], ['succeeded', 'succeeded'])


if __name__ == '__main__':
    unittest.main()
