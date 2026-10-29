import importlib
import unittest

try:
    benchmark = importlib.import_module('looking_glass.throughput_benchmark')
except ModuleNotFoundError as exc:
    if exc.name != 'looking_glass.throughput_benchmark':
        raise
    benchmark = None


class ThroughputSummaryTests(unittest.TestCase):
    def test_attempts_failures_and_time_boundaries_are_accounted_for(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        samples = [
            {'job_id': 'a', 'team_id': 'red', 'workload_class': 'short',
             'units': 10, 'state': 'succeeded', 'attempt_ns': 100,
             'submitted_ns': 120, 'ready_ns': 130, 'started_ns': 150, 'finished_ns': 250,
             'returned_ns': 300, 'output_digest': 'abc', 'error': None},
            {'job_id': 'b', 'team_id': 'blue', 'workload_class': 'short',
             'units': 20, 'state': 'failed', 'attempt_ns': 200,
             'submitted_ns': 220, 'ready_ns': 240, 'started_ns': 260, 'finished_ns': 460,
             'returned_ns': 600, 'output_digest': None, 'error': 'bad'},
            {'job_id': 'c', 'team_id': 'red', 'workload_class': 'short',
             'units': 5, 'state': 'rejected', 'attempt_ns': 300,
             'submitted_ns': None, 'started_ns': None, 'finished_ns': None,
             'returned_ns': 320, 'output_digest': None, 'error': 'full'},
        ]
        summary = benchmark.summarize_window(samples, start_ns=0, end_ns=1_000_000_000)
        self.assertEqual(summary['counts'], {
            'attempted': 3, 'accepted': 2, 'rejected': 1, 'succeeded': 1,
            'failed': 1, 'cancelled': 0, 'unresolved': 0, 'invalid_output': 0, 'submit_error': 0})
        self.assertEqual(summary['successes_per_second'], 1.0)
        self.assertEqual(summary['successful_units_per_second'], 10.0)
        self.assertEqual(summary['successful_units'], 10)
        self.assertAlmostEqual(summary['nonsuccess_rate'], 2 / 3)
        self.assertEqual(summary['queue_wait_ns_started'], {'p50': 20, 'p95': 20, 'max': 20, 'n': 2})
        self.assertEqual(summary['submission_to_service_ns_started'], {'p50': 35, 'p95': 40, 'max': 40, 'n': 2})
        self.assertEqual(summary['service_ns_started'], {'p50': 150, 'p95': 200, 'max': 200, 'n': 2})
        self.assertEqual(summary['response_ns_all_attempts'], {'p50': 200, 'p95': 400, 'max': 400, 'n': 3})
        self.assertEqual(summary['teams']['red']['counts']['attempted'], 2)
        self.assertEqual(summary['teams']['red']['counts']['succeeded'], 1)
        self.assertEqual(summary['teams']['red']['successful_units_per_second'], 10.0)
        self.assertEqual(summary['teams']['blue']['counts']['failed'], 1)
        self.assertAlmostEqual(summary['completion_balance_jain'], 0.5)

    def test_empty_and_invalid_intervals_are_rejected(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        with self.assertRaises(ValueError):
            benchmark.summarize_window([], start_ns=0, end_ns=0)
        with self.assertRaises(ValueError):
            benchmark.summarize_window([], start_ns=0, end_ns=1)
        row = {'job_id': 'a', 'team_id': 'red', 'workload_class': 'short', 'units': 1,
               'state': 'succeeded', 'attempt_ns': 100, 'submitted_ns': 90,
               'started_ns': 150, 'finished_ns': 160, 'returned_ns': 170,
               'output_digest': 'x', 'error': None}
        with self.assertRaises(ValueError):
            benchmark.summarize_window([row], start_ns=0, end_ns=1000)

class StateBoundaryTests(unittest.TestCase):
    def test_rejection_latency_does_not_hide_success_latency(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        samples = [
            {'job_id': 'a', 'team_id': 't', 'workload_class': 'w', 'units': 1,
             'state': 'succeeded', 'attempt_ns': 100, 'submitted_ns': 110,
             'started_ns': 120, 'finished_ns': 1_100, 'returned_ns': 1_200,
             'output_digest': 'abc', 'error': None},
            {'job_id': 'b', 'team_id': 't', 'workload_class': 'w', 'units': 1,
             'state': 'rejected', 'attempt_ns': 150, 'submitted_ns': None,
             'started_ns': None, 'finished_ns': None, 'returned_ns': 151,
             'output_digest': None, 'error': 'full'},
        ]
        summary = benchmark.summarize_window(samples, start_ns=100, end_ns=1_300)
        self.assertEqual(summary['timing_by_state']['succeeded']['response_ns']['p50'], 1_100)
        self.assertEqual(summary['timing_by_state']['rejected']['response_ns']['p50'], 1)
        self.assertEqual(summary['timing_by_state']['rejected']['queue_wait_ns']['n'], 0)
        with self.assertRaises(ValueError):
            benchmark.summarize_window(samples, start_ns=100, end_ns=1_199)
        with self.assertRaises(ValueError):
            benchmark.summarize_window([samples[0], samples[0]], start_ns=100, end_ns=1_300)
        malformed = dict(samples[1], error=None)
        with self.assertRaises(ValueError):
            benchmark.summarize_window([malformed], start_ns=100, end_ns=1_300)


class TerminalIntegrityTests(unittest.TestCase):
    def test_unresolved_cannot_have_a_finished_timestamp_and_success_cannot_hide_error(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        row = {'job_id': 'a', 'team_id': 'red', 'workload_class': 'short', 'units': 1,
               'state': 'unresolved', 'attempt_ns': 100, 'submitted_ns': 110,
               'started_ns': 120, 'finished_ns': 130, 'returned_ns': 140,
               'output_digest': None, 'error': 'wait timed out'}
        with self.assertRaises(ValueError):
            benchmark.summarize_window([row], start_ns=100, end_ns=200)
        row = dict(row, state='succeeded', output_digest='abc')
        with self.assertRaises(ValueError):
            benchmark.summarize_window([row], start_ns=100, end_ns=200)


class DigestIntegrityTests(unittest.TestCase):
    def test_success_digest_must_be_a_string(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        row = {'job_id': 'a', 'team_id': 'red', 'workload_class': 'short', 'units': 1,
               'state': 'succeeded', 'attempt_ns': 100, 'submitted_ns': 110,
               'started_ns': 120, 'finished_ns': 130, 'returned_ns': 140,
               'output_digest': True, 'error': None}
        with self.assertRaises(ValueError):
            benchmark.summarize_window([row], start_ns=100, end_ns=200)

class RunnerTests(unittest.TestCase):
    def test_worker_syncs_on_success_and_error(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        calls = []
        class Worker:
            poisoned = False
            def snapshot_request(self, request):
                return request
            def execute(self, request, cancel_event):
                calls.append('execute')
                if request == 'bad':
                    raise RuntimeError('failure')
                return request
            def close(self):
                calls.append('close')
        worker = benchmark.SynchronizedWorker(Worker(), lambda: calls.append('sync'))
        self.assertEqual(worker.execute('good', None), 'good')
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            worker.execute('bad', None)
        self.assertEqual(calls, ['sync', 'execute', 'sync', 'sync', 'execute', 'sync'])
        self.assertEqual(worker.snapshot_request('test'), 'test')
        worker.close()
        self.assertEqual(calls[-1], 'close')

    def test_serial_and_concurrent_windows_keep_every_outcome_and_same_jobs(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        class QueueFull(Exception):
            pass

        class CPUFakeBroker:
            def __init__(self):
                self.executor = ThreadPoolExecutor(max_workers=1)
                self.lock = threading.Lock()
                self.active = 0
                self.peak_active = 0
            def submit(self, team_id, request):
                submitted = time.perf_counter_ns()
                if request == 'reject':
                    raise QueueFull('full')
                with self.lock:
                    self.active += 1
                    self.peak_active = max(self.active, self.peak_active)
                def execute():
                    started = time.perf_counter_ns()
                    time.sleep(.01)
                    result = SimpleNamespace(state='failed' if request == 'fail' else 'succeeded',
                        submitted_ns=submitted, started_ns=started, finished_ns=time.perf_counter_ns(),
                        result=request, error_type='RuntimeError' if request == 'fail' else None,
                        error_message='injected' if request == 'fail' else None)
                    with self.lock:
                        self.active -= 1
                    return result
                future = self.executor.submit(execute)
                return SimpleNamespace(wait=lambda timeout: future.result(timeout=timeout),
                                       cancel=future.cancel, submitted_ns=submitted)
            def close(self):
                self.executor.shutdown()
        def digest(value):
            return hashlib.sha256(value.encode()).hexdigest()
        jobs = [benchmark.WorkItem('a', 'red', 'short', 2, 7, digest('x'), 'none', 'x'),
                benchmark.WorkItem('b', 'blue', 'short', 2, 7, digest('y'), 'none', 'y'),
                benchmark.WorkItem('c', 'red', 'short', 2, 7, digest('fail'), 'none', 'fail'),
                benchmark.WorkItem('d', 'blue', 'short', 2, 7, digest('reject'), 'none', 'reject')]
        windows = []
        for mode in ('serial', 'shared'):
            broker = CPUFakeBroker()
            try:
                result = benchmark.run_window(broker, jobs, mode=mode, output_digest=digest,
                    input_digest=digest, rejection_errors=(QueueFull,), model_spec={'id': 'fake'}, unit_name='tokens')
                windows.append(result)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['summary']['counts']['attempted'], 4)
                self.assertEqual(result['summary']['counts']['succeeded'], 2)
                self.assertEqual(result['summary']['counts']['failed'], 1)
                self.assertEqual(result['summary']['counts']['rejected'], 1)
                self.assertEqual(result['summary']['successful_units'], 4)
                self.assertEqual({r['job_id'] for r in result['samples']}, {'a', 'b', 'c', 'd'})
                if mode == 'shared':
                    self.assertGreaterEqual(broker.peak_active, 2)
            finally:
                broker.close()
        comparison = benchmark.compare_windows(*windows)
        self.assertTrue(comparison['same_manifest'])
        self.assertTrue(comparison['matching_common_outputs'])
        self.assertEqual(comparison['compared_successes'], 2)

    def test_mutated_input_and_output_mismatch_invalidate_comparison(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        # The manifest is made from actual inputs by a caller-provided digest;
        # both windows must also agree on successful per-job outputs.
        base = {'mode': 'serial', 'start_ns': 1, 'end_ns': 2, 'status': 'complete', 'manifest_sha256': 'same', 'samples': [
            {'job_id': 'one', 'state': 'succeeded', 'output_digest': 'a'}]}
        different = {'mode': 'shared', 'start_ns': 3, 'end_ns': 4, 'status': 'complete', 'manifest_sha256': 'same', 'samples': [
            {'job_id': 'one', 'state': 'succeeded', 'output_digest': 'b'}]}
        self.assertFalse(benchmark.compare_windows(base, different)['matching_common_outputs'])
        different['manifest_sha256'] = 'different'
        self.assertFalse(benchmark.compare_windows(base, different)['same_manifest'])

    def test_missing_or_tampered_manifest_jobs_invalidate_comparison(self):
        import hashlib
        import json
        manifest = {'model_spec': {'id': 'fixture'}, 'unit_name': 'positions', 'jobs': [
            {'job_id': 'a', 'team_id': 'one', 'workload_class': 'short', 'units': 1},
            {'job_id': 'b', 'team_id': 'two', 'workload_class': 'short', 'units': 1}]}
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        first = {'mode': 'serial', 'status': 'complete', 'start_ns': 1, 'end_ns': 2,
                 'manifest': manifest, 'manifest_sha256': digest,
                 'unattempted_job_ids': [], 'fatal_errors': [],
                 'samples': [{'job_id': 'a', 'state': 'succeeded', 'output_digest': 'digest'}]}
        second = dict(first, mode='shared', start_ns=3, end_ns=4)
        self.assertFalse(benchmark.compare_windows(first, second)['valid_paired_comparison'])
        self.assertFalse(benchmark.compare_windows(first, second)['all_jobs_succeeded_in_both'])
        manifest['jobs'].pop()
        self.assertFalse(benchmark.compare_windows(first, second)['valid_paired_comparison'])

class FailureHandlingTests(unittest.TestCase):
    def test_unresolved_timeout_marks_window_incomplete_and_stops_that_client(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import time
        class Full(Exception):
            pass
        class NeverFinishes:
            def __init__(self):
                self.submitted_ns = time.perf_counter_ns()
                self.cancelled = False
            def wait(self, timeout):
                raise TimeoutError('still running')
            def cancel(self):
                self.cancelled = True
                return True
        class Broker:
            def submit(self, team_id, request):
                self.handle = NeverFinishes()
                return self.handle
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        jobs = [benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a'),
                benchmark.WorkItem('b', 'red', 'short', 1, 0, digest('b'), 'none', 'b')]
        broker = Broker()
        window = benchmark.run_window(broker, jobs, mode='serial', output_digest=digest,
                                      input_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='tokens',
                                      wait_timeout=.001, cancel_timeout=.001)
        self.assertEqual(window['status'], 'incomplete')
        self.assertEqual(window['unattempted_job_ids'], ['b'])
        self.assertTrue(broker.handle.cancelled)
        self.assertEqual(window['summary']['counts']['unresolved'], 1)
        self.assertFalse(benchmark.compare_windows(window, window)['valid_paired_comparison'])

    def test_changed_input_is_rejected_before_any_submission(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        class Broker:
            def submit(self, team_id, request):
                raise AssertionError('should not be submitted')
        class Full(Exception):
            pass
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'original', 'none', 'changed')
        with self.assertRaisesRegex(ValueError, 'input digest changed'):
            benchmark.run_window(Broker(), [job], mode='serial', output_digest=str,
                                 input_digest=str, rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='tokens')

class ReviewRegressionTests(unittest.TestCase):
    def test_manifest_changes_when_request_configuration_changes(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        from dataclasses import dataclass
        @dataclass
        class Request:
            request_id: str
            seed: int
            mode: str
        original = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'same', 'none', Request('a', 0, 'direct'))
        changed = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'same', 'none', Request('a', 0, 'gradient'))
        _, digest_a = benchmark.workload_manifest([original], lambda r: 'same', model_spec={'id': 'model-a'}, unit_name='tokens')
        _, digest_b = benchmark.workload_manifest([changed], lambda r: 'same', model_spec={'id': 'model-a'}, unit_name='tokens')
        _, digest_c = benchmark.workload_manifest([original], lambda r: 'same', model_spec={'id': 'model-b'}, unit_name='tokens')
        self.assertNotEqual(digest_a, digest_b)
        self.assertNotEqual(digest_a, digest_c)

    def test_unexpected_submission_error_is_an_attempt_not_an_unattempted_job(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        class Full(Exception):
            pass
        class Broken:
            def submit(self, team_id, request):
                raise ValueError('unexpected')
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'a', 'none', 'a')
        result = benchmark.run_window(Broken(), [job], mode='serial', input_digest=str,
                                      output_digest=str, rejection_errors=(Full,),
                                      model_spec={'id': 'fake'}, unit_name='tokens')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['unattempted_job_ids'], [])
        self.assertEqual(result['summary']['counts']['submit_error'], 1)
        self.assertEqual(result['summary']['counts']['attempted'], 1)

    def test_synchronization_failure_poisons_proxy(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        class Worker:
            poisoned = False
            def execute(self, request, cancel):
                return 'ok'
        n = [0]
        def sync():
            n[0] += 1
            if n[0] == 2:
                raise RuntimeError('sync failed')
        proxy = benchmark.SynchronizedWorker(Worker(), sync)
        with self.assertRaisesRegex(RuntimeError, 'sync failed'):
            proxy.execute('x', None)
        self.assertTrue(proxy.poisoned)

    def test_identical_window_is_not_a_paired_load_comparison(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        first = {'mode': 'serial', 'status': 'complete', 'start_ns': 1, 'end_ns': 2,
                 'manifest_sha256': 'same', 'samples': [
                     {'job_id': 'one', 'state': 'succeeded', 'output_digest': 'a'}]}
        self.assertFalse(benchmark.compare_windows(first, first)['valid_paired_comparison'])

class WaitFailureTests(unittest.TestCase):
    def test_unexpected_wait_error_preserves_attempt_as_unresolved(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import time
        class Full(Exception):
            pass
        class Handle:
            def __init__(self):
                self.submitted_ns = time.perf_counter_ns()
                self.cancelled = False
            def wait(self, timeout):
                raise ValueError('broken wait')
            def cancel(self):
                self.cancelled = True
                return True
        class Broker:
            def submit(self, team_id, request):
                self.handle = Handle()
                return self.handle
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'a', 'none', 'a')
        broker = Broker()
        window = benchmark.run_window(broker, [job], mode='shared', input_digest=str,
            output_digest=str, rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='tokens')
        self.assertEqual(window['status'], 'incomplete')
        self.assertEqual(window['unattempted_job_ids'], [])
        self.assertEqual(window['summary']['counts']['unresolved'], 1)
        self.assertTrue(broker.handle.cancelled)
        self.assertIn('broken wait', window['samples'][0]['error'])

class ProtocolArtifactTests(unittest.TestCase):
    def test_protocol_saves_raw_samples_and_never_overwrites(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import json
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Broker:
            broker_id = 'a' * 32
            def __init__(self):
                self.counter = 0
            def submit(self, team_id, request):
                self.counter += 1
                submitted = time.perf_counter_ns()
                ready = time.perf_counter_ns()
                started = time.perf_counter_ns()
                finished = time.perf_counter_ns()
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=request,
                    error_type=None, error_message=None)
                return SimpleNamespace(job_id=self.counter, wait=lambda timeout: outcome)
        class Memory:
            def before(self):
                return {'device_free_bytes': 100, 'device_total_bytes': 1000,
                        'process_allocated_bytes': 10, 'process_reserved_bytes': 20}
            def after(self):
                return {'device_free_bytes': 80, 'device_total_bytes': 1000,
                        'process_allocated_bytes': 15, 'process_reserved_bytes': 25,
                        'process_peak_allocated_bytes': 20, 'process_peak_reserved_bytes': 30}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        jobs = [benchmark.WorkItem('a', 'red', 'short', 2, 7, digest('a'), 'none', 'a'),
                benchmark.WorkItem('b', 'blue', 'short', 3, 8, digest('b'), 'none', 'b')]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'evidence'
            result = benchmark.run_protocol(output, Broker(), jobs, input_digest=digest,
                output_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake-cpu'},
                unit_name='tokens', metadata={'external_gpu_activity': 'none; cpu fake', 'source_commit': 'test-fixture'},
                telemetry=Memory(), repetitions=2, warmup_rounds=1, protocol_seed=3)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['comparisons']), 2)
            self.assertTrue(all(c['valid_paired_comparison'] for c in result['comparisons']))
            self.assertEqual(len(result['windows']), 4)
            self.assertEqual(result['windows'][0]['memory']['after']['process_peak_allocated_bytes'], 20)
            self.assertEqual(len((output / 'raw_samples.jsonl').read_text().splitlines()), 8)
            self.assertEqual(len((output / 'warmup_samples.jsonl').read_text().splitlines()), 8)
            self.assertEqual(json.loads((output / 'results.json').read_text())['status'], 'complete')
            self.assertEqual(result['evidence_sha256']['raw_samples.jsonl'],
                hashlib.sha256((output / 'raw_samples.jsonl').read_bytes()).hexdigest())
            benchmark._validate_recorded_windows(output, result, jobs=result['manifest']['jobs'], repetitions=2, protocol_seed=3)
            raw_path = output / 'raw_samples.jsonl'
            original_raw = raw_path.read_text()
            rows = [json.loads(line) for line in original_raw.splitlines()]
            rows[0].update(state='failed', error='synthetic failure', output_digest=None)
            raw_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'unsuccessful'):
                benchmark._validate_recorded_windows(output, result, jobs=result['manifest']['jobs'], repetitions=2, protocol_seed=3)
            raw_path.write_text(original_raw)
            warm_path = output / 'warmup_samples.jsonl'
            original_warm = warm_path.read_text()
            warm = [json.loads(line) for line in original_warm.splitlines()]
            for name in ('attempt_ns', 'submitted_ns', 'ready_ns', 'started_ns', 'finished_ns', 'returned_ns'):
                warm[0][name] += 10 ** 12
            warm_path.write_text(''.join(json.dumps(row) + '\n' for row in warm))
            with self.assertRaisesRegex(ValueError, 'chronology'):
                benchmark._validate_recorded_windows(output, result, jobs=result['manifest']['jobs'], repetitions=2, protocol_seed=3)
            warm_path.write_text(original_warm)
            overlapping = json.loads(json.dumps(result))
            serial = next(window for window in overlapping['windows']
                          if window['repetition'] == 0 and window['mode'] == 'serial')
            first = next(row for row in serial['samples'] if row['job_id'] == 'a')
            second = next(row for row in serial['samples'] if row['job_id'] == 'b')
            second['attempt_ns'] = first['attempt_ns']
            serial['summary'] = benchmark.summarize_window(serial['samples'],
                start_ns=serial['start_ns'], end_ns=serial['end_ns'])
            raw_rows = [json.loads(line) for line in original_raw.splitlines()]
            next(row for row in raw_rows if row['repetition'] == 0 and row['mode'] == 'serial'
                 and row['job_id'] == 'b')['attempt_ns'] = second['attempt_ns']
            raw_path.write_text(''.join(json.dumps(row) + '\n' for row in raw_rows))
            partial = output / 'partial_windows.jsonl'
            partial.write_text(''.join(json.dumps(row) + '\n' for row in overlapping['windows']))
            with self.assertRaisesRegex(ValueError, 'closed-loop'):
                benchmark._validate_recorded_windows(output, overlapping, jobs=overlapping['manifest']['jobs'],
                                                     repetitions=2, protocol_seed=3)
            raw_path.write_text(original_raw)
            tampered = json.loads(json.dumps(result))
            tampered['windows'][0]['summary']['successful_units'] += 1
            partial = output / 'partial_windows.jsonl'
            partial.write_text(''.join(json.dumps(row) + '\n' for row in tampered['windows']))
            with self.assertRaisesRegex(ValueError, 'summary'):
                benchmark._validate_recorded_windows(output, tampered, jobs=tampered['manifest']['jobs'], repetitions=2, protocol_seed=3)
            invalid_memory = json.loads(json.dumps(result))
            invalid_memory['windows'][0]['memory']['before']['process_allocated_bytes'] = 21
            invalid_memory['windows'][0]['memory']['before']['process_reserved_bytes'] = 25
            partial.write_text(''.join(json.dumps(row) + '\n' for row in invalid_memory['windows']))
            with self.assertRaisesRegex(ValueError, 'baseline or total'):
                benchmark._validate_recorded_windows(output, invalid_memory,
                    jobs=invalid_memory['manifest']['jobs'], repetitions=2, protocol_seed=3)
            with self.assertRaises(FileExistsError):
                benchmark.run_protocol(output, Broker(), jobs, input_digest=digest, output_digest=digest,
                    rejection_errors=(Full,), model_spec={'id': 'fake-cpu'}, unit_name='tokens',
                    metadata={'external_gpu_activity': 'none; cpu fake', 'source_commit': 'test-fixture'}, telemetry=Memory())

    def test_owner_evidence_gate_accepts_complete_records_and_rejects_changed_raw(self):
        import hashlib
        import json
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class Full(Exception): pass
        class Broker:
            broker_id = 'b' * 32
            counter = 0
            def submit(self, team, request):
                self.counter += 1
                submitted, ready, started, finished = (time.perf_counter_ns() for _ in range(4))
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=request, error_type=None, error_message=None)
                return SimpleNamespace(job_id=self.counter, wait=lambda timeout: outcome)
        class Memory:
            def before(self):
                return {'device_free_bytes': 100, 'device_total_bytes': 1000,
                        'process_allocated_bytes': 10, 'process_reserved_bytes': 20}
            def after(self):
                return {'device_free_bytes': 80, 'device_total_bytes': 1000,
                        'process_allocated_bytes': 15, 'process_reserved_bytes': 25,
                        'process_peak_allocated_bytes': 20, 'process_peak_reserved_bytes': 30}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        jobs = [benchmark.WorkItem('a', 'red', 'two', 2, 7, digest('a'), 'none', 'a'),
                benchmark.WorkItem('b', 'blue', 'three', 3, 8, digest('b'), 'none', 'b')]
        gpu = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
        environment = {'device': 'cuda:0', 'runtime_gpu_uuid': gpu, 'torch': 'test-torch',
            'transformers': 'test-transformers', 'cuda_runtime': '12.8', 'device_name': 'fake',
            'compute_capability': [7, 5], 'deterministic_algorithms': True,
            'bytecode_isolation': 'empty_prefix_no_writes',
            'python_no_site': True, 'python_ignores_environment': True,
            'python_isolated': True, 'python_hash_randomization': True,
            'cuda_environment': benchmark._configure_cuda_environment({}),
            'allocator_backend': 'native', 'allocator_fraction': 0.75,
            'torch_threads': 1, 'torch_interop_threads': 1,
            'cudnn_benchmark': False, 'cudnn_deterministic': True, 'cudnn_allow_tf32': False,
            'matmul_allow_tf32': False, 'cublas_workspace_config': ':4096:8'}
        source = {'source_commit': 'a' * 40, 'source_dirty': False,
                  'source_sha256': {name: 'c' * 64 for name in benchmark._SOURCE_FILES}}
        metadata = {'model_id': 'fake', 'device': 'cpu', 'dtype': 'float32',
                    'torch_version': 'test-torch', 'transformers_version': 'test-transformers'}
        model_spec = dict(metadata, device='cuda:0', eval_mode=True, capture_policy='none')
        fingerprint = benchmark._model_fingerprint(model_spec, environment)
        model_spec['service_model_fingerprint'] = fingerprint
        limits = {'max_batch_size': 1, 'max_sequence_length': 128, 'max_input_bytes': 1024 * 1024,
                  'max_output_bytes': 8 * 1024 * 1024, 'max_gpu_trace_bytes': 0,
                  'max_host_trace_bytes': 0, 'gpu_model_load_estimate_bytes': 2 * 1024 ** 3,
                  'gpu_per_forward_estimate_bytes': 1024 ** 3, 'broker_max_pending': 4,
                  'broker_max_pending_per_team': 1}
        profiles = []
        dtypes = (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64'))
        for length in (2, 3):
            profiles.append({'profile_id': 'fixed-v1-' + benchmark.canonical_digest(
                {'shape': (1, length), 'dtypes': dtypes})[:16], 'model_id': fingerprint,
                'device': 'cuda:0', 'shape': [1, length], 'input_dtypes': [list(x) for x in dtypes],
                'output_limit_bytes': 8 * 1024 * 1024, 'estimated_incremental_bytes': 1024 ** 3})
        replay = {'public_prompts': ['fixture']}
        run_metadata = {**source, 'external_gpu_activity': 'fixture', 'environment': environment,
                        'workload_replay': replay, 'smoke_only': True,
                        'memory_policy': {**limits, 'profiles': profiles}}
        supervision = SimpleNamespace(run_id='a' * 32, pid=123)
        with tempfile.TemporaryDirectory() as temp:
            owner = Path(temp) / 'owner'
            owner.mkdir()
            result = benchmark.run_protocol(owner / 'measurements', Broker(), jobs,
                input_digest=digest, output_digest=digest, rejection_errors=(Full,), model_spec=model_spec,
                unit_name='input_positions_including_padding', metadata=run_metadata,
                telemetry=Memory(), repetitions=1, protocol_seed=3)
            preflight = {'model_metadata': metadata, 'manifest_jobs': result['manifest']['jobs'],
                         'source': source, 'public_prompts': ['fixture'],
                         'cuda_environment': environment['cuda_environment']}
            setup = {'protocol': 'shared-model-throughput-v1', 'owner_run_id': supervision.run_id,
                     'owner_pid': supervision.pid, 'environment': environment, 'source': source,
                     'smoke_only': True, 'repetitions': 1, 'protocol_seed': 3,
                     'external_gpu_activity': 'fixture', 'workload_replay': replay, 'limits': limits}
            for name, data in (('setup.json', setup), ('admission_profiles.json',
                    {'status': 'provisional_unmeasured_estimates', 'profiles': profiles}),
                    ('cleanup.json', {'status': 'complete', 'broker_state': 'stopped',
                        'memory_lease_released': True, 'owner_attested_clean': True,
                        'cuda_allocator_after_cleanup': {'allocated_bytes': 0, 'reserved_bytes': 0}}),
                    ('run_status.json', {'status': 'complete', 'measurement_status': 'complete',
                                         'cleanup_status': 'complete', 'work_may_still_be_running': False})):
                (owner / name).write_text(json.dumps(data))
            def validate():
                return benchmark._validate_owner_evidence(owner, supervision=supervision,
                    preflight=preflight, gpu_uuid=gpu, repetitions=1, quick=True, protocol_seed=3)
            with patch.object(benchmark, '_validate_preflight'):
                self.assertEqual(validate()['status'], 'complete')
                status_path = owner / 'run_status.json'
                original_status = status_path.read_text()
                status = json.loads(original_status)
                status['work_may_still_be_running'] = True
                status_path.write_text(json.dumps(status))
                with self.assertRaisesRegex(ValueError, 'cleanup did not complete'):
                    validate()
                status_path.write_text(original_status)
                raw = owner / 'measurements' / 'raw_samples.jsonl'
                records = [json.loads(line) for line in raw.read_text().splitlines()]
                records[0].update(state='failed', error='injected', output_digest=None)
                raw.write_text(''.join(json.dumps(row) + '\n' for row in records))
                result['evidence_sha256'] = benchmark._evidence_hashes(owner / 'measurements')
                (owner / 'measurements' / 'results.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, 'unsuccessful'):
                    validate()

class FailureArtifactTests(unittest.TestCase):
    def test_incomplete_window_preserves_attempt_and_stops_before_other_mode(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import json
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Broker:
            calls = 0
            def submit(self, team, request):
                self.calls += 1
                submitted = time.perf_counter_ns()
                if self.calls == 1:
                    ready, started, finished = (time.perf_counter_ns() for _ in range(3))
                    outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                        started_ns=started, finished_ns=finished, result=request,
                        error_type=None, error_message=None)
                    return SimpleNamespace(wait=lambda timeout: outcome)
                class Unresolved:
                    submitted_ns = submitted
                    def wait(self, timeout):
                        raise TimeoutError('not terminal')
                    def cancel(self):
                        return True
                return Unresolved()
        class Memory:
            def before(self):
                return {'free_bytes': 100}
            def after(self):
                raise AssertionError('cannot sample an undrained device')
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        jobs = [benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a')]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'failure'
            broker = Broker()
            with self.assertRaisesRegex(RuntimeError, 'did not drain'):
                benchmark.run_protocol(output, broker, jobs, input_digest=digest, output_digest=digest,
                    rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='tokens',
                    metadata={'external_gpu_activity': 'cpu', 'source_commit': 'fixture'},
                    telemetry=Memory(), repetitions=1)
            self.assertEqual(broker.calls, 2)
            self.assertTrue((output / 'manifest.json').exists())
            raw = [json.loads(line) for line in (output / 'raw_samples.jsonl').read_text().splitlines()]
            self.assertEqual(raw[0]['state'], 'unresolved')
            self.assertEqual(json.loads((output / 'failure.json').read_text())['completed_windows'], 0)
            self.assertFalse((output / 'results.json').exists())

class ComparisonEvidenceTests(unittest.TestCase):
    def test_output_disagreement_marks_run_invalid_without_losing_raw_samples(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Broker:
            calls = 0
            def submit(self, team, request):
                self.calls += 1
                submitted, ready, started, finished = (time.perf_counter_ns() for _ in range(4))
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=str(self.calls),
                    error_type=None, error_message=None)
                return SimpleNamespace(wait=lambda timeout: outcome)
        class Memory:
            def before(self):
                return {}
            def after(self):
                return {}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            result = benchmark.run_protocol(output, Broker(), [job], input_digest=digest,
                output_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake'},
                unit_name='tokens', metadata={'external_gpu_activity': 'cpu', 'source_commit': 'fixture'},
                telemetry=Memory(), repetitions=1)
            self.assertEqual(result['status'], 'invalid_comparison')
            self.assertFalse(result['comparisons'][0]['matching_common_outputs'])
            self.assertEqual(len((output / 'raw_samples.jsonl').read_text().splitlines()), 2)

class BrokerCompatibilityTests(unittest.TestCase):
    def test_synchronization_proxy_preserves_worker_cancellation_contract(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        class Cancelled(Exception):
            pass
        class Worker:
            cancellation_exception = Cancelled
            poisoned = False
        proxy = benchmark.SynchronizedWorker(Worker(), lambda: None)
        self.assertIs(proxy.cancellation_exception, Cancelled)

class InterruptionTests(unittest.TestCase):
    def test_interrupt_requests_cancellation_and_marks_device_state_unknown(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import json
        import tempfile
        import threading
        import time
        from pathlib import Path
        from unittest.mock import patch
        class Full(Exception):
            pass
        submitted = threading.Event()
        class Handle:
            def __init__(self):
                self.submitted_ns = time.perf_counter_ns()
                self.cancelled = threading.Event()
            def wait(self, timeout):
                self.cancelled.wait(5)
                raise TimeoutError('cancelled by harness')
            def cancel(self):
                self.cancelled.set()
                return True
        class Broker:
            def submit(self, team, request):
                self.handle = Handle()
                submitted.set()
                return self.handle
        class Memory:
            def before(self):
                return {}
            def after(self):
                return {}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a')
        broker = Broker()
        def interrupted_join(*args, **kwargs):
            self.assertTrue(submitted.wait(2))
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'interrupted'
            with patch.object(threading.Thread, 'join', interrupted_join):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_protocol(output, broker, [job], input_digest=digest,
                        output_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake'},
                        unit_name='positions', metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                        telemetry=Memory(), repetitions=1)
            self.assertTrue(broker.handle.cancelled.is_set())
            failure = json.loads((output / 'failure.json').read_text())
            self.assertEqual(failure['status'], 'cancelled')
            self.assertTrue(failure['work_may_still_be_running'])

class InputDriftTests(unittest.TestCase):
    def test_protocol_stops_if_full_request_changes_between_windows(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Broker:
            def submit(self, team, request):
                submitted, ready, started, finished = (time.perf_counter_ns() for _ in range(4))
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=request['text'],
                    error_type=None, error_message=None)
                return SimpleNamespace(wait=lambda timeout: outcome)
        request = {'text': 'x', 'policy': 'original'}
        class MutatingMemory:
            def before(self):
                return {}
            def after(self):
                request['policy'] = 'changed'
                return {}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('x'), 'none', request)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                benchmark.run_protocol(output, Broker(), [job],
                    input_digest=lambda value: digest(value['text']), output_digest=digest,
                    rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='positions',
                    metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                    telemetry=MutatingMemory(), repetitions=2)
            self.assertTrue((output / 'failure.json').exists())
            self.assertTrue((output / 'raw_samples.jsonl').exists())


class TensorHashTests(unittest.TestCase):
    def test_canonical_tensor_hash_covers_bfloat_noncontiguous_and_empty(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        try:
            import torch
            import numpy
        except ImportError:
            self.skipTest('torch and numpy are required for CPU tensor checks')
        tensors = [torch.tensor([1., 2.], dtype=torch.bfloat16),
                   torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
                   torch.tensor([], dtype=torch.int64)]
        for value in tensors:
            with self.subTest(dtype=value.dtype, shape=tuple(value.shape)):
                raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                self.assertEqual(benchmark._canonical(value)['sha256'], hashlib.sha256(raw).hexdigest())

class FailureDigestTests(unittest.TestCase):
    def test_failure_evidence_still_published_when_file_hashing_fails(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        class Full(Exception):
            pass
        class Memory:
            def before(self):
                return {}
            def after(self):
                return {}
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, 'a', 'none', 'a')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'failed'
            with patch.object(benchmark, 'run_window', side_effect=RuntimeError('work failed')):
                with patch.object(benchmark, '_evidence_hashes', side_effect=OSError('digest denied')):
                    with self.assertRaisesRegex(RuntimeError, 'work failed'):
                        benchmark.run_protocol(output, object(), [job], input_digest=str,
                            output_digest=str, rejection_errors=(Full,), model_spec={'id': 'fake'},
                            unit_name='positions', metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                            telemetry=Memory(), repetitions=1)
            failure = json.loads((output / 'failure.json').read_text())
            self.assertIn('work failed', failure['message'])
            self.assertIn('digest denied', failure['evidence_hash_error'])

class SharedAbortTests(unittest.TestCase):
    def test_one_unresolved_client_stops_other_team_before_its_next_job(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import threading
        import time
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Handle:
            def __init__(self, request):
                self.request = request
                self.submitted_ns = time.perf_counter_ns()
                self.cancelled = threading.Event()
            def wait(self, timeout):
                if self.request == 'red':
                    raise TimeoutError('red unresolved')
                self.cancelled.wait(1)
                now = time.perf_counter_ns()
                return SimpleNamespace(state='cancelled', submitted_ns=self.submitted_ns,
                    ready_ns=self.submitted_ns, started_ns=self.submitted_ns, finished_ns=now,
                    result=None, error_type=None, error_message=None)
            def cancel(self):
                self.cancelled.set()
                return True
        class Broker:
            def __init__(self):
                self.submitted = []
            def submit(self, team, request):
                self.submitted.append(request)
                return Handle(request)
        jobs = [benchmark.WorkItem('r', 'red', 'w', 1, 0, 'red', 'none', 'red'),
                benchmark.WorkItem('b1', 'blue', 'w', 1, 0, 'blue1', 'none', 'blue1'),
                benchmark.WorkItem('b2', 'blue', 'w', 1, 0, 'blue2', 'none', 'blue2')]
        broker = Broker()
        window = benchmark.run_window(broker, jobs, mode='shared', input_digest=str, output_digest=str,
            rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='positions',
            wait_timeout=.01, cancel_timeout=.01)
        self.assertEqual(window['status'], 'incomplete')
        self.assertNotIn('blue2', broker.submitted)


class CompletedBeforeInterruptTests(unittest.TestCase):
    def test_interruption_preserves_completed_rows_and_marks_active_handles_unknown(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import json
        import tempfile
        import threading
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class Full(Exception):
            pass
        second_started = threading.Event()
        class Handle:
            def __init__(self, request, submitted):
                self.request, self.submitted_ns = request, submitted
                self.cancelled = threading.Event()
            def wait(self, timeout):
                if self.request == 'first':
                    now = time.perf_counter_ns()
                    return SimpleNamespace(state='succeeded', submitted_ns=self.submitted_ns,
                        ready_ns=self.submitted_ns, started_ns=self.submitted_ns, finished_ns=now,
                        result='first', error_type=None, error_message=None)
                self.cancelled.wait(5)
                raise TimeoutError('cancelled')
            def cancel(self):
                self.cancelled.set()
                return True
        class Broker:
            def submit(self, team, request):
                handle = Handle(request, time.perf_counter_ns())
                if request == 'second':
                    second_started.set()
                return handle
        class Memory:
            def before(self): return {}
            def after(self): return {}
        jobs = [benchmark.WorkItem('a', 'red', 'w', 1, 0, 'first', 'none', 'first'),
                benchmark.WorkItem('b', 'red', 'w', 1, 0, 'second', 'none', 'second')]
        def interrupted_join(*args, **kwargs):
            self.assertTrue(second_started.wait(2))
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with patch.object(threading.Thread, 'join', interrupted_join):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_protocol(output, Broker(), jobs, input_digest=str, output_digest=str,
                        rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='positions',
                        metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                        telemetry=Memory(), repetitions=1)
            rows = [json.loads(line) for line in (output / 'warmup_samples.jsonl').read_text().splitlines()]
            self.assertEqual([row['job_id'] for row in rows], ['a'])
            snapshot = json.loads((output / 'interrupted_windows.jsonl').read_text())
            self.assertEqual(snapshot['active_jobs'][0]['job_id'], 'b')
            self.assertTrue(json.loads((output / 'failure.json').read_text())['work_may_still_be_running'])

class ActualBrokerIntegrationTests(unittest.TestCase):
    def test_real_cpu_broker_worker_produce_matching_paired_outputs(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import tempfile
        from pathlib import Path
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required for real broker integration')
        try:
            from looking_glass.broker import Broker, BrokerUnavailable
            from looking_glass.fair_queue import QueueFull
            from looking_glass.resident_worker import ForwardRequest, InvalidRequest, ResidentWorker
        except ImportError:
            self.skipTest('broker and worker must be integrated')
        class Model(torch.nn.Module):
            def forward(self, input_ids):
                values = input_ids.float()
                return {'logits': torch.stack([values, values + 1], dim=-1)}
        class CpuMemory:
            def before(self): return {'device': 'cpu'}
            def after(self): return {'device': 'cpu'}
        worker = ResidentWorker('cpu-test', Model().eval(), device='cpu', max_sequence_length=4)
        proxy = benchmark.SynchronizedWorker(worker, lambda: None)
        jobs = []
        for i in range(4):
            request = ForwardRequest(f'job-{i}', {'input_ids': torch.tensor([[i, i + 1]])}, seed=i)
            jobs.append(benchmark.WorkItem(request.request_id, f'team-{i % 2}', 'two-tokens',
                request.inputs['input_ids'].numel(), i, benchmark.canonical_digest(request), 'none', request))
        with tempfile.TemporaryDirectory() as temp:
            with Broker(proxy, max_pending=4, max_pending_per_team=1) as broker:
                result = benchmark.run_protocol(Path(temp) / 'actual', broker, jobs,
                    input_digest=benchmark.canonical_digest, output_digest=benchmark.canonical_digest,
                    rejection_errors=(QueueFull, BrokerUnavailable, InvalidRequest),
                    model_spec={'id': 'cpu-test'}, unit_name='input_positions_including_padding',
                    metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                    telemetry=CpuMemory(), repetitions=1)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['comparisons'][0]['valid_paired_comparison'])
            self.assertEqual([w['summary']['counts']['succeeded'] for w in result['windows']], [4, 4])
            self.assertEqual([w['summary']['queue_wait_ns_started']['n'] for w in result['windows']], [4, 4])

class ConcreteWorkloadTests(unittest.TestCase):
    def test_work_units_are_derived_from_token_tensor_and_teams_get_same_mix(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        try:
            import torch
            from looking_glass.resident_worker import ForwardRequest
        except ImportError:
            self.skipTest('resident worker and torch are required')
        class Bundle:
            def encode(self, text):
                count = 2 if text == 'short' else 3
                return {'input_ids': torch.tensor([list(range(count))]),
                        'attention_mask': torch.ones((1, count), dtype=torch.int64)}
        jobs = benchmark.make_distilgpt2_workload(Bundle(), prompts=('short', 'long'))
        self.assertEqual(len(jobs), 4)
        self.assertEqual([job.units for job in jobs], [2, 3, 2, 3])
        self.assertEqual([job.team_id for job in jobs], ['team-a', 'team-a', 'team-b', 'team-b'])
        for job in jobs:
            self.assertEqual(job.units, job.request.inputs['input_ids'].numel())
            self.assertEqual(job.input_digest, benchmark.canonical_digest(job.request))
            self.assertEqual(job.capture_policy, 'none')

class ConcreteOutputValidationTests(unittest.TestCase):
    def test_nonfinite_or_wrong_shape_logits_are_not_valid_output(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        try:
            import torch
            from looking_glass.resident_worker import ForwardResult
        except ImportError:
            self.skipTest('resident worker and torch are required')
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                bad = ForwardResult('job', torch.tensor([[[value, 1.0]]], dtype=torch.float32))
                with self.assertRaises(ValueError):
                    benchmark.digest_forward_result(bad, vocab_size=2)
        wrong = ForwardResult('job', torch.ones((1, 2, 3), dtype=torch.float32))
        with self.assertRaises(ValueError):
            benchmark.digest_forward_result(wrong, vocab_size=2)
        correct = ForwardResult('job', torch.ones((1, 2, 2), dtype=torch.float32))
        self.assertEqual(benchmark.digest_forward_result(correct, vocab_size=2), benchmark.canonical_digest(correct))

class OutcomeCorrelationTests(unittest.TestCase):
    def test_mismatched_broker_outcome_cannot_count_as_success(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import time
        from types import SimpleNamespace
        class Full(Exception):
            pass
        class Broker:
            broker_id = 'original-broker'
            def submit(self, team, request):
                submitted = time.perf_counter_ns()
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted,
                    ready_ns=submitted, started_ns=submitted, finished_ns=time.perf_counter_ns(),
                    broker_id='different-broker', result='anything', error_type=None, error_message=None)
                return SimpleNamespace(job_id=5, submitted_ns=submitted, wait=lambda timeout: outcome,
                                       cancel=lambda: True)
        job = benchmark.WorkItem('a', 'red', 'w', 1, 0, 'a', 'none', 'a')
        result = benchmark.run_window(Broker(), [job], mode='serial', input_digest=str, output_digest=str,
            rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='positions')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['summary']['counts']['unresolved'], 1)
        self.assertEqual(result['unattempted_job_ids'], [])

class SyncQuarantineTests(unittest.TestCase):
    def test_proxy_does_not_release_worker_after_uncertain_device_sync(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        class Worker:
            poisoned = False
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
            def execute(self, request, cancel):
                return 'done'
        worker = Worker()
        def fail_sync():
            raise RuntimeError('device drain failed')
        proxy = benchmark.SynchronizedWorker(worker, fail_sync)
        with self.assertRaisesRegex(RuntimeError, 'device drain failed'):
            proxy.execute('input', None)
        with self.assertRaisesRegex(RuntimeError, 'quarantin'):
            proxy.close()
        self.assertFalse(worker.closed)

class DriverExitAndCudaConfigTests(unittest.TestCase):
    def test_invalid_comparison_returns_nonzero_exit_status(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import contextlib
        import io
        from unittest.mock import patch
        args = ['driver', '--output', 'unused', '--cache-dir', 'unused']
        with patch('sys.argv', args), patch.object(benchmark, 'run_distilgpt2_benchmark',
                return_value={'status': 'invalid_comparison', 'comparisons': []}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(), 1)

    def test_cublas_determinism_is_set_before_fresh_cuda_and_rejects_preinitialized_context(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        from types import SimpleNamespace
        fresh = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        initialized = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        environment = {}
        self.assertEqual(benchmark.configure_cuda_determinism(fresh, environment), ':4096:8')
        self.assertEqual(environment['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        with self.assertRaisesRegex(RuntimeError, 'already initialized'):
            benchmark.configure_cuda_determinism(initialized, {})
        with self.assertRaises(ValueError):
            benchmark.configure_cuda_determinism(fresh, {'CUBLAS_WORKSPACE_CONFIG': 'invalid'})

class DriverSetupFailureTests(unittest.TestCase):
    def test_setup_manifest_write_failure_still_records_failure(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        try:
            import torch
            import transformers
            import numpy
            import looking_glass.pretrained
            import looking_glass.broker
        except ImportError:
            self.skipTest('concrete driver dependencies are required')
        original = benchmark._atomic_write_json
        def fail_setup(path, value):
            if path.name == 'setup.json':
                raise OSError('setup write denied')
            return original(path, value)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'driver'
            with patch.object(benchmark, '_atomic_write_json', fail_setup):
                with self.assertRaisesRegex(OSError, 'setup write denied'):
                    benchmark.run_distilgpt2_benchmark(output, cache_dir='unused', device='cpu', quick=True)
            failure = json.loads((output / 'failure.json').read_text())
            self.assertIn('setup write denied', failure['message'])

class TensorSubclassSafetyTests(unittest.TestCase):
    def test_tensor_subclass_is_rejected_before_reading_native_pointer(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        try:
            import torch
        except ImportError:
            self.skipTest('torch required')
        class UntrustedTensor(torch.Tensor):
            def data_ptr(self):
                raise AssertionError('must not dereference subclass pointer')
        value = torch.tensor([1.0]).as_subclass(UntrustedTensor)
        with self.assertRaises(TypeError):
            benchmark.canonical_digest(value)


class InterruptedSubmissionTests(unittest.TestCase):
    def test_blocked_submission_is_recorded_as_unknown_and_late_handle_cancelled(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import json
        import tempfile
        import threading
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class Full(Exception):
            pass
        started, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        class Broker:
            def submit(self, team, request):
                started.set()
                release.wait(2)
                submitted = time.perf_counter_ns()
                def cancel():
                    cancelled.set()
                    return True
                return SimpleNamespace(submitted_ns=submitted, cancel=cancel,
                    wait=lambda timeout: (_ for _ in ()).throw(TimeoutError('cancelled')))
        class Memory:
            def before(self): return {}
            def after(self): return {}
        job = benchmark.WorkItem('blocked', 'red', 'w', 1, 0, 'x', 'none', 'x')
        def interrupted_join(*args, **kwargs):
            self.assertTrue(started.wait(2))
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with patch.object(threading.Thread, 'join', interrupted_join):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_protocol(output, Broker(), [job], input_digest=str, output_digest=str,
                        rejection_errors=(Full,), model_spec={'id': 'fake'}, unit_name='positions',
                        metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                        telemetry=Memory(), repetitions=1)
            snapshot = json.loads((output / 'interrupted_windows.jsonl').read_text())
            self.assertEqual(snapshot['submissions_unknown'][0]['job_id'], 'blocked')
            release.set()
            self.assertTrue(cancelled.wait(2))

class DriverFinalStatusTests(unittest.TestCase):
    def test_cleanup_failure_prevents_operational_success(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        try:
            import torch
            import transformers
            import numpy
            import looking_glass.pretrained
            import looking_glass.broker
        except ImportError:
            self.skipTest('concrete driver dependencies are required')
        class Worker:
            poisoned = False
            cancellation_exception = RuntimeError
        class BadBroker:
            def close(self, **kwargs):
                raise RuntimeError('cleanup failed')
        bundle = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=2)), metadata={})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'driver'
            with patch('looking_glass.pretrained.load_distilgpt2', return_value=bundle), \
                 patch('looking_glass.resident_worker.ResidentWorker', return_value=Worker()), \
                 patch('looking_glass.broker.Broker', return_value=BadBroker()), \
                 patch.object(benchmark, 'make_distilgpt2_workload', return_value=[]), \
                 patch.object(benchmark, 'run_protocol', return_value={'status': 'complete'}):
                with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                    benchmark.run_distilgpt2_benchmark(output, cache_dir='unused', device='cpu', quick=True)
            self.assertEqual(json.loads((output / 'cleanup.json').read_text())['status'], 'failed')
            self.assertEqual(json.loads((output / 'run_status.json').read_text())['status'], 'cleanup_failed')

class GpuGateTests(unittest.TestCase):
    def test_cuda_driver_requires_supervised_owner_before_device_access(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        try:
            import torch
            import transformers
            import numpy
            import looking_glass.pretrained
            import looking_glass.broker
        except ImportError:
            self.skipTest('concrete driver dependencies are required')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'gpu'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch('looking_glass.supervisor.OwnerSession.from_environment', side_effect=RuntimeError('supervised owner required')) as owner:
                with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('must not access CUDA')):
                    with self.assertRaisesRegex(RuntimeError, 'supervised'):
                        benchmark.run_distilgpt2_benchmark(output, cache_dir='unused', device='cuda:0', quick=True)
            owner.assert_called_once()
            self.assertFalse(output.exists())

class AtomicTerminalTests(unittest.TestCase):
    def test_published_result_does_not_gain_conflicting_failure_if_temp_cleanup_fails(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class Full(Exception):
            pass
        class Broker:
            def submit(self, team, request):
                submitted, ready, started, finished = (time.perf_counter_ns() for _ in range(4))
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=request,
                    error_type=None, error_message=None)
                return SimpleNamespace(wait=lambda timeout: outcome)
        class Memory:
            def before(self): return {}
            def after(self): return {}
        original_unlink = Path.unlink
        def injected_unlink(path, *args, **kwargs):
            if path.name.startswith('.results.json.'):
                raise OSError('temporary cleanup denied')
            return original_unlink(path, *args, **kwargs)
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with patch.object(Path, 'unlink', injected_unlink):
                result = benchmark.run_protocol(output, Broker(), [job], input_digest=digest,
                    output_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake'},
                    unit_name='positions', metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                    telemetry=Memory(), repetitions=1)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue((output / 'results.json').exists())
            self.assertFalse((output / 'failure.json').exists())

    def test_original_publication_error_survives_temp_cleanup_failure(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            with patch('looking_glass.throughput_benchmark.os.link', side_effect=OSError('publish denied')), \
                 patch.object(Path, 'unlink', side_effect=OSError('temp cleanup denied')):
                with self.assertRaisesRegex(OSError, 'publish denied') as caught:
                    benchmark._atomic_write_json(path, {'status': 'complete'})
            self.assertIn('temp cleanup denied', str(caught.exception.__cause__))

class TerminalPublicationAmbiguityTests(unittest.TestCase):
    def test_exception_after_result_publication_does_not_publish_failure_too(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        import hashlib
        import tempfile
        import time
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class Full(Exception):
            pass
        class Broker:
            def submit(self, team, request):
                submitted, ready, started, finished = (time.perf_counter_ns() for _ in range(4))
                outcome = SimpleNamespace(state='succeeded', submitted_ns=submitted, ready_ns=ready,
                    started_ns=started, finished_ns=finished, result=request,
                    error_type=None, error_message=None)
                return SimpleNamespace(wait=lambda timeout: outcome)
        class Memory:
            def before(self): return {}
            def after(self): return {}
        original = benchmark._atomic_write_json
        def publish_then_raise(path, value):
            original(path, value)
            if path.name == 'results.json':
                raise OSError('interrupted after publication')
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        job = benchmark.WorkItem('a', 'red', 'short', 1, 0, digest('a'), 'none', 'a')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with patch.object(benchmark, '_atomic_write_json', publish_then_raise):
                with self.assertRaisesRegex(OSError, 'interrupted after publication'):
                    benchmark.run_protocol(output, Broker(), [job], input_digest=digest,
                        output_digest=digest, rejection_errors=(Full,), model_spec={'id': 'fake'},
                        unit_name='positions', metadata={'external_gpu_activity': 'cpu', 'source_commit': 'test'},
                        telemetry=Memory(), repetitions=1)
            self.assertTrue((output / 'results.json').exists())
            self.assertFalse((output / 'failure.json').exists())


class SupervisedIdentityTests(unittest.TestCase):
    def test_owner_environment_excludes_unselected_parent_values(self):
        environment = {'PATH': 'trusted-search-path', 'SYSTEMROOT': 'system-root',
                       'CUDA_LAUNCH_BLOCKING': '1', 'UNRELATED_SECRET': 'private'}
        runtime = benchmark._configure_cuda_environment({})
        child = benchmark._supervised_child_environment(runtime, environment)
        self.assertEqual(child['PATH'], 'trusted-search-path')
        self.assertEqual(child['SYSTEMROOT'], 'system-root')
        self.assertEqual(child['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertNotIn('CUDA_LAUNCH_BLOCKING', child)
        self.assertNotIn('UNRELATED_SECRET', child)
        self.assertEqual(benchmark._configure_cuda_environment(dict(child)), runtime)

    def test_cuda_environment_policy_rejects_overrides_and_unknown_keys(self):
        baseline = benchmark._configure_cuda_environment({})
        self.assertEqual(baseline['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(baseline['CUDA_MODULE_LOADING'], 'LAZY')
        self.assertEqual(baseline['OMP_NUM_THREADS'], '1')
        self.assertEqual(baseline['MKL_NUM_THREADS'], '1')
        for unsafe in ({'CUDA_LAUNCH_BLOCKING': '1'}, {'PYTORCH_CUDA_ALLOC_CONF': 'backend:cudaMallocAsync'},
                       {'NVIDIA_TF32_OVERRIDE': '0'}, {'CUDA_UNKNOWN_OVERRIDE': 'secret'},
                       {'OMP_NUM_THREADS': '8'}, {'OMP_PROC_BIND': 'true'}, {'MKL_DYNAMIC': 'true'},
                       {'KMP_AFFINITY': 'compact'}, {'ONEDNN_MAX_CPU_ISA': 'AVX2'}):
            with self.assertRaises(ValueError):
                benchmark._configure_cuda_environment(unsafe)

    def test_allocator_fraction_must_be_finite_and_match_cap(self):
        from types import SimpleNamespace
        for fraction in (float('nan'), float('inf'), -1.0, 0.5):
            cuda = SimpleNamespace(get_allocator_backend=lambda: 'native',
                set_per_process_memory_fraction=lambda value, device: None,
                get_per_process_memory_fraction=lambda device: fraction)
            with self.assertRaisesRegex(RuntimeError, 'fraction'):
                benchmark._configure_cuda_allocator(SimpleNamespace(cuda=cuda), 'cuda:0')
        cuda = SimpleNamespace(get_allocator_backend=lambda: 'native',
            set_per_process_memory_fraction=lambda value, device: None,
            get_per_process_memory_fraction=lambda device: 0.75)
        self.assertEqual(benchmark._configure_cuda_allocator(SimpleNamespace(cuda=cuda), 'cuda:0'), ('native', 0.75))

    def test_cpu_preflight_masks_gpu_and_restores_visibility(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        seen = []
        class Version(str):
            pass
        bundle = SimpleNamespace(metadata={'id': 'fake', 'torch_version': Version('2.8'),
                                           'transformers_version': Version('4.57')})
        def load(**kwargs):
            seen.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
            return bundle
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-original'}), \
             patch('looking_glass.pretrained.load_distilgpt2', side_effect=load), \
             patch.object(benchmark, 'make_distilgpt2_workload', return_value=['job']), \
             patch.object(benchmark, 'workload_manifest', return_value=({'jobs': []}, 'digest')), \
             patch.object(benchmark, '_source_metadata', return_value={'source_commit': 'test'}):
            result = benchmark._preflight_distilgpt2('unused', quick=True)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-original')
        self.assertEqual(seen, [''])
        self.assertEqual(result['model_metadata'], {'id': 'fake', 'torch_version': '2.8',
                                                     'transformers_version': '4.57'})
        self.assertIs(type(result['model_metadata']['torch_version']), str)
        self.assertIs(type(result['model_metadata']['transformers_version']), str)
        self.assertIs(type(bundle.metadata['torch_version']), Version)

    def test_cpu_bootstrap_status_requires_successful_startup_validation(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(benchmark, '_require_fresh_bytecode_prefix') as check:
            self.assertIsNone(benchmark._checked_bytecode_isolation())
            check.assert_not_called()
        with patch.dict(os.environ, {'LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID': 'fake'}, clear=True), \
             patch.object(benchmark, '_require_fresh_bytecode_prefix') as check:
            self.assertEqual(benchmark._checked_bytecode_isolation(), 'empty_prefix_no_writes')
            check.assert_called_once()
        with patch.dict(os.environ, {'LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID': 'fake'}, clear=True), \
             patch.object(benchmark, '_require_fresh_bytecode_prefix', side_effect=RuntimeError('bad startup')):
            with self.assertRaisesRegex(RuntimeError, 'bad startup'):
                benchmark._checked_bytecode_isolation()

    def test_runtime_uuid_is_verified_under_the_same_lease_before_model_work(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        from types import SimpleNamespace
        from unittest.mock import patch
        events = []
        gpu_uuid = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
        class Session:
            def verify_runtime_binding(self, lease, *, runtime_uuid, device_count, cuda_device_index=0):
                events.append(('verify', lease, runtime_uuid, device_count, cuda_device_index))
                if runtime_uuid != gpu_uuid:
                    raise ValueError('wrong device')
        class Cuda:
            def is_initialized(self):
                events.append('initialized')
                return False
            def is_available(self):
                events.append('available')
                return True
            def get_device_properties(self, index):
                events.append(('properties', index))
                return SimpleNamespace(uuid=gpu_uuid)
            def device_count(self):
                events.append('device_count')
                return 1
        environment = {}
        lease = object()
        with patch.object(benchmark, '_check_owner_before_cuda') as check:
            result = benchmark._initialize_verified_cuda(Session(), lease, SimpleNamespace(cuda=Cuda()), environment)
        check.assert_called_once()
        self.assertEqual(result, gpu_uuid)
        self.assertEqual(events[-1], ('verify', lease, gpu_uuid, 1, 0))
        self.assertEqual(environment['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_runtime_uuid_mismatch_stops_before_a_verified_binding(self):
        if benchmark is None:
            self.fail('throughput benchmark module is missing')
        from types import SimpleNamespace
        from unittest.mock import patch
        class Session:
            def verify_runtime_binding(self, lease, **kwargs):
                raise ValueError('wrong physical GPU')
        cuda = SimpleNamespace(is_initialized=lambda: False, is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(uuid='GPU-01234567-89ab-cdef-0123-456789abcdef'),
            device_count=lambda: 1)
        with patch.object(benchmark, '_check_owner_before_cuda'):
            with self.assertRaisesRegex(ValueError, 'wrong physical GPU'):
                benchmark._initialize_verified_cuda(Session(), object(), SimpleNamespace(cuda=cuda), {})

    def test_pytorch_cuuuid_is_strictly_normalized(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class CudaUuid:
            def __str__(self): return '01234567-89ab-cdef-0123-456789abcdef'
        calls = []
        class Session:
            def verify_runtime_binding(self, lease, **kwargs): calls.append(kwargs['runtime_uuid'])
        cuda = SimpleNamespace(is_initialized=lambda: False, is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(uuid=CudaUuid()), device_count=lambda: 1)
        torch_fake = SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_CUuuid=CudaUuid))
        with patch.object(benchmark, '_check_owner_before_cuda'):
            result = benchmark._initialize_verified_cuda(Session(), object(), torch_fake, {})
        self.assertEqual(result, 'GPU-01234567-89ab-cdef-0123-456789abcdef')
        self.assertEqual(calls, [result])

    def test_unknown_runtime_uuid_type_is_rejected_without_binding(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Session:
            def verify_runtime_binding(self, *args, **kwargs):
                raise AssertionError('unknown UUID type must be rejected')
        cuda = SimpleNamespace(is_initialized=lambda: False, is_available=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(uuid=object()), device_count=lambda: 1)
        with patch.object(benchmark, '_check_owner_before_cuda'):
            with self.assertRaisesRegex(ValueError, 'UUID type'):
                benchmark._initialize_verified_cuda(Session(), object(), SimpleNamespace(cuda=cuda), {})

    def test_lost_owner_marker_prevents_all_cuda_callbacks(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        touched = []
        cuda = SimpleNamespace(is_initialized=lambda: touched.append('query'))
        with patch.object(benchmark, '_check_owner_before_cuda', side_effect=RuntimeError('marker missing')):
            with self.assertRaisesRegex(RuntimeError, 'marker missing'):
                benchmark._initialize_verified_cuda(object(), object(), SimpleNamespace(cuda=cuda), {})
        self.assertEqual(touched, [])

    def test_clean_attestation_requires_verified_identity_drain_and_empty_budget(self):
        calls = []
        class Session:
            def attest_clean(self, lease):
                calls.append(lease)
        class Budget:
            def __init__(self, active): self.active = active
            def active_leases(self): return self.active
        lease = object()
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            benchmark._attest_owner(Session(), lease, runtime_verified=False, budget=Budget(()),
                                    synchronize=lambda: calls.append('sync'))
        with self.assertRaisesRegex(RuntimeError, 'leases'):
            benchmark._attest_owner(Session(), lease, runtime_verified=True, budget=Budget((object(),)),
                                    synchronize=lambda: calls.append('sync'))
        self.assertEqual(calls, [])
        benchmark._attest_owner(Session(), lease, runtime_verified=True, budget=Budget(()),
                                synchronize=lambda: calls.append('sync'))
        self.assertEqual(calls, ['sync', lease])

    def test_failed_final_synchronization_does_not_attest(self):
        calls = []
        class Session:
            def attest_clean(self, lease): calls.append(lease)
        def fail(): raise RuntimeError('drain failed')
        with self.assertRaisesRegex(RuntimeError, 'drain failed'):
            benchmark._attest_owner(Session(), object(), runtime_verified=True, budget=None, synchronize=fail)
        self.assertEqual(calls, [])

    def test_cuda_allocator_must_be_empty_before_attestation(self):
        from types import SimpleNamespace
        for allocated, reserved in ((1, 1), (0, 4096), (-1, 0), (None, 0)):
            cuda = SimpleNamespace(synchronize=lambda device: None, empty_cache=lambda: None,
                memory_allocated=lambda device: allocated, memory_reserved=lambda device: reserved)
            with self.assertRaisesRegex(RuntimeError, 'allocations remain'):
                benchmark._clear_cuda_allocator(SimpleNamespace(cuda=cuda), 'cuda:0')
        cuda = SimpleNamespace(synchronize=lambda device: None, empty_cache=lambda: None,
            memory_allocated=lambda device: 0, memory_reserved=lambda device: 0)
        self.assertEqual(benchmark._clear_cuda_allocator(SimpleNamespace(cuda=cuda), 'cuda:0'),
                         {'allocated_bytes': 0, 'reserved_bytes': 0})

    def test_fixed_workload_profiles_bind_actual_inputs_and_deduplicate_signatures(self):
        import torch
        from looking_glass.resident_worker import ForwardRequest
        jobs = []
        for index, length in enumerate((3, 3, 5)):
            request = ForwardRequest(str(index), {'input_ids': torch.ones((1, length), dtype=torch.int64),
                'attention_mask': torch.ones((1, length), dtype=torch.int64)}, seed=index)
            digest = benchmark.canonical_digest(request)
            jobs.append(benchmark.WorkItem(str(index), 'team', 'short', length, index,
                                           digest, 'none', request))
        fingerprint = 'a' * 64
        profiles = benchmark._fixed_workload_profiles(jobs, model_fingerprint=fingerprint,
            device='cuda:0', output_limit_bytes=1024, estimated_incremental_bytes=4096)
        self.assertEqual(len(profiles), 2)
        self.assertEqual({profile.shape for profile in profiles}, {(1, 3), (1, 5)})
        for profile in profiles:
            self.assertEqual(profile.model_id, fingerprint)
            self.assertEqual(profile.input_dtypes,
                (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')))
            self.assertEqual(profile.estimated_incremental_bytes, 4096)

    def test_supervised_driver_orders_binding_admission_drain_and_attestation(self):
        from contextlib import ExitStack
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from looking_glass.resident_worker import ForwardRequest
        events = []
        request = ForwardRequest('job', {'input_ids': torch.ones((1, 3), dtype=torch.int64)}, seed=0)
        jobs = [benchmark.WorkItem('job', 'team', 'short', 3, 0,
                                   benchmark.canonical_digest(request), 'none', request)]
        class Session:
            run_id = 'a' * 32
            def attest_clean(self, lease): events.append(('attest', lease))
        device_lease = object()
        class Model:
            config = SimpleNamespace(vocab_size=2)
            def to(self, device): events.append(('model_to', str(device)))
        bundle = SimpleNamespace(model=Model(), metadata={'weights_sha256': 'b' * 64})
        class Budget:
            def __init__(self, *args, **kwargs): self.leases = []
            def acquire(self, *args, **kwargs):
                budget = self
                class Lease:
                    released = False
                    def release(self):
                        events.append('release_memory')
                        self.released = True
                        budget.leases.remove(self)
                lease = Lease()
                self.leases.append(lease)
                return lease
            def active_leases(self): return tuple(self.leases)
        class Worker:
            poisoned = quarantined = False
            cancellation_exception = RuntimeError
            def __init__(self, *args, **kwargs): events.append('resident')
            def close(self): events.append('worker_close')
        class Admission:
            quarantined = False
            def __init__(self, *args, **kwargs): events.append('admission')
        class Budgeted:
            poisoned = quarantined = False
            cancellation_exception = RuntimeError
            def __init__(self, worker, admission, profiles):
                self.worker = worker
                events.append(('budgeted', len(profiles)))
            def close(self): self.worker.close()
        class Broker:
            def __init__(self, worker, **kwargs): self.worker = worker
            def close(self, **kwargs):
                events.append('broker_drain')
                self.worker.close()
                return True
        def claim():
            events.append('claim')
            return Session(), device_lease
        def bind(*args):
            events.append('bind')
            return 'GPU-01234567-89ab-cdef-0123-456789abcdef'
        def sync(*args): events.append('sync')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            patches = [patch.object(benchmark, '_require_fresh_bytecode_prefix'),
                patch.object(benchmark, '_configure_cuda_environment', return_value={}),
                patch.object(benchmark, '_claim_supervised_owner', side_effect=claim),
                patch.object(benchmark, '_initialize_verified_cuda', side_effect=bind),
                patch('looking_glass.pretrained.load_distilgpt2', return_value=bundle),
                patch('looking_glass.resident_worker.ResidentWorker', Worker),
                patch('looking_glass.budgeted_worker.DeviceAdmission', Admission),
                patch('looking_glass.budgeted_worker.BudgetedWorker', Budgeted),
                patch('looking_glass.memory_budget.MemoryBudget', Budget),
                patch('looking_glass.broker.Broker', Broker),
                patch.object(benchmark, 'make_distilgpt2_workload', return_value=jobs),
                patch.object(benchmark, 'run_protocol', return_value={'status': 'complete'}),
                patch.object(torch.cuda, 'synchronize', side_effect=sync),
                patch.object(torch.cuda, 'get_allocator_backend', return_value='native'),
                patch.object(torch.cuda, 'set_per_process_memory_fraction'),
                patch.object(torch.cuda, 'get_per_process_memory_fraction', return_value=0.75),
                patch.object(torch, 'set_num_interop_threads'),
                patch.object(torch, 'get_num_interop_threads', return_value=1),
                patch.object(torch.cuda, 'empty_cache', side_effect=lambda: events.append('empty_cache')),
                patch.object(torch.cuda, 'memory_allocated', return_value=0),
                patch.object(torch.cuda, 'memory_reserved', return_value=0),
                patch.object(torch.cuda, 'get_device_name', return_value='fake GPU'),
                patch.object(torch.cuda, 'get_device_capability', return_value=(7, 5))]
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                result = benchmark.run_distilgpt2_benchmark(output, cache_dir='unused',
                                                          device='cuda:0', quick=True)
            self.assertEqual(result['status'], 'complete')
            self.assertLess(events.index('claim'), events.index('bind'))
            self.assertLess(events.index('bind'), events.index(('model_to', 'cuda:0')))
            self.assertLess(events.index('admission'), events.index(('budgeted', 1)))
            self.assertLess(events.index('broker_drain'), events.index('release_memory'))
            self.assertLess(events.index('release_memory'), events.index(('attest', device_lease)))
            self.assertEqual(events[-1], ('attest', device_lease))

    def test_failed_runtime_binding_never_starts_model_or_attests(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        calls = []
        session = SimpleNamespace(run_id='a' * 32, attest_clean=lambda lease: calls.append('attest'))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch.object(benchmark, '_claim_supervised_owner', return_value=(session, object())), \
                 patch.object(benchmark, '_initialize_verified_cuda', side_effect=ValueError('UUID mismatch')), \
                 patch('looking_glass.pretrained.load_distilgpt2', side_effect=AssertionError('model load')), \
                 patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('wrong GPU touched')):
                with self.assertRaisesRegex(ValueError, 'UUID mismatch'):
                    benchmark.run_distilgpt2_benchmark(output, cache_dir='unused', device='cuda:0', quick=True)
            self.assertEqual(calls, [])
            self.assertEqual(json.loads((output / 'cleanup.json').read_text())['status'], 'failed')

    def test_supervisor_controller_requires_clean_exit_and_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        identity = SimpleNamespace(lock_path=Path('lock'), marker_path=Path('marker'),
                                   gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef')
        supervision = SimpleNamespace(run_id='a' * 32, pid=123, exit_code=0,
                                      clean=True, timed_out=False, reason='clean')
        preflight = {'model_metadata': {'weights_sha256': 'b' * 64}, 'manifest_jobs': [],
                     'source': {'source_commit': 'x', 'source_sha256': {}}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'controller'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch.object(benchmark, '_preflight_distilgpt2', return_value=preflight), \
                 patch.object(benchmark, '_validate_preflight'), \
                 patch.object(benchmark, '_physical_gpu_inventory', return_value=(identity.gpu_uuid,)), \
                 patch('looking_glass.device_identity.resolve_device_identity', return_value=identity), \
                 patch.object(benchmark, '_direct_python_module', return_value=['direct-python', '--owner-child']), \
                 patch('looking_glass.supervisor.run_supervised', return_value=supervision) as run, \
                 patch.object(benchmark, '_validate_owner_evidence', return_value={'status': 'complete'}) as validate:
                result = benchmark.run_supervised_distilgpt2_benchmark(output, cache_dir='cache',
                    device_root=temp, gpu_uuid=identity.gpu_uuid, quick=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(json.loads((output / 'controller_status.json').read_text())['status'], 'complete')
            self.assertEqual(run.call_args.kwargs['device_lease_path'], identity.lock_path)
            self.assertEqual(run.call_args.kwargs['identity'], identity)
            self.assertIs(run.call_args.kwargs['inherit_environment'], False)
            self.assertNotIn('UNRELATED_SECRET', run.call_args.kwargs['env'])
            self.assertIn('--owner-child', run.call_args.args[0])
            validate.assert_called_once()

    def test_supervisor_controller_refuses_unclean_or_failed_owner(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        identity = SimpleNamespace(lock_path=Path('lock'), marker_path=Path('marker'),
                                   gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef')
        preflight = {'model_metadata': {}, 'manifest_jobs': [], 'source': {}}
        for clean, exit_code, timed_out in ((False, 0, False), (True, 1, False), (True, 0, True)):
            supervision = SimpleNamespace(run_id='a' * 32, pid=123, exit_code=exit_code,
                                          clean=clean, timed_out=timed_out, reason='test')
            with tempfile.TemporaryDirectory() as temp:
                with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                     patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                     patch.object(benchmark, '_preflight_distilgpt2', return_value=preflight), \
                     patch.object(benchmark, '_validate_preflight'), \
                     patch.object(benchmark, '_physical_gpu_inventory', return_value=(identity.gpu_uuid,)), \
                     patch('looking_glass.device_identity.resolve_device_identity', return_value=identity), \
                     patch.object(benchmark, '_direct_python_module', return_value=['direct-python', '--owner-child']), \
                     patch('looking_glass.supervisor.run_supervised', return_value=supervision), \
                     patch.object(benchmark, '_validate_owner_evidence') as validate:
                    result = benchmark.run_supervised_distilgpt2_benchmark(Path(temp) / 'run',
                        cache_dir='cache', device_root=temp, gpu_uuid=identity.gpu_uuid, quick=True)
                self.assertEqual(result['status'], 'failed')
                validate.assert_not_called()

    def test_preflight_interruption_has_parent_start_and_failure_records(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch.object(benchmark, '_preflight_distilgpt2', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_supervised_distilgpt2_benchmark(output, cache_dir='cache',
                        device_root=temp, gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef', quick=True)
            self.assertEqual(json.loads((output / 'controller_manifest.json').read_text())['status'], 'initializing')
            failure = json.loads((output / 'controller_failure.json').read_text())
            self.assertEqual(failure['status'], 'cancelled')
            self.assertFalse(failure['device_state_may_be_unsafe'])

    def test_validated_preflight_is_preserved_when_inventory_fails(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch.object(benchmark, '_preflight_distilgpt2', return_value={'identity': 'test'}), \
                 patch.object(benchmark, '_validate_preflight'), \
                 patch.object(benchmark, '_physical_gpu_inventory', side_effect=OSError('inventory failed')):
                with self.assertRaisesRegex(OSError, 'inventory failed'):
                    benchmark.run_supervised_distilgpt2_benchmark(output, cache_dir='cache',
                        device_root=temp, gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef', quick=True)
            preflight = output / 'preflight.json'
            self.assertEqual(json.loads(preflight.read_text()), {'identity': 'test', 'cuda_environment': {}})
            failure = json.loads((output / 'controller_failure.json').read_text())
            self.assertFalse(failure['device_state_may_be_unsafe'])
            self.assertEqual(failure['controller_evidence_sha256']['preflight.json'],
                             hashlib.sha256(preflight.read_bytes()).hexdigest())
            self.assertIn('controller_manifest.json', failure['controller_evidence_sha256'])

    def test_fresh_bytecode_prefix_requires_direct_bootstrap_and_empty_directory(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / 'cache'
            prefix.mkdir()
            fake_sys = SimpleNamespace(pycache_prefix=str(prefix),
                _xoptions={'pycache_prefix': str(prefix)},
                flags=SimpleNamespace(dont_write_bytecode=1, no_site=1, ignore_environment=1, isolated=1),
                dont_write_bytecode=True)
            with patch.dict(os.environ, {'LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID': str(os.getpid()),
                                         'LOOKING_GLASS_THROUGHPUT_SITE_PATHS': json.dumps([temp])}):
                self.assertEqual(benchmark._require_fresh_bytecode_prefix(fake_sys), prefix.resolve())
                (prefix / 'stale.pyc').write_bytes(b'bad')
                with self.assertRaisesRegex(RuntimeError, 'must be empty'):
                    benchmark._require_fresh_bytecode_prefix(fake_sys)
                (prefix / 'stale.pyc').unlink()
                fake_sys.flags.dont_write_bytecode = 0
                with self.assertRaisesRegex(RuntimeError, 'source bootstrap'):
                    benchmark._require_fresh_bytecode_prefix(fake_sys)
                fake_sys.flags.dont_write_bytecode = 1
                fake_sys.flags.no_site = 0
                with self.assertRaisesRegex(RuntimeError, 'source bootstrap'):
                    benchmark._require_fresh_bytecode_prefix(fake_sys)
                fake_sys.flags.no_site = 1
            fake_sys.flags.dont_write_bytecode = 1
            with patch.dict(os.environ, {'LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID': 'different'}):
                with self.assertRaisesRegex(RuntimeError, 'source bootstrap'):
                    benchmark._require_fresh_bytecode_prefix(fake_sys)

    def test_direct_source_bootstrap_ignores_stale_timestamp_valid_bytecode(self):
        import json
        import os
        import py_compile
        import site
        import subprocess
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'cached_subject.py'
            source.write_text('VALUE = "old"\n', encoding='utf-8')
            original = source.stat()
            py_compile.compile(str(source), doraise=True)
            source.write_text('VALUE = "new"\n', encoding='utf-8')
            os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
            (root / 'cache_probe.py').write_text('import importlib.util, json, sys\n'
                'from pathlib import Path\n'
                'spec = importlib.util.spec_from_file_location("cached_subject", Path(__file__).with_name("cached_subject.py"))\n'
                'cached_subject = importlib.util.module_from_spec(spec)\n'
                'spec.loader.exec_module(cached_subject)\n'
                'with open(sys.argv[1], "w", encoding="utf-8") as f:\n'
                '    json.dump({"value": cached_subject.VALUE}, f)\n', encoding='utf-8')
            prefix, output = root / 'prefix', root / 'output.json'
            prefix.mkdir()
            command = benchmark._direct_python_module('cache_probe', [str(output)],
                project=root, pycache_prefix=prefix, site_paths=site.getsitepackages())
            subprocess.run(command, cwd=root, check=True, timeout=30)
            self.assertEqual(json.loads(output.read_text())['value'], 'new')
            self.assertEqual(list(prefix.iterdir()), [])

    def test_direct_source_bootstrap_ignores_environment_sitecustomize(self):
        import json
        import os
        import site
        import subprocess
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / 'sitecustomize-ran'
            (root / 'sitecustomize.py').write_text(
                'from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("ran")\n', encoding='utf-8')
            output = root / 'output.json'
            (root / 'environment_probe.py').write_text('import json, sys\n'
                'with open(sys.argv[1], "w", encoding="utf-8") as f:\n'
                '    json.dump({"no_site": sys.flags.no_site, "ignore": sys.flags.ignore_environment}, f)\n',
                encoding='utf-8')
            prefix = root / 'prefix'
            prefix.mkdir()
            command = benchmark._direct_python_module('environment_probe', [str(output)],
                project=root, pycache_prefix=prefix, site_paths=site.getsitepackages())
            child_env = os.environ.copy()
            child_env['PYTHONPATH'] = str(root)
            subprocess.run(command, cwd=root, env=child_env, check=True, timeout=30)
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads(output.read_text()), {'no_site': 1, 'ignore': 1})

    def test_isolated_bootstrap_ignores_sourceless_script_directory_shadow(self):
        import json
        import py_compile
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script_dir = root / 'scripts'
            script_dir.mkdir()
            source_bootstrap = Path(benchmark.__file__).resolve().parent.parent / 'scripts' / 'throughput_bootstrap.py'
            bootstrap = script_dir / 'throughput_bootstrap.py'
            shutil.copyfile(source_bootstrap, bootstrap)
            marker = root / 'shadow-ran'
            shadow = script_dir / 'json.py'
            shadow.write_text('open(' + repr(str(marker)) + ', "w").write("ran")\n', encoding='utf-8')
            py_compile.compile(str(shadow), cfile=str(script_dir / 'json.pyc'), doraise=True)
            shadow.unlink()
            (root / 'probe.py').write_text('VALUE = "source"\n', encoding='utf-8')
            prefix = root / 'prefix'
            prefix.mkdir()
            spec = json.dumps({'module': 'probe', 'project': str(root), 'site_paths': [str(root)]})
            command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}',
                       str(bootstrap), '--bootstrap-internal', spec]
            subprocess.run(command, cwd=root, check=True, timeout=30)
            self.assertFalse(marker.exists())
            self.assertEqual(list(prefix.iterdir()), [])

    def test_project_bootstrap_ignores_root_shadow_and_rejects_sourceless_package_module(self):
        import json
        import py_compile
        import shutil
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts, package, site = root / 'scripts', root / 'looking_glass', root / 'site'
            scripts.mkdir()
            package.mkdir()
            site.mkdir()
            bootstrap = scripts / 'throughput_bootstrap.py'
            shutil.copyfile(Path(benchmark.__file__).resolve().parent.parent / 'scripts' / 'throughput_bootstrap.py', bootstrap)
            marker = root / 'root-shadow-ran'
            shadow = root / 'fractions.py'
            shadow.write_text('open(' + repr(str(marker)) + ', "w").write("ran")\n', encoding='utf-8')
            py_compile.compile(str(shadow), cfile=str(root / 'fractions.pyc'), doraise=True)
            shadow.unlink()
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'throughput_benchmark.py').write_text('import statistics\n', encoding='utf-8')
            prefix = root / 'prefix'
            prefix.mkdir()
            spec = json.dumps({'module': 'looking_glass.throughput_benchmark',
                               'project': str(root), 'site_paths': [str(site)]})
            command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}',
                       str(bootstrap), '--bootstrap-internal', spec]
            subprocess.run(command, cwd=root, check=True, timeout=30)
            self.assertFalse(marker.exists())
            bad_source = package / 'broker.py'
            bad_source.write_text('open(' + repr(str(marker)) + ', "w").write("ran")\n', encoding='utf-8')
            py_compile.compile(str(bad_source), cfile=str(package / 'broker.pyc'), doraise=True)
            bad_source.unlink()
            (package / '__init__.py').write_text('from . import broker\n', encoding='utf-8')
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('project module source is unavailable', result.stderr)
            self.assertFalse(marker.exists())

    def test_repetition_cap_is_checked_before_preflight(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with patch.object(benchmark, '_preflight_distilgpt2') as preflight:
                with self.assertRaisesRegex(ValueError, '1 through 10'):
                    benchmark.run_supervised_distilgpt2_benchmark(output, cache_dir='unused',
                        device_root=temp, gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef', repetitions=11)
            preflight.assert_not_called()
            self.assertFalse(output.exists())

    def test_preflight_requires_full_clean_identity_and_fixed_jobs(self):
        import copy
        from unittest.mock import patch
        from looking_glass.pretrained import ASSETS, MODEL_ID, REVISION
        prompts = benchmark.DISTILGPT2_PROMPTS[:2]
        metadata = {'model_id': MODEL_ID, 'model_revision': REVISION,
            'tokenizer_id': MODEL_ID, 'tokenizer_revision': REVISION,
            'model_class': 'GPT2LMHeadModel', 'tokenizer_class': 'GPT2TokenizerFast',
            'parameter_count': 81912576, 'torch_version': 'test', 'transformers_version': 'test',
            'asset_sha256': {name: digest for name, (_, digest) in ASSETS.items()},
            'weights_sha256': ASSETS['model.safetensors'][1], 'device': 'cpu', 'dtype': 'float32',
            'attention_implementation': 'eager', 'use_cache': False}
        jobs = [{'job_id': f'{team}-{index}', 'team_id': team, 'seed': 1000 + index,
                 'capture_policy': 'none', 'units': 5, 'workload_class': '5-positions',
                 'input_digest': 'c' * 64, 'request_sha256': 'c' * 64}
                for team in ('team-a', 'team-b') for index in range(2)]
        preflight = {'model_metadata': metadata, 'manifest_jobs': jobs, 'public_prompts': list(prompts),
            'cuda_environment': benchmark._configure_cuda_environment({}),
            'source': {'source_commit': 'a' * 40, 'source_dirty': False,
                       'source_sha256': {name: 'b' * 64 for name in benchmark._SOURCE_FILES}}}
        with patch.object(benchmark, '_verify_source_commit') as verify:
            benchmark._validate_preflight(preflight, quick=True)
        verify.assert_called_once_with(preflight['source'])
        invalid = copy.deepcopy(preflight)
        invalid['source']['source_dirty'] = True
        with self.assertRaisesRegex(ValueError, 'clean source'):
            benchmark._validate_preflight(invalid, quick=True)
        invalid = copy.deepcopy(preflight)
        invalid['model_metadata']['tokenizer_revision'] = 'different'
        with self.assertRaisesRegex(ValueError, 'pinned model'):
            benchmark._validate_preflight(invalid, quick=True)
        invalid = copy.deepcopy(preflight)
        invalid['manifest_jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'workload size'):
            benchmark._validate_preflight(invalid, quick=True)

    def test_source_hashes_must_match_git_commit_blobs(self):
        import hashlib
        import subprocess
        from pathlib import Path
        root = Path(benchmark.__file__).resolve().parent.parent
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        hashes = {}
        for name in benchmark._SOURCE_FILES:
            data = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=root,
                                  check=True, capture_output=True).stdout
            hashes[name] = hashlib.sha256(data).hexdigest()
        source = {'source_commit': commit, 'source_sha256': hashes}
        benchmark._verify_source_commit(source, root=root)
        hashes[benchmark._SOURCE_FILES[0]] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'differs from recorded commit'):
            benchmark._verify_source_commit(source, root=root)

    def test_interrupted_published_controller_status_has_no_conflicting_terminal(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        identity = SimpleNamespace(lock_path=Path('lock'), marker_path=Path('marker'),
                                   gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef')
        supervision = SimpleNamespace(run_id='a' * 32, pid=123, exit_code=0,
                                      clean=True, timed_out=False, reason='clean')
        original = benchmark._atomic_write_json
        def publish_then_interrupt(path, value):
            original(path, value)
            if path.name == 'controller_status.json':
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            with patch.object(benchmark, '_require_fresh_bytecode_prefix'), \
                 patch.object(benchmark, '_configure_cuda_environment', return_value={}), \
                 patch.object(benchmark, '_preflight_distilgpt2', return_value={}), \
                 patch.object(benchmark, '_validate_preflight'), \
                 patch.object(benchmark, '_physical_gpu_inventory', return_value=(identity.gpu_uuid,)), \
                 patch('looking_glass.device_identity.resolve_device_identity', return_value=identity), \
                 patch.object(benchmark, '_direct_python_module', return_value=['direct-python', '--owner-child']), \
                 patch('looking_glass.supervisor.run_supervised', return_value=supervision), \
                 patch.object(benchmark, '_validate_owner_evidence', return_value={'status': 'complete'}), \
                 patch.object(benchmark, '_atomic_write_json', side_effect=publish_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_supervised_distilgpt2_benchmark(output, cache_dir='cache',
                        device_root=temp, gpu_uuid=identity.gpu_uuid, quick=True)
            self.assertTrue((output / 'controller_status.json').exists())
            self.assertFalse((output / 'controller_failure.json').exists())

    def test_direct_python_bootstrap_is_the_supervisors_real_child(self):
        import json
        import os
        import tempfile
        import site
        from pathlib import Path
        from unittest.mock import patch
        from looking_glass.supervisor import run_supervised
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = root / 'owner_probe.py'
            output = root / 'probe.json'
            probe.write_text('import json, os, sys, torch\n'
                'from looking_glass.supervisor import OwnerSession\n'
                'from looking_glass.device_lease import DeviceLease\n'
                'session = OwnerSession.from_environment()\n'
                'lease = DeviceLease.acquire(session.device_lease_path, timeout=5)\n'
                'session.check(lease)\n'
                'with open(sys.argv[1], "w", encoding="utf-8") as f:\n'
                '    json.dump({"pid": os.getpid(), "parent": os.getppid(), '
                '"torch": torch.__version__, "cuda_initialized": torch.cuda.is_initialized(), '
                '"unexpected_inherited": "THROUGHPUT_TEST_SECRET" in os.environ}, f)\n'
                'session.attest_clean(lease)\n', encoding='utf-8')
            prefix = root / 'owner-pycache'
            prefix.mkdir()
            command = benchmark._direct_python_module('owner_probe', [str(output)], project=root,
                pycache_prefix=prefix, site_paths=site.getsitepackages())
            with patch.dict(os.environ, {'THROUGHPUT_TEST_SECRET': 'must-not-be-inherited'}):
                child_env = benchmark._supervised_child_environment(
                    benchmark._configure_cuda_environment({}), os.environ)
                result = run_supervised(command, device_lease_path=root / 'device.lock',
                    marker_path=root / 'device.unsafe.json', timeout=30,
                    cwd=str(Path(benchmark.__file__).resolve().parent.parent),
                    env=child_env, inherit_environment=False)
            self.assertTrue(result.clean)
            self.assertEqual(result.exit_code, 0)
            evidence = json.loads(output.read_text())
            import torch
            self.assertEqual(evidence['pid'], result.pid)
            self.assertEqual(evidence['parent'], os.getpid())
            self.assertEqual(evidence['torch'], torch.__version__)
            self.assertFalse(evidence['cuda_initialized'])
            self.assertFalse(evidence['unexpected_inherited'])


if __name__ == '__main__':
    unittest.main()
