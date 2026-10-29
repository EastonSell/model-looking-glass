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
                return {'free_bytes': 100}
            def after(self):
                return {'free_bytes': 80, 'peak_allocated_bytes': 25}
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
            self.assertEqual(result['windows'][0]['memory']['after']['peak_allocated_bytes'], 25)
            self.assertEqual(len((output / 'raw_samples.jsonl').read_text().splitlines()), 8)
            self.assertEqual(len((output / 'warmup_samples.jsonl').read_text().splitlines()), 8)
            self.assertEqual(json.loads((output / 'results.json').read_text())['status'], 'complete')
            self.assertEqual(result['evidence_sha256']['raw_samples.jsonl'],
                hashlib.sha256((output / 'raw_samples.jsonl').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                benchmark.run_protocol(output, Broker(), jobs, input_digest=digest, output_digest=digest,
                    rejection_errors=(Full,), model_spec={'id': 'fake-cpu'}, unit_name='tokens',
                    metadata={'external_gpu_activity': 'none; cpu fake', 'source_commit': 'test-fixture'}, telemetry=Memory())

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
    def test_cuda_driver_fails_closed_before_device_access_until_supervision_integrates(self):
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
            with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('must not access CUDA')):
                with self.assertRaisesRegex(RuntimeError, 'supervised'):
                    benchmark.run_distilgpt2_benchmark(output, cache_dir='unused', device='cuda:0', quick=True)
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


if __name__ == '__main__':
    unittest.main()
