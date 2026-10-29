import importlib
import importlib.util
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn


class TinyTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 7)

    def forward(self, input_ids, attention_mask):
        values = self.embedding(input_ids)
        return self.projection(values * attention_mask.unsqueeze(-1))


class MicrobatchTests(unittest.TestCase):
    @staticmethod
    def fixture():
        m = importlib.import_module('looking_glass.microbatch')
        contract = m.ForwardContract('model-v1', 'checkpoint-sha', 'adapter-v1', 'cpu', torch.float32,
                                     runtime_id='torch-eager-fixture', seed=17, deterministic=True, batch_separable=True,
                                     mode='eval', rng_policy='none', cache_policy='disabled',
                                     mutable_state_policy='none', tracing=False, intervention=False)
        profile = m.BatchProfile(
            profile_id='tiny-v1', certification_id='cpu-test-cert', contract=contract,
            input_schema=(('attention_mask', torch.int64), ('input_ids', torch.int64)),
            input_shape=(1, 3), output_dtype=torch.float32, output_width=7,
            serial_atol=1e-7, serial_rtol=1e-6,
            approved_rows=(1, 2), max_requests=2, max_per_owner=1,
            max_batch_rows=2, max_token_positions=6,
            max_logical_input_bytes=96, max_logical_output_bytes=168, max_logical_total_bytes=264)
        jobs = [m.MicrobatchRequest(f'job-{i}', f'team-{i}', f'req-{i}', contract,
                    {'input_ids': torch.tensor([[i, i + 1, i + 2]], dtype=torch.int64),
                     'attention_mask': torch.ones((1, 3), dtype=torch.int64)}) for i in (1, 2)]
        return m, profile, jobs

    def test_eval_batch_matches_serial_and_preserves_ownership(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.microbatch'),
                             'microbatch component missing')
        module = importlib.import_module('looking_glass.microbatch')
        self.assertTrue(hasattr(module, 'Microbatch'), 'microbatch component missing')
        contract = module.ForwardContract('model-v1', 'checkpoint-sha', 'adapter-v1',
                                          'cpu', torch.float32, runtime_id='torch-eager-fixture',
                                          seed=17, deterministic=True,
                                          batch_separable=True, mode='eval', rng_policy='none',
                                          cache_policy='disabled', mutable_state_policy='none',
                                          tracing=False, intervention=False)
        profile = module.BatchProfile(
            profile_id='tiny-v1', certification_id='cpu-test-cert', contract=contract,
            input_schema=(('attention_mask', torch.int64), ('input_ids', torch.int64)),
            input_shape=(1, 3), output_dtype=torch.float32, output_width=7,
            serial_atol=1e-7, serial_rtol=1e-6,
            approved_rows=(1, 2), max_requests=2, max_per_owner=1,
            max_batch_rows=2, max_token_positions=6,
            max_logical_input_bytes=96, max_logical_output_bytes=168, max_logical_total_bytes=264)
        ids_a = torch.tensor([[1, 2, 3]])
        ids_b = torch.tensor([[4, 5, 6]])
        jobs = [module.MicrobatchRequest('job-a', 'team-a', 'req-a', contract,
                                         {'input_ids': ids_a, 'attention_mask': torch.ones_like(ids_a)}),
                module.MicrobatchRequest('job-b', 'team-b', 'req-b', contract,
                                         {'input_ids': ids_b, 'attention_mask': torch.ones_like(ids_b)})]
        torch.manual_seed(12)
        model = TinyTokenModel().eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        before_gradients = [parameter.grad.clone() for parameter in model.parameters()]
        with torch.inference_mode():
            expected = [model(**job.inputs) for job in jobs]
        plan = module.Microbatch(jobs, profile)
        ids_a.fill_(15)  # Original caller storage no longer controls the snapshot.
        launch = plan.launch()
        with torch.inference_mode():
            logits = model(**launch.inputs)
        outcome = plan.finish(logits, launch_id=launch.launch_id)
        self.assertEqual([(r.job_id, r.owner, r.request_id, r.status) for r in outcome.results],
                         [('job-a', 'team-a', 'req-a', 'ok'), ('job-b', 'team-b', 'req-b', 'ok')])
        for record, target in zip(outcome.results, expected):
            torch.testing.assert_close(record.logits, target, rtol=1e-6, atol=1e-7)
            self.assertFalse(record.logits.requires_grad)
        outcome.results[0].logits.fill_(123)
        torch.testing.assert_close(outcome.results[1].logits, expected[1], rtol=1e-6, atol=1e-7)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in before.items()))
        self.assertTrue(all(torch.equal(parameter.grad, prior)
                            for parameter, prior in zip(model.parameters(), before_gradients)))

    def test_rejects_incompatible_or_unsafe_execution_contracts(self):
        m, profile, jobs = self.fixture()
        variants = [replace(jobs[1].contract, model_id='other'),
                    replace(jobs[1].contract, checkpoint='other'),
                    replace(jobs[1].contract, adapter='other'),
                    replace(jobs[1].contract, runtime_id='other'),
                    replace(jobs[1].contract, device='cuda:0'),
                    replace(jobs[1].contract, model_dtype=torch.float16),
                    replace(jobs[1].contract, seed=18),
                    replace(jobs[1].contract, deterministic=False),
                    replace(jobs[1].contract, batch_separable=False),
                    replace(jobs[1].contract, mode='train'),
                    replace(jobs[1].contract, rng_policy='random'),
                    replace(jobs[1].contract, cache_policy='enabled'),
                    replace(jobs[1].contract, mutable_state_policy='mutable'),
                    replace(jobs[1].contract, tracing=True),
                    replace(jobs[1].contract, intervention=True)]
        for contract in variants:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                m.Microbatch([jobs[0], replace(jobs[1], contract=contract)], profile)
        for contract in [replace(profile.contract, batch_separable=False),
                         replace(profile.contract, tracing=True),
                         replace(profile.contract, intervention=True)]:
            with self.subTest(profile_contract=contract), self.assertRaises(ValueError):
                m.Microbatch([replace(jobs[0], contract=contract)], replace(profile, contract=contract))

    def test_seed_range_matches_resident_worker_contract(self):
        m, profile, jobs = self.fixture()
        for seed in [-1, 2**63, True]:
            contract = replace(profile.contract, seed=seed)
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                m.Microbatch([replace(jobs[0], contract=contract)], replace(profile, contract=contract))
        contract = replace(profile.contract, seed=2**63 - 1)
        plan = m.Microbatch([replace(jobs[0], contract=contract)], replace(profile, contract=contract))
        self.assertEqual(plan.launch().job_ids, ('job-1',))

    def test_accepts_readonly_snapshots_from_resident_worker(self):
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker
        m, profile, jobs = self.fixture()
        worker = ResidentWorker('tiny', TinyTokenModel().eval(), device='cpu', max_batch_size=2,
                                max_sequence_length=3, max_input_bytes=96, max_output_bytes=168)
        try:
            snapshots = [worker.snapshot_request(ForwardRequest(job.request_id, job.inputs, seed=17))
                         for job in jobs]
            requests = [replace(job, inputs=snapshot.inputs) for job, snapshot in zip(jobs, snapshots)]
            launch = m.Microbatch(requests, profile).launch()
            self.assertEqual(launch.job_ids, ('job-1', 'job-2'))
            self.assertTrue(torch.equal(launch.inputs['input_ids'][0:1], snapshots[0].inputs['input_ids']))
        finally:
            worker.close()

    def test_rejects_input_shape_schema_dtype_and_identity_mismatches(self):
        m, profile, jobs = self.fixture()

        class UnsafeDict(dict):
            def __getitem__(self, key):
                raise AssertionError('untrusted getter executed')

        variants = [replace(jobs[1], inputs={'input_ids': torch.ones((1, 3), dtype=torch.int64)}),
                    replace(jobs[1], inputs={**jobs[1].inputs, 'attention_mask': torch.ones((1, 4), dtype=torch.int64)}),
                    replace(jobs[1], inputs={**jobs[1].inputs, 'input_ids': torch.ones((1, 3), dtype=torch.int32)}),
                    replace(jobs[1], inputs={**jobs[1].inputs, 'input_ids': torch.ones((1, 3), dtype=torch.float32, requires_grad=True)}),
                    replace(jobs[1], inputs=UnsafeDict(jobs[1].inputs)),
                    replace(jobs[1], job_id=jobs[0].job_id),
                    replace(jobs[1], owner=jobs[0].owner, request_id=jobs[0].request_id)]
        for bad in variants:
            with self.subTest(job=bad), self.assertRaises(ValueError):
                m.Microbatch([jobs[0], bad], profile)

    def test_caps_request_rows_tokens_and_owned_bytes_before_launch(self):
        m, profile, jobs = self.fixture()
        for changes in [{'max_requests': 1}, {'max_per_owner': 1, 'max_batch_rows': 1},
                        {'max_token_positions': 5}, {'max_logical_input_bytes': 95},
                        {'max_logical_output_bytes': 167}, {'max_logical_total_bytes': 263},
                        {'serial_atol': -1}, {'serial_rtol': float('inf')}, {'serial_rtol': True},
                        {'serial_atol': 10**1000}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.Microbatch(jobs, replace(profile, **changes))
        with self.assertRaises(ValueError):
            m.Microbatch([jobs[0], replace(jobs[1], owner=jobs[0].owner)], profile)

    def test_cancel_before_launch_excludes_input_and_preserves_order(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        self.assertTrue(plan.cancel('job-1'))
        launch = plan.launch()
        self.assertEqual(launch.job_ids, ('job-2',))
        self.assertTrue(torch.equal(launch.inputs['input_ids'], jobs[1].inputs['input_ids']))
        outcome = plan.finish(torch.ones((1, 3, 7), dtype=torch.float32), launch_id=launch.launch_id)
        self.assertEqual([r.status for r in outcome.results], ['cancelled', 'ok'])
        self.assertIsNone(outcome.results[0].logits)
        self.assertEqual(outcome.results[1].owner, 'team-2')
        self.assertFalse(plan.cancel('job-2'))

    def test_cancel_after_launch_suppresses_only_cancelled_owner(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        self.assertEqual(launch.job_ids, ('job-1', 'job-2'))
        self.assertTrue(plan.cancel('job-2'))
        outcome = plan.finish(torch.arange(42, dtype=torch.float32).reshape(2, 3, 7), launch_id=launch.launch_id)
        self.assertEqual([r.status for r in outcome.results], ['ok', 'cancelled'])
        self.assertIsNone(outcome.results[1].logits)
        self.assertEqual(outcome.results[0].logits.shape, (1, 3, 7))

    def test_all_cancelled_skips_launch_and_unprofiled_remainder_fails_closed(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        plan.cancel('job-1')
        plan.cancel('job-2')
        self.assertIsNone(plan.launch())
        self.assertEqual([r.status for r in plan.outcome().results], ['cancelled', 'cancelled'])
        plan = m.Microbatch(jobs, replace(profile, approved_rows=(2,)))
        plan.cancel('job-1')
        with self.assertRaises(ValueError):
            plan.launch()
        self.assertEqual([r.status for r in plan.outcome().results], ['cancelled', 'failed'])
        self.assertEqual(plan.outcome().results[1].reason, 'unprofiled_batch')

    def test_oom_fails_every_launched_owner_and_requires_quarantine(self):
        from looking_glass.resident_worker import WorkerOutOfMemory
        m, profile, jobs = self.fixture()
        for error in [MemoryError('simulated host OOM'), torch.OutOfMemoryError('simulated tensor OOM'),
                      WorkerOutOfMemory('simulated resident-worker OOM')]:
            with self.subTest(error=type(error)):
                plan = m.Microbatch(jobs, profile)
                launch = plan.launch()
                outcome = plan.fail(error, launch_id=launch.launch_id)
                self.assertTrue(outcome.quarantine_required)
                self.assertEqual([r.status for r in outcome.results], ['failed', 'failed'])
                self.assertEqual([r.reason for r in outcome.results], ['out_of_memory', 'out_of_memory'])
                self.assertTrue(all(r.logits is None for r in outcome.results))

    def test_invalid_or_autograd_output_fails_whole_batch(self):
        m, profile, jobs = self.fixture()
        invalid = [torch.zeros((2, 3, 6)), torch.zeros((1, 3, 7)),
                   torch.zeros((2, 3, 7), dtype=torch.float64),
                   torch.full((2, 3, 7), float('nan')),
                   torch.zeros((2, 3, 7), requires_grad=True)]
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype, requires_grad=value.requires_grad):
                plan = m.Microbatch(jobs, profile)
                launch = plan.launch()
                outcome = plan.finish(value, launch_id=launch.launch_id)
                self.assertTrue(outcome.quarantine_required)
                self.assertEqual([r.status for r in outcome.results], ['failed', 'failed'])
                self.assertTrue(all(r.logits is None for r in outcome.results))

    def test_cancel_finish_race_is_linearized(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        barrier = threading.Barrier(2)
        results = {}

        def cancel():
            barrier.wait()
            results['cancelled'] = plan.cancel('job-1')

        def finish():
            barrier.wait()
            results['outcome'] = plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)

        threads = [threading.Thread(target=cancel), threading.Thread(target=finish)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        first = results['outcome'].results[0]
        self.assertEqual(first.status, 'cancelled' if results['cancelled'] else 'ok')
        self.assertEqual(results['outcome'].results[1].status, 'ok')

    def test_stale_launch_token_fails_closed_without_delivering_logits(self):
        m, profile, jobs = self.fixture()
        first, second = m.Microbatch(jobs, profile), m.Microbatch(jobs, profile)
        launch_1, launch_2 = first.launch(), second.launch()
        self.assertNotEqual(launch_1.launch_id, launch_2.launch_id)
        outcome = first.finish(torch.zeros((2, 3, 7)), launch_id=launch_2.launch_id)
        self.assertTrue(outcome.quarantine_required)
        self.assertEqual([r.status for r in outcome.results], ['failed', 'failed'])
        self.assertEqual([r.reason for r in outcome.results], ['invalid_result', 'invalid_result'])
        self.assertTrue(all(r.logits is None for r in outcome.results))

    def test_request_limit_does_not_consume_unbounded_iterables(self):
        m, profile, jobs = self.fixture()
        consumed = []

        def many_requests():
            for i in range(100):
                consumed.append(i)
                yield replace(jobs[i % 2], job_id=f'job-{i}', owner=f'team-{i}')

        with self.assertRaises(ValueError):
            m.Microbatch(many_requests(), profile)
        self.assertEqual(consumed, [0, 1, 2])

    def test_ambient_default_device_cannot_allocate_outside_cpu(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        m, profile, jobs = self.fixture()

        class RecordAllocations(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.devices = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func is torch.ops.aten.empty.memory_format:
                    self.devices.append(result.device.type)
                return result

        previous = torch.get_default_device()
        monitor = RecordAllocations()
        try:
            torch.set_default_device('meta')
            with monitor:
                launch = m.Microbatch(jobs, profile).launch()
            self.assertTrue(all(value.device.type == 'cpu' for value in launch.inputs.values()))
            self.assertTrue(all(device == 'cpu' for device in monitor.devices), monitor.devices)
        finally:
            torch.set_default_device(previous)

    def test_launch_nonce_failure_leaves_terminal_failed_results(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        with patch('looking_glass.microbatch.uuid.uuid4', side_effect=OSError('entropy unavailable')):
            with self.assertRaises(OSError):
                plan.launch()
        outcome = plan.outcome()
        self.assertEqual([result.status for result in outcome.results], ['failed', 'failed'])
        self.assertFalse(outcome.quarantine_required)
        self.assertTrue(all(result.logits is None for result in outcome.results))
        self.assertFalse(plan.cancel('job-1'))

    def test_prelaunch_assembly_oom_fails_all_without_model_quarantine(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        with patch('looking_glass.microbatch.torch.cat', side_effect=MemoryError('simulated')):
            with self.assertRaises(MemoryError):
                plan.launch()
        outcome = plan.outcome()
        self.assertEqual([r.reason for r in outcome.results], ['out_of_memory', 'out_of_memory'])
        self.assertFalse(outcome.quarantine_required)

    def test_launch_record_allocation_failure_is_terminal_without_launch(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        with patch('looking_glass.microbatch.BatchLaunch', side_effect=MemoryError('record allocation failed')):
            with self.assertRaises(MemoryError):
                plan.launch()
        outcome = plan.outcome()
        self.assertEqual([r.reason for r in outcome.results], ['out_of_memory', 'out_of_memory'])
        self.assertFalse(outcome.quarantine_required)

    def test_second_output_clone_oom_cannot_release_first_result(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        logits = torch.ones((2, 3, 7))

        class FailSecondClone(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.clones = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.clone.default:
                    self.clones += 1
                    if self.clones == 2:
                        raise MemoryError('simulated output clone failure')
                return func(*args, **(kwargs or {}))

        with FailSecondClone():
            outcome = plan.finish(logits, launch_id=launch.launch_id)
        self.assertEqual([r.status for r in outcome.results], ['failed', 'failed'])
        self.assertEqual([r.reason for r in outcome.results], ['out_of_memory', 'out_of_memory'])
        self.assertTrue(all(r.logits is None for r in outcome.results))
        self.assertTrue(outcome.quarantine_required)

    def test_result_publication_oom_is_fatal_and_keeps_postlaunch_quarantine(self):
        m, profile, jobs = self.fixture()
        for failed_type in ('RequestResult', 'BatchOutcome'):
            with self.subTest(failed_type=failed_type):
                plan = m.Microbatch(jobs, profile)
                snapshot = weakref.ref(plan._snapshots['job-1']['input_ids'])
                launch = plan.launch()
                with patch(f'looking_glass.microbatch.{failed_type}', side_effect=MemoryError('cannot publish')):
                    with self.assertRaises(MemoryError):
                        plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
                self.assertTrue(plan.publication_failed)
                self.assertTrue(plan.publication_quarantine_required)
                self.assertFalse(plan.cancel('job-1'))
                with self.assertRaises(RuntimeError):
                    plan.outcome()
                with self.assertRaises(RuntimeError):
                    plan.launch()
                with self.assertRaises(RuntimeError):
                    plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
                with self.assertRaises(RuntimeError):
                    plan.fail(MemoryError('retry'), launch_id=launch.launch_id)
                self.assertIsNone(snapshot())

    def test_prelaunch_publication_oom_does_not_claim_launch_quarantine(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        plan.cancel('job-1')
        plan.cancel('job-2')
        with patch('looking_glass.microbatch.BatchOutcome', side_effect=MemoryError('cannot publish')):
            with self.assertRaises(MemoryError):
                plan.launch()
        self.assertTrue(plan.publication_failed)
        self.assertFalse(plan.publication_quarantine_required)
        self.assertFalse(plan.cancel('job-1'))

    def test_prelaunch_active_list_oom_finalizes_without_device_quarantine(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        builtin_tuple = tuple
        calls = 0

        def injected_tuple(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError('active list allocation failed')
            return builtin_tuple(value)

        with patch('looking_glass.microbatch.tuple', side_effect=injected_tuple, create=True):
            with self.assertRaises(MemoryError):
                plan.launch()
        self.assertEqual([r.reason for r in plan.outcome().results], ['out_of_memory', 'out_of_memory'])
        self.assertFalse(plan.outcome().quarantine_required)

    def test_postlaunch_shape_oom_or_interrupt_quarantines_and_cannot_stay_running(self):
        m, profile, jobs = self.fixture()
        for injected in (MemoryError('shape allocation failed'), KeyboardInterrupt()):
            with self.subTest(injected=type(injected)):
                plan = m.Microbatch(jobs, profile)
                launch = plan.launch()

                class BadLength(tuple):
                    def __len__(self):
                        raise injected

                plan._active = BadLength(plan._active)
                if isinstance(injected, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
                    outcome = plan.outcome()
                else:
                    outcome = plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
                self.assertTrue(outcome.quarantine_required)
                self.assertEqual([r.status for r in outcome.results], ['failed', 'failed'])
                self.assertFalse(plan.cancel('job-1'))

    def test_cancellation_bookkeeping_oom_is_unpublishable_and_fail_closed(self):
        m, profile, jobs = self.fixture()

        class FailingCancellationSet(set):
            def add(self, item):
                raise MemoryError('cancel bookkeeping failed')

        for launched in (False, True):
            with self.subTest(launched=launched):
                plan = m.Microbatch(jobs, profile)
                if launched:
                    plan.launch()
                plan._cancelled = FailingCancellationSet()
                with self.assertRaises(MemoryError):
                    plan.cancel('job-1')
                self.assertTrue(plan.publication_failed)
                self.assertEqual(plan.publication_quarantine_required, launched)
                with self.assertRaises(RuntimeError):
                    plan.outcome()
                self.assertFalse(plan.cancel('job-1'))

    def test_failure_classification_oom_is_unpublishable_and_quarantined(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        with patch('looking_glass.microbatch._failure_reason', side_effect=MemoryError('cannot classify')):
            with self.assertRaises(MemoryError):
                plan.fail(MemoryError('device OOM'), launch_id=launch.launch_id)
        self.assertTrue(plan.publication_failed)
        self.assertTrue(plan.publication_quarantine_required)
        self.assertFalse(plan.cancel('job-1'))

    def test_classifier_failure_during_launch_or_finish_is_unpublishable(self):
        m, profile, jobs = self.fixture()
        for launched in (False, True):
            with self.subTest(launched=launched):
                plan = m.Microbatch(jobs, profile)
                launch = plan.launch() if launched else None
                target = 'torch.isfinite' if launched else 'torch.cat'
                with patch(f'looking_glass.microbatch.{target}', side_effect=MemoryError('original')):
                    with patch('looking_glass.microbatch._failure_reason', side_effect=MemoryError('classification')):
                        with self.assertRaises(MemoryError):
                            if launched:
                                plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
                            else:
                                plan.launch()
                self.assertTrue(plan.publication_failed)
                self.assertEqual(plan.publication_quarantine_required, launched)
                self.assertFalse(plan.cancel('job-1'))
                with self.assertRaises(RuntimeError):
                    plan.outcome()

    def test_interrupt_during_launch_state_commit_remains_prelaunch(self):
        import inspect
        import sys
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        lines, first_line = inspect.getsourcelines(m.Microbatch.launch)
        return_line = next(first_line + index for index, line in enumerate(lines)
                           if line.strip() == 'return launch')
        previous_trace = sys.gettrace()

        def interrupt_at_return(frame, event, arg):
            if event == 'line' and frame.f_code is m.Microbatch.launch.__code__ and frame.f_lineno == return_line:
                raise KeyboardInterrupt
            return interrupt_at_return

        try:
            sys.settrace(interrupt_at_return)
            with self.assertRaises(KeyboardInterrupt):
                plan.launch()
        finally:
            sys.settrace(previous_trace)
        self.assertEqual([r.status for r in plan.outcome().results], ['failed', 'failed'])
        self.assertFalse(plan.outcome().quarantine_required)
        self.assertFalse(plan.cancel('job-1'))

    def test_interrupt_after_completion_state_commit_is_unpublishable(self):
        import inspect
        import sys
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        lines, first_line = inspect.getsourcelines(m.Microbatch._complete)
        return_line = next(first_line + index for index, line in enumerate(lines)
                           if line.strip().startswith('return '))
        previous_trace = sys.gettrace()

        def interrupt_at_return(frame, event, arg):
            if event == 'line' and frame.f_code is m.Microbatch._complete.__code__ and frame.f_lineno == return_line:
                raise KeyboardInterrupt
            return interrupt_at_return

        try:
            sys.settrace(interrupt_at_return)
            with self.assertRaises(KeyboardInterrupt):
                plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
        finally:
            sys.settrace(previous_trace)
        self.assertTrue(plan.publication_failed)
        self.assertTrue(plan.publication_quarantine_required)
        self.assertFalse(plan.cancel('job-1'))
        self.assertFalse(plan._snapshots)
        with self.assertRaises(RuntimeError):
            plan.outcome()

    def test_interrupt_during_worker_failure_processing_preserves_error_and_quarantines(self):
        import inspect
        import sys
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        original_error = MemoryError('original worker failure')
        lines, first_line = inspect.getsourcelines(m.Microbatch.fail)
        comparison_line = next(first_line + index for index, line in enumerate(lines)
                               if line.strip().startswith('if launch_id !='))
        previous_trace = sys.gettrace()

        def interrupt_at_comparison(frame, event, arg):
            if event == 'line' and frame.f_code is m.Microbatch.fail.__code__ and frame.f_lineno == comparison_line:
                raise KeyboardInterrupt
            return interrupt_at_comparison

        try:
            sys.settrace(interrupt_at_comparison)
            with self.assertRaises(KeyboardInterrupt) as raised:
                plan.fail(original_error, launch_id=launch.launch_id)
        finally:
            sys.settrace(previous_trace)
        self.assertIs(raised.exception.__cause__, original_error)
        self.assertTrue(plan.publication_failed)
        self.assertTrue(plan.publication_quarantine_required)
        self.assertFalse(plan.cancel('job-1'))

    def test_snapshot_failure_does_not_expose_a_partial_plan(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        m, profile, jobs = self.fixture()
        original = [job.inputs['input_ids'].clone() for job in jobs]
        sentinel = object()
        plan = sentinel

        class FailSecondClone(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.clones = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.clone.default:
                    self.clones += 1
                    if self.clones == 2:
                        raise MemoryError('simulated snapshot failure')
                return func(*args, **(kwargs or {}))

        with FailSecondClone():
            with self.assertRaises(MemoryError):
                plan = m.Microbatch(jobs, profile)
        self.assertIs(plan, sentinel)
        self.assertTrue(all(torch.equal(job.inputs['input_ids'], before)
                            for job, before in zip(jobs, original)))

    def test_interrupt_during_assembly_or_result_check_finishes_and_reraises(self):
        m, profile, jobs = self.fixture()
        plan = m.Microbatch(jobs, profile)
        with patch('looking_glass.microbatch.torch.cat', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                plan.launch()
        self.assertEqual([r.status for r in plan.outcome().results], ['failed', 'failed'])
        self.assertFalse(plan.outcome().quarantine_required)
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        with patch('looking_glass.microbatch.torch.isfinite', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
        self.assertEqual([r.status for r in plan.outcome().results], ['failed', 'failed'])
        self.assertTrue(plan.outcome().quarantine_required)

    def test_multiple_rows_per_request_split_in_request_order(self):
        m, profile, jobs = self.fixture()
        schema = (('attention_mask', torch.bool), ('input_ids', torch.int32), ('position_ids', torch.int64))
        profile = replace(profile, input_schema=schema, input_shape=(2, 3), approved_rows=(2, 4),
                          max_batch_rows=4, max_token_positions=12, max_logical_input_bytes=156,
                          max_logical_output_bytes=336, max_logical_total_bytes=492)
        values = [{'attention_mask': torch.ones((2, 3), dtype=torch.bool),
                   'input_ids': torch.full((2, 3), i, dtype=torch.int32),
                   'position_ids': torch.arange(3).repeat(2, 1)} for i in (1, 2)]
        jobs = [replace(job, inputs=value) for job, value in zip(jobs, values)]
        plan = m.Microbatch(jobs, profile)
        launch = plan.launch()
        self.assertEqual(launch.inputs['input_ids'].shape, (4, 3))
        logits = torch.arange(84, dtype=torch.float32).reshape(4, 3, 7)
        outcome = plan.finish(logits, launch_id=launch.launch_id)
        self.assertTrue(torch.equal(outcome.results[0].logits, logits[:2]))
        self.assertTrue(torch.equal(outcome.results[1].logits, logits[2:]))
        outcome.results[0].logits.fill_(5)
        self.assertEqual(logits[0, 0, 0], 0)
        self.assertEqual(outcome.results[1].logits[0, 0, 0], 42)

    def test_batch_preparation_does_not_change_torch_rng_state(self):
        m, profile, jobs = self.fixture()
        torch.manual_seed(872)
        before = torch.get_rng_state().clone()
        plan = m.Microbatch(jobs, profile)
        plan.launch()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_inference_context_returns_ordinary_owned_input_and_output_tensors(self):
        m, profile, jobs = self.fixture()
        with torch.inference_mode():
            plan = m.Microbatch(jobs, profile)
            launch = plan.launch()
            outcome = plan.finish(torch.zeros((2, 3, 7)), launch_id=launch.launch_id)
        self.assertFalse(launch.inputs['input_ids'].is_inference())
        self.assertFalse(outcome.results[0].logits.is_inference())
        launch.inputs['input_ids'].fill_(9)
        outcome.results[0].logits.fill_(3)
        self.assertEqual(jobs[0].inputs['input_ids'][0, 0], 1)
        self.assertEqual(outcome.results[1].logits[0, 0, 0], 0)


if __name__ == '__main__':
    unittest.main()
