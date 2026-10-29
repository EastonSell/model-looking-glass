"""CPU-only checks for the planned five-token calibration evidence boundary."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from looking_glass.agreement_calibration_collector import validate_plan_metadata


SITES = [(f'transformer.h.{layer}.{component}', 4, [] if component == 'mlp' else [0])
         for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')]
SITES += [('transformer.h.5', 4, [0]), ('transformer.h.5', 1, [0])]
CASES = {
    **{f'layer{layer}_{component}_cross_patch': (f'transformer.h.{layer}.{component}', 4, 'patch', None)
       for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')},
    'null': ('transformer.h.1.attn', 4, 'null', None),
    'self_patch': ('transformer.h.1.attn', 4, 'patch', 'before'),
    'block_last_cross_patch': ('transformer.h.5', 4, 'patch', None),
    'block_subject_cross_patch': ('transformer.h.5', 1, 'patch', None),
}


def event(name, operation, path, token, selector, *, whole=False, source=None, order=0, preceding=()):
    return {'name': name, 'operation': operation, 'order': order, 'module_path': path,
            'module_aliases': [path], 'call_site_known': False, 'forward_index': 0,
            'invocation': 0, 'batch_index': None if whole else 0,
            'token_index': None if whole else token, 'output_selector': selector,
            'output_shape': [1, 5, 768], 'selected_shape': [1, 5, 768] if whole else [768],
            'dtype': 'torch.float32', 'device': 'cuda:0', 'source_capture': source,
            'preceding_same_site_actions': list(preceding)}


def metadata(kind):
    if kind in ('warmup', 'clean_capture'):
        steps = [event(f'capture_{i}', 'capture', path, token, selector, order=i,
                       preceding=('capture_8',) if i == 9 else ())
                 for i, (path, token, selector) in enumerate(SITES)]
        return {'steps': steps, 'capture_bytes': 30720, 'source_bytes': 0}
    path, token, operation, source = CASES[kind]
    selector = [] if path.endswith('.mlp') else [0]
    specs = [('whole_before', 'capture', True, None), ('before', 'capture', False, None),
             ('action', operation, False, source), ('after', 'capture', False, None),
             ('whole_after', 'capture', True, None)]
    steps = [event(name, action, path, token, selector, whole=whole, source=source_name, order=i,
                   preceding=tuple(item[0] for item in specs[:i]))
             for i, (name, action, whole, source_name) in enumerate(specs)]
    return {'steps': steps, 'capture_bytes': 36864,
            'source_bytes': 3072 if kind.endswith('cross_patch') else 0}


class AgreementCalibrationMetadataTests(unittest.TestCase):
    def test_accepts_exact_fixed_capture_and_all_action_sites(self):
        for kind in ('warmup', 'clean_capture', *CASES):
            with self.subTest(kind=kind):
                validate_plan_metadata(kind, metadata(kind))

    def test_rejects_mutated_actual_plan_and_accounting(self):
        kind = 'layer2_mlp_cross_patch'
        mutations = []
        for field, value in (('module_path', 'transformer.h.3.mlp'), ('token_index', 1),
                             ('operation', 'null'), ('output_selector', [0]),
                             ('output_shape', [1, 4, 768]), ('selected_shape', [1, 768]),
                             ('dtype', 'torch.float16'), ('device', 'cpu'), ('invocation', True)):
            altered = deepcopy(metadata(kind))
            altered['steps'][2][field] = value
            mutations.append(altered)
        for field, value in (('capture_bytes', 36863), ('source_bytes', 0)):
            altered = deepcopy(metadata(kind))
            altered[field] = value
            mutations.append(altered)
        extra = deepcopy(metadata(kind))
        extra['steps'][0]['logits'] = [1.0]
        mutations.append(extra)
        for altered in mutations:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                validate_plan_metadata(kind, altered)

    def test_rejects_missing_or_reordered_clean_capture(self):
        for altered in (metadata('clean_capture')['steps'][:-1],
                        list(reversed(metadata('clean_capture')['steps']))):
            with self.subTest(steps=altered), self.assertRaises(ValueError):
                validate_plan_metadata('clean_capture', {'steps': altered, 'capture_bytes': 30720, 'source_bytes': 0})

    def test_matches_real_cpu_plan_metadata_after_device_normalization(self):
        # Exercise the actual intervention plan; CPU is substituted only for
        # the device field, so this is not evidence of CUDA execution.
        import torch
        from torch import nn
        from looking_glass.agreement_patch_runner import AgreementCase, _SITES, _steps_for_action, _steps_for_capture
        from looking_glass.intervention_plan import ActivationPlan

        class Attention(nn.Module):
            def forward(self, value): return (value, None)
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn, self.mlp = Attention(), nn.Identity()
            def forward(self, value): return (self.mlp(self.attn(value)[0]),)
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.ModuleList(Block() for _ in range(6))
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer()
            def forward(self, value):
                for block in self.transformer.h:
                    value = block(value)[0]
                return value

        model = Model().eval()
        case = AgreementCase('fixed', 'calibration', 'cat', 'singular', (464, 3797, 1474, 262, 9168), 1)
        for kind in ('warmup', 'clean_capture', *CASES):
            if kind in ('warmup', 'clean_capture'):
                steps = _steps_for_capture(case)
            else:
                path, token, operation, source = CASES[kind]
                site = next(item for item in _SITES if item['module_path'] == path and
                            (1 if item['token_role'] == 'subject' else 4) == token)
                donor = torch.zeros(768, dtype=torch.float32) if kind.endswith('cross_patch') else None
                steps = _steps_for_action(case, site, operation, donor)
            with ActivationPlan(model, steps, max_tensor_bytes=1024 * 1024,
                                max_capture_bytes=512 * 1024, max_source_bytes=128 * 1024,
                                max_steps=16, max_invocations=1) as plan:
                model(torch.zeros((1, 5, 768), dtype=torch.float32))
            actual = plan.metadata
            for step in actual['steps']:
                self.assertEqual(step['device'], 'cpu')
                step['device'] = 'cuda:0'
            with self.subTest(kind=kind):
                validate_plan_metadata(kind, actual)


class AgreementCalibrationSamplingTests(unittest.TestCase):
    def setUp(self):
        from looking_glass import agreement_calibration_collector as collector
        self.collector = collector
        self.custody = patch.object(collector, '_SAMPLING_CUSTODY', [None] * 4)
        self.custody.start()
        self.addCleanup(self.custody.stop)

    def runner(self, failure=None):
        from test_agreement_cuda_runner import workload
        from looking_glass.agreement_cuda_calibration import CudaAgreementCalibrator
        from looking_glass.agreement_cuda_runner import WARMUP_POLICY_SHA256, calibration_case_sha256
        from looking_glass.memory_budget import CudaMemorySnapshot, MemoryBudget
        key = workload()
        runner = object.__new__(CudaAgreementCalibrator)
        runner._expected_key, runner._warmup_completed, runner._calibration_index = key, False, 0
        runner._quarantine_latched = runner._device_drain_uncertain = False
        calls = []
        def check():
            calls.append('check')
            if runner.quarantined:
                raise RuntimeError('unsafe')
        def sync():
            calls.append('sync')
            if failure == 'sync': raise RuntimeError('sync failed')
        snapshot_count = [0]
        def snapshot():
            calls.append('snapshot')
            snapshot_count[0] += 1
            if (failure == 'bad-baseline' and snapshot_count[0] == 1 or
                    failure == 'bad-after' and snapshot_count[0] == 2):
                return CudaMemorySnapshot(0, 8 * 1024 ** 3, 400 * 1024 ** 2, 500 * 1024 ** 2)
            return CudaMemorySnapshot(6 * 1024 ** 3, 8 * 1024 ** 3, 400 * 1024 ** 2, 500 * 1024 ** 2)
        budget_enabled = [False]
        def telemetry():
            calls.append('budget')
            if budget_enabled[0] and failure == 'bad-budget':
                return CudaMemorySnapshot(0, 8 * 1024 ** 3, 400 * 1024 ** 2, 500 * 1024 ** 2)
            return CudaMemorySnapshot(6 * 1024 ** 3, 8 * 1024 ** 3, 400 * 1024 ** 2, 500 * 1024 ** 2)
        def reset(): calls.append('reset')
        def peaks():
            calls.append('peaks')
            return (450 * 1024 ** 2, 540 * 1024 ** 2)
        def warmup():
            calls.append('warmup')
            runner._warmup_completed = True
            return {'sample_id': 'warmup', 'policy_sha256': WARMUP_POLICY_SHA256, 'seed': 20261028,
                    'model_state_sha256': 'd' * 64, 'plan_metadata': metadata('warmup')}
        def action(kind):
            calls.append(kind)
            if failure == kind: raise RuntimeError('action failed')
            runner._calibration_index += 1
            result = {'sample_id': kind, 'case_sha256': calibration_case_sha256(kind), 'seed': 20261028,
                      'plan_metadata': metadata(kind), 'capture_or_landing_checked': True}
            if failure == 'bad-plan': result['plan_metadata']['steps'][0]['token_index'] = 3
            return result
        runner._check_owner, runner.synchronize_for_measurement = check, sync
        runner.measurement_snapshot, runner.reset_measurement_peak, runner.measurement_peaks = snapshot, reset, peaks
        runner.warmup, runner.calibrate_action = warmup, action
        budget = MemoryBudget(telemetry, max_gpu_trace_bytes=2 * 1024 ** 2, max_host_trace_bytes=2 * 1024 ** 2)
        runner._resident_lease = budget.acquire(2 * 1024 ** 3, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        runner._memory_lease = budget.acquire(3 * 1024 ** 3 // 2, max_gpu_trace_bytes=2 * 1024 ** 2,
                                              max_host_trace_bytes=2 * 1024 ** 2)
        runner._memory_lease.reserve_trace('gpu', 2 * 1024 ** 2)
        runner._memory_lease.reserve_trace('host', 2 * 1024 ** 2)
        budget_enabled[0] = True
        calls.clear()
        return runner, key, calls

    def test_default_gate_stops_before_any_measurement(self):
        runner, key, calls = self.runner()
        with self.assertRaises(RuntimeError):
            self.collector.measure_calibration(runner, key, lambda value: None)
        self.assertEqual(calls, [])

    def test_measures_cold_then_every_fixed_warm_case(self):
        from looking_glass.agreement_cuda_runner import CALIBRATION_KINDS, WARMUP_POLICY_SHA256
        runner, key, calls = self.runner()
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            cold, warm = self.collector.measure_calibration(runner, key, records.append)
        self.assertIsNotNone(cold)
        self.assertEqual(cold.sample_id, 'warmup')
        self.assertEqual(cold.case_sha256, WARMUP_POLICY_SHA256)
        self.assertEqual([sample.sample_id for sample in warm], list(CALIBRATION_KINDS))
        self.assertEqual(len(records), 84)
        self.assertEqual([item['kind'] for item in records if item['phase'] == 'completed'],
                         ['warmup', *CALIBRATION_KINDS])
        self.assertEqual(calls.count('sync'), 42)
        self.assertEqual(calls.count('reset'), 14)
        self.assertEqual(calls.count('peaks'), 14)
        first_sync = calls.index('sync')
        reset = calls.index('reset', first_sync + 1)
        second_sync = calls.index('sync', reset + 1)
        baseline = calls.index('snapshot', second_sync + 1)
        self.assertLess(baseline, calls.index('warmup'))
        self.assertFalse(runner.quarantined)

    def test_uncertain_callback_quarantines_and_does_not_retry_sync(self):
        runner, key, calls = self.runner(failure='layer1_attn_cross_patch')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaisesRegex(RuntimeError, 'action failed'):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertEqual(calls[-1], 'layer1_attn_cross_patch')
        self.assertNotIn('layer1_mlp_cross_patch', calls)
        self.assertEqual([item['kind'] for item in records if item['phase'] == 'completed'],
                         ['warmup', 'clean_capture'])

    def test_failed_callback_preserves_its_prework_baseline(self):
        runner, key, calls = self.runner(failure='layer1_attn_cross_patch')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(RuntimeError):
                self.collector.measure_calibration(runner, key, records.append)
        attempt = [item for item in records if item['kind'] == 'layer1_attn_cross_patch']
        self.assertEqual([item['phase'] for item in attempt], ['started', 'baseline', 'baseline_check', 'failed'])
        self.assertEqual(attempt[1]['snapshot']['device_total_bytes'], 8 * 1024 ** 3)
        self.assertEqual(attempt[-1]['error_type'], 'RuntimeError')
        self.assertEqual(calls[-1], 'layer1_attn_cross_patch')

    def test_invalid_return_metadata_quarantines_before_post_callback_sync(self):
        runner, key, calls = self.runner(failure='bad-plan')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertEqual(calls[-1], 'check')
        self.assertEqual(calls.count('sync'), 5)  # Three for warmup; two before the invalid action.
        self.assertEqual([item['kind'] for item in records if item['phase'] == 'completed'], ['warmup'])

    def test_failed_sync_is_not_retried(self):
        runner, key, calls = self.runner(failure='sync')
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                self.collector.measure_calibration(runner, key, lambda value: None)
        self.assertTrue(runner.quarantined)
        self.assertEqual(calls.count('sync'), 1)
        self.assertNotIn('warmup', calls)

    def test_insufficient_recorded_baseline_blocks_callback(self):
        runner, key, calls = self.runner(failure='bad-baseline')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(Exception):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertNotIn('warmup', calls)
        self.assertEqual(next(item for item in records if item['phase'] == 'baseline')['snapshot']['device_free_bytes'], 0)
        self.assertFalse(any(item['phase'] == 'completed' for item in records))

    def test_insufficient_recorded_endpoint_is_partial_and_stops(self):
        runner, key, calls = self.runner(failure='bad-after')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(Exception):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertIn('warmup', calls)
        self.assertNotIn('clean_capture', calls)
        self.assertEqual(next(item for item in records if item['phase'] == 'after_observed')['snapshot']['device_free_bytes'], 0)
        self.assertFalse(any(item['phase'] == 'completed' for item in records))

    def test_insufficient_fresh_budget_blocks_callback(self):
        runner, key, calls = self.runner(failure='bad-budget')
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertNotIn('warmup', calls)
        self.assertEqual(next(item for item in records if item['phase'] == 'baseline_check')
                         ['budget']['telemetry']['device_free_bytes'], 0)

    def test_lost_resident_claim_stops_before_callback(self):
        runner, key, calls = self.runner()
        records = []
        def write(record):
            records.append(record)
            if record['phase'] == 'baseline':
                runner._resident_lease.release()
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, write)
        self.assertTrue(runner.quarantined)
        self.assertNotIn('warmup', calls)
        self.assertTrue(runner._resident_lease.released)
        self.assertFalse(runner._memory_lease.released)

    def test_missing_initial_claim_roots_owner_and_marks_quarantine(self):
        runner, key, calls = self.runner()
        runner._resident_lease.release()
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, records.append)
        self.assertTrue(runner.quarantined)
        self.assertIs(self.collector._SAMPLING_CUSTODY[0], runner)
        self.assertIsInstance(self.collector._SAMPLING_CUSTODY[3], ValueError)
        self.assertNotIn('sync', calls)
        self.assertNotIn('warmup', calls)

    def test_failed_stage_write_quarantines_without_next_device_call(self):
        runner, key, calls = self.runner()
        stages = []
        def write(record):
            stages.append(record)
            if record['kind'] == 'warmup' and record['phase'] == 'completed':
                raise OSError('journal failed')
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaisesRegex(OSError, 'journal failed'):
                self.collector.measure_calibration(runner, key, write)
        self.assertTrue(runner.quarantined)
        self.assertNotIn('clean_capture', calls)
        self.assertEqual(calls.count('sync'), 3)
        self.assertIsInstance(self.collector._SAMPLING_CUSTODY[3], OSError)

    def test_wrong_key_roots_and_quarantines_owner_before_device_work(self):
        from dataclasses import replace
        runner, key, calls = self.runner()
        records = []
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, replace(key, sequence_length=4), records.append)
        self.assertEqual(calls, [])
        self.assertIs(self.collector._SAMPLING_CUSTODY[0], runner)
        self.assertTrue(runner.quarantined)
        self.assertEqual(records[-1]['phase'], 'failed')

    def test_used_owner_is_rooted_and_quarantined_before_device_work(self):
        runner, key, calls = self.runner()
        runner._warmup_completed = True
        with patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, lambda value: None)
        self.assertEqual(calls, [])
        self.assertIs(self.collector._SAMPLING_CUSTODY[0], runner)
        self.assertTrue(runner.quarantined)

    def test_changed_action_order_rejects_before_device_work(self):
        from looking_glass import agreement_cuda_runner as adapter
        runner, key, calls = self.runner()
        with (patch.object(self.collector, 'CALIBRATION_EXECUTION_READY', True),
              patch.object(adapter, 'CALIBRATION_KINDS', tuple(reversed(adapter.CALIBRATION_KINDS)))):
            with self.assertRaises(ValueError):
                self.collector.measure_calibration(runner, key, lambda value: None)
        self.assertEqual(calls, [])
        self.assertIs(self.collector._SAMPLING_CUSTODY[0], runner)
        self.assertTrue(runner.quarantined)


if __name__ == '__main__':
    unittest.main()
