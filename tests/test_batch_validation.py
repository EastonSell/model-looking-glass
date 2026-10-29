import importlib.util
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class BatchValidationTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec('looking_glass.batch_validation') is None:
            self.fail('batch validation module is missing')
        from looking_glass import batch_validation
        self.validator = batch_validation
        self.binding = {
            'profile_id': 'test-profile', 'certification_id': 'test-certification',
            'model_id': 'test-model', 'model_fingerprint': 'a' * 64,
            'checkpoint': 'test-checkpoint', 'checkpoint_sha256': 'b' * 64,
            'adapter': 'test-adapter', 'adapter_sha256': 'c' * 64,
            'runtime_id': 'cpu-torch-test', 'backend': 'eager', 'device': 'cpu',
            'model_dtype': 'float32', 'output_dtype': 'float32',
            'input_schema': [['input_ids', 'int64']], 'input_shape': [1, 3],
            'output_width': 1, 'approved_rows': [2],
            'max_requests': 2, 'max_per_owner': 1, 'max_batch_rows': 2,
            'max_token_positions': 6, 'max_logical_input_bytes': 48,
            'max_logical_output_bytes': 24, 'max_logical_total_bytes': 72,
            'seed': 17, 'serial_atol': 0.0, 'serial_rtol': 0.0,
            'mode': 'eval', 'deterministic': True, 'batch_separable': True,
            'rng_policy': 'none', 'cache_policy': 'disabled',
            'mutable_state_policy': 'none', 'tracing': False, 'intervention': False,
        }
        self.groups = [{'group_id': 'group-1', 'cases': [
            {'case_id': 'case-a', 'owner': 'team-a',
             'inputs': {'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.int64, device='cpu')}},
            {'case_id': 'case-b', 'owner': 'team-b',
             'inputs': {'input_ids': torch.tensor([[3, 4, 5]], dtype=torch.int64, device='cpu')}},
        ]}]
        self.discovery = [{'input_ids': torch.tensor([[8, 9, 10]], dtype=torch.int64, device='cpu')}]

    @staticmethod
    def separable(inputs, *, request_id, seed):
        return inputs['input_ids'].to(torch.float32).unsqueeze(-1)

    def protocol(self):
        hashes = [digest for item in self.discovery
                  for digest in self.validator.input_row_digests(self.binding, item)]
        return self.validator.freeze_protocol(self.binding, self.groups, discovery_row_hashes=hashes)

    def run_protocol(self, protocol, binding, groups, execute, **kwargs):
        return self.validator.run_validation(protocol, binding, groups, execute,
                                             expected_protocol_sha256=protocol['protocol_sha256'], **kwargs)

    def test_separable_outputs_match_per_request_against_serial_reference(self):
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['complete'])
        self.assertEqual(report['calls'], {'serial': 2, 'mixed': 1})
        self.assertEqual([row['status'] for row in report['groups'][0]['cases']], ['matched', 'matched'])
        self.assertEqual([row['max_abs_error'] for row in report['groups'][0]['cases']], [0.0, 0.0])

    def test_batch_coupling_is_detected_despite_deterministic_eval(self):
        def batch_mean(inputs, *, request_id, seed):
            values = inputs['input_ids'].to(torch.float32)
            return (values - values.mean(dim=0, keepdim=True)).unsqueeze(-1)
        report = self.run_protocol(self.protocol(), self.binding, self.groups, batch_mean)
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['complete'])
        cases = report['groups'][0]['cases']
        self.assertEqual([row['status'] for row in cases], ['mismatch', 'mismatch'])
        self.assertEqual([row['max_abs_error'] for row in cases], [1.0, 1.0])
        self.assertEqual([row['max_relative_error'] for row in cases], [None, None])
        self.assertEqual([row['relative_error_nonfinite_reason'] for row in cases],
                         ['zero_reference', 'zero_reference'])

    def test_real_resident_worker_runs_distinct_serial_and_mixed_forwards(self):
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker
        class TokenModel(torch.nn.Module):
            def forward(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1)
        worker = ResidentWorker('test-model', TokenModel().eval(), device='cpu', max_batch_size=2,
                                max_sequence_length=3, max_input_bytes=48, max_output_bytes=24)
        def execute(inputs, *, request_id, seed):
            result = worker.execute(ForwardRequest(request_id, inputs, seed=seed))
            if result.request_id != request_id:
                raise RuntimeError('worker returned the wrong request')
            return result.logits
        try:
            report = self.run_protocol(self.protocol(), self.binding, self.groups, execute)
        finally:
            worker.close()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['calls'], {'serial': 2, 'mixed': 1})
        self.assertEqual([case['max_abs_error'] for case in report['groups'][0]['cases']], [0.0, 0.0])

    def test_random_small_transformer_runs_through_real_cpu_worker(self):
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('optional transformers dependency not installed')
        from looking_glass.resident_worker import ForwardRequest, ResidentWorker
        self.binding.update({'output_width': 32, 'max_logical_output_bytes': 768,
                             'max_logical_total_bytes': 816, 'serial_atol': 1e-6,
                             'serial_rtol': 1e-5})
        previous_threads = torch.get_num_threads()
        worker = None
        try:
            torch.set_num_threads(1)
            with torch.device('cpu'), torch.random.fork_rng(devices=[]):
                from transformers import GPT2Config, GPT2LMHeadModel
                torch.random.default_generator.manual_seed(12345)
                config = GPT2Config(vocab_size=32, n_positions=8, n_ctx=8, n_embd=16,
                                    n_layer=2, n_head=2, use_cache=False,
                                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
                model = GPT2LMHeadModel(config).to(device='cpu', dtype=torch.float32).eval()
                worker = ResidentWorker('test-model', model, device='cpu', max_batch_size=2,
                                        max_sequence_length=3, max_input_bytes=48, max_output_bytes=768)
                def execute(inputs, *, request_id, seed):
                    result = worker.execute(ForwardRequest(request_id, inputs, seed=seed))
                    if result.request_id != request_id:
                        raise RuntimeError('worker returned the wrong request')
                    return result.logits
                report = self.run_protocol(self.protocol(), self.binding, self.groups, execute)
        finally:
            if worker is not None:
                worker.close()
            torch.set_num_threads(previous_threads)
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['complete'])
        self.assertEqual(report['calls'], {'serial': 2, 'mixed': 1})

    def test_binding_from_real_batch_profile_detects_profile_drift(self):
        from dataclasses import replace
        from test_microbatch import MicrobatchTests
        _, profile, jobs = MicrobatchTests.fixture()
        bind = lambda p: self.validator.binding_from_profile(
            p, model_fingerprint='a' * 64, checkpoint_sha256='b' * 64,
            adapter_sha256='c' * 64, backend='cpu-eager')
        binding = bind(profile)
        cases = [{'case_id': job.request_id, 'owner': job.owner, 'inputs': job.inputs} for job in jobs]
        groups = [{'group_id': 'profile-group', 'cases': cases}]
        discovery = {'input_ids': torch.tensor([[8, 9, 10]]),
                     'attention_mask': torch.ones((1, 3), dtype=torch.int64)}
        protocol = self.validator.freeze_protocol(binding, groups,
            discovery_row_hashes=self.validator.input_row_digests(binding, discovery))
        def execute(inputs, *, request_id, seed):
            return inputs['input_ids'].float().unsqueeze(-1).expand(-1, -1, 7)
        report = self.run_protocol(protocol, binding, groups, execute)
        self.assertEqual(report['status'], 'passed')
        changed = bind(replace(profile, serial_atol=profile.serial_atol + 0.1))
        with self.assertRaises(ValueError):
            self.run_protocol(protocol, changed, groups, execute)

    def test_evidence_only_authorizes_row_counts_it_actually_tests(self):
        self.binding.update({'approved_rows': [2, 4], 'max_requests': 4, 'max_batch_rows': 4,
                             'max_token_positions': 12, 'max_logical_input_bytes': 96,
                             'max_logical_output_bytes': 48, 'max_logical_total_bytes': 144})
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        self.assertEqual(report['tested_rows'], [2])
        self.validator.require_evidence(report, protocol=protocol, actual_binding=self.binding,
            expected_protocol_sha256=protocol['protocol_sha256'],
            evidence_sha256=report['evidence_sha256'], rows=2)
        with self.assertRaises(ValueError):
            self.validator.require_evidence(report, protocol=protocol, actual_binding=self.binding,
                expected_protocol_sha256=protocol['protocol_sha256'],
                evidence_sha256=report['evidence_sha256'], rows=4)

    def test_admission_uses_owned_report_after_validation(self):
        self.binding.update({'approved_rows': [2, 4], 'max_requests': 4, 'max_batch_rows': 4,
                             'max_token_positions': 12, 'max_logical_input_bytes': 96,
                             'max_logical_output_bytes': 48, 'max_logical_total_bytes': 144})
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        verify = self.validator._verify_evidence
        def changes_caller_report(*args, **kwargs):
            verify(*args, **kwargs)
            report['tested_rows'].append(4)
        with patch.object(self.validator, '_verify_evidence', side_effect=changes_caller_report):
            with self.assertRaises(ValueError):
                self.validator.require_evidence(report, protocol=protocol, actual_binding=self.binding,
                    expected_protocol_sha256=protocol['protocol_sha256'],
                    evidence_sha256=report['evidence_sha256'], rows=4)

    def test_relative_tolerance_uses_serial_as_reference(self):
        self.binding['serial_rtol'] = 0.1
        def outputs(inputs, *, request_id, seed):
            value = 111.0 if request_id.startswith('mixed-') else 100.0
            return torch.full((inputs['input_ids'].shape[0], 3, 1), value, dtype=torch.float32, device='cpu')
        report = self.run_protocol(self.protocol(), self.binding, self.groups, outputs)
        self.assertEqual(report['status'], 'failed')
        for case in report['groups'][0]['cases']:
            self.assertEqual(case['max_abs_error'], 11.0)
            self.assertAlmostEqual(case['max_relative_error'], 0.11)

    def test_tolerance_boundary_does_not_round_a_mismatch_into_a_match(self):
        reference = torch.tensor([1.0], dtype=torch.float64)
        candidate = torch.tensor([math.nextafter(1.0, math.inf)], dtype=torch.float64)
        difference = 2.0 ** -52
        atol = math.nextafter(difference, 0.0)
        # atol is difference - 2**-105; the additional 2**-106 is too small.
        result = self.validator._comparison(reference, candidate, atol, 2.0 ** -106)
        self.assertEqual(result['status'], 'mismatch')
        # Exactly filling the gap must pass.
        result = self.validator._comparison(reference, candidate, atol, 2.0 ** -105)
        self.assertEqual(result['status'], 'matched')

    def test_nonrepresentable_integer_tolerance_is_rejected(self):
        self.binding['serial_atol'] = 9007199254740995
        with self.assertRaises(ValueError):
            self.protocol()

    def test_malformed_approved_rows_are_rejected_as_invalid_binding(self):
        self.binding['approved_rows'] = [[2]]
        with self.assertRaises(ValueError):
            self.protocol()

    def test_extreme_finite_values_do_not_compare_infinity_to_infinity(self):
        self.binding.update({'model_dtype': 'float64', 'output_dtype': 'float64', 'serial_rtol': 1.5,
                             'max_logical_output_bytes': 48, 'max_logical_total_bytes': 96})
        maximum = torch.finfo(torch.float64).max
        def outputs(inputs, *, request_id, seed):
            value = maximum if request_id.startswith('mixed-') else -maximum
            return torch.full((inputs['input_ids'].shape[0], 3, 1), value, dtype=torch.float64, device='cpu')
        report = self.run_protocol(self.protocol(), self.binding, self.groups, outputs)
        self.assertEqual(report['status'], 'failed')
        for case in report['groups'][0]['cases']:
            self.assertIsNone(case['max_abs_error'])
            self.assertTrue(case['absolute_error_overflow'])
            self.assertEqual(case['max_relative_error'], 2.0)

    def test_subnormals_require_a_denormal_preserving_runtime(self):
        script = r'''
import sys, torch
sys.path.insert(0, 'tests')
from test_batch_validation import BatchValidationTests
from looking_glass import batch_validation as v
if not torch.set_flush_denormal(False):
    sys.exit(77)
for dtype, itype in ((torch.float32, torch.int32), (torch.float64, torch.int64)):
    zero = torch.zeros((1,), dtype=dtype, device='cpu')
    tiny = torch.tensor([1], dtype=itype, device='cpu').view(dtype)
    result = v._comparison(zero, tiny, 0.0, 0.0)
    assert result['status'] == 'mismatch'
    assert result['max_abs_error'] == tiny.tolist()[0]
    assert result['max_relative_error'] is None
    assert result['relative_error_nonfinite_reason'] == 'zero_reference'
test = BatchValidationTests(); test.setUp()
protocol = test.protocol()
assert torch.set_flush_denormal(True)
try:
    v.run_validation(protocol, test.binding, test.groups, test.separable,
                     expected_protocol_sha256=protocol['protocol_sha256'])
except v.NumericModeError:
    pass
else:
    raise AssertionError('flushed denormals accepted')
assert torch.set_flush_denormal(False)
def changes_mode(inputs, *, request_id, seed):
    torch.set_flush_denormal(True)
    return test.separable(inputs, request_id=request_id, seed=seed)
try:
    v.run_validation(protocol, test.binding, test.groups, changes_mode,
                     expected_protocol_sha256=protocol['protocol_sha256'])
except v.NumericModeError:
    pass
else:
    raise AssertionError('mode change during execution accepted')
assert torch.set_flush_denormal(False)
import threading
cancel = threading.Event()
def cancels_and_changes_mode(inputs, *, request_id, seed):
    cancel.set()
    torch.set_flush_denormal(True)
    return test.separable(inputs, request_id=request_id, seed=seed)
try:
    v.run_validation(protocol, test.binding, test.groups, cancels_and_changes_mode,
                     expected_protocol_sha256=protocol['protocol_sha256'], cancel_event=cancel)
except v.NumericModeError:
    pass
else:
    raise AssertionError('mode change during cancellation accepted')
'''
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
        if result.returncode == 77:
            self.skipTest('flush-denormal control unavailable on this CPU')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_frozen_binding_and_heldout_contents_cannot_be_changed(self):
        protocol = self.protocol()
        for field, value in [('model_fingerprint', 'd' * 64), ('runtime_id', 'other-runtime'),
                             ('profile_id', 'other-profile'), ('serial_atol', 1.0)]:
            changed = copy.deepcopy(self.binding)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_protocol(protocol, changed, self.groups, self.separable)
        self.groups[0]['cases'][0]['inputs']['input_ids'][0, 0] = 8
        with self.assertRaises(ValueError):
            self.run_protocol(protocol, self.binding, self.groups, self.separable)

    def test_executor_cannot_change_tolerance_after_protocol_check(self):
        protocol = self.protocol()
        def changes_binding(inputs, *, request_id, seed):
            self.binding['serial_atol'] = 100.0
            logits = self.separable(inputs, request_id=request_id, seed=seed)
            return logits + (1.0 if request_id.startswith('mixed-') else 0.0)
        report = self.run_protocol(protocol, self.binding, self.groups, changes_binding)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([case['max_abs_error'] for case in report['groups'][0]['cases']], [1.0, 1.0])

    def test_executor_cannot_rewrite_later_inputs_and_protocol_manifest(self):
        second = copy.deepcopy(self.groups[0])
        second['group_id'] = 'group-2'
        self.groups.append(second)
        protocol = self.protocol()
        def alters_later_case(inputs, *, request_id, seed):
            if request_id == 'mixed-0':
                target = self.groups[1]['cases'][0]['inputs']
                target['input_ids'][0, 0] = 7
                protocol['groups'][1]['cases'][0]['input_sha256'] = self.validator.input_digest(self.binding, target)
            return self.separable(inputs, request_id=request_id, seed=seed)
        report = self.run_protocol(protocol, self.binding, self.groups, alters_later_case)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertFalse(report['complete'])
        self.assertEqual(report['error']['kind'], 'input_changed')

    def test_discovery_overlap_and_nonmixed_groups_are_rejected(self):
        overlap = self.validator.input_row_digests(self.binding, self.groups[0]['cases'][0]['inputs'])[0]
        with self.assertRaises(ValueError):
            self.validator.freeze_protocol(self.binding, self.groups, discovery_row_hashes=[overlap])
        with self.assertRaises(ValueError):
            self.validator.freeze_protocol(self.binding, self.groups, discovery_row_hashes=[])
        self.groups[0]['cases'][1]['owner'] = 'team-a'
        with self.assertRaises(ValueError):
            self.protocol()
        self.groups[0]['cases'][1]['owner'] = 'team-b'
        self.groups[0]['cases'][1]['inputs']['input_ids'] = self.groups[0]['cases'][0]['inputs']['input_ids'].clone()
        with self.assertRaises(ValueError):
            self.protocol()

    def test_single_row_overlap_is_rejected_in_multirow_requests(self):
        self.binding.update({'input_shape': [2, 3], 'approved_rows': [4], 'max_batch_rows': 4,
                             'max_token_positions': 12, 'max_logical_input_bytes': 96,
                             'max_logical_output_bytes': 48, 'max_logical_total_bytes': 144})
        self.groups[0]['cases'][0]['inputs']['input_ids'] = torch.tensor([[1, 2, 3], [7, 8, 9]])
        self.groups[0]['cases'][1]['inputs']['input_ids'] = torch.tensor([[3, 4, 5], [10, 11, 12]])
        discovery = {'input_ids': torch.tensor([[1, 2, 3], [20, 21, 22]])}
        hashes = self.validator.input_row_digests(self.binding, discovery)
        with self.assertRaises(ValueError):
            self.validator.freeze_protocol(self.binding, self.groups, discovery_row_hashes=hashes)

    def test_row_hashes_ignore_input_schema_order(self):
        self.binding.update({'input_schema': [['input_ids', 'int64'], ['attention_mask', 'int64']],
                             'max_logical_input_bytes': 96, 'max_logical_total_bytes': 120})
        for case in self.groups[0]['cases']:
            case['inputs']['attention_mask'] = torch.ones((1, 3), dtype=torch.int64)
        overlapping = self.groups[0]['cases'][0]['inputs']
        hashes = self.validator.input_row_digests(self.binding, overlapping)
        reordered = copy.deepcopy(self.binding)
        reordered['input_schema'].reverse()
        self.assertEqual(hashes, self.validator.input_row_digests(reordered, overlapping))
        with self.assertRaises(ValueError):
            self.validator.freeze_protocol(reordered, self.groups, discovery_row_hashes=hashes)

    def test_cancel_before_or_during_execution_cannot_pass(self):
        cancel = threading.Event()
        cancel.set()
        report = self.run_protocol(self.protocol(), self.binding, self.groups,
                                   self.separable, cancel_event=cancel)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertFalse(report['complete'])
        self.assertEqual(report['calls'], {'serial': 0, 'mixed': 0})
        self.assertEqual(report['error']['kind'], 'cancelled')
        cancel.clear()
        def cancel_after_first(inputs, *, request_id, seed):
            cancel.set()
            return self.separable(inputs, request_id=request_id, seed=seed)
        report = self.run_protocol(self.protocol(), self.binding, self.groups,
                                   cancel_after_first, cancel_event=cancel)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertFalse(report['complete'])
        self.assertEqual(report['calls'], {'serial': 1, 'mixed': 0})
        self.assertEqual(report['error']['kind'], 'cancelled')

    def test_cancel_during_comparison_cannot_publish_a_pass(self):
        cancel = threading.Event()
        compare = self.validator._comparison
        def cancel_while_comparing(*args, **kwargs):
            result = compare(*args, **kwargs)
            cancel.set()
            return result
        with patch.object(self.validator, '_comparison', side_effect=cancel_while_comparing):
            report = self.run_protocol(self.protocol(), self.binding, self.groups,
                                       self.separable, cancel_event=cancel)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertFalse(report['complete'])
        self.assertEqual(report['error']['kind'], 'cancelled')

    def test_invalid_outputs_and_execution_error_are_inconclusive(self):
        invalid = [torch.zeros((1, 3, 2), dtype=torch.float32),
                   torch.zeros((1, 3, 1), dtype=torch.float64),
                   torch.full((1, 3, 1), float('nan'), dtype=torch.float32)]
        for output in invalid:
            with self.subTest(output=output):
                report = self.run_protocol(self.protocol(), self.binding, self.groups,
                    lambda inputs, *, request_id, seed: output)
                self.assertEqual(report['status'], 'inconclusive')
                self.assertFalse(report['complete'])
                self.assertEqual(len(report['groups'][0]['cases']), 2)
        def raises(inputs, *, request_id, seed):
            raise RuntimeError('executor failed')
        report = self.run_protocol(self.protocol(), self.binding, self.groups, raises)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['error']['kind'], 'phase_error')

    def test_failed_and_cancelled_reports_round_trip_but_cannot_be_admitted(self):
        protocol = self.protocol()
        def batch_mean(inputs, *, request_id, seed):
            values = inputs['input_ids'].float()
            return (values - values.mean(dim=0, keepdim=True)).unsqueeze(-1)
        cancel = threading.Event()
        cancel.set()
        reports = [self.run_protocol(protocol, self.binding, self.groups, batch_mean),
                   self.run_protocol(protocol, self.binding, self.groups, self.separable, cancel_event=cancel)]
        self.assertEqual([report['status'] for report in reports], ['failed', 'inconclusive'])
        with tempfile.TemporaryDirectory() as folder:
            for index, report in enumerate(reports):
                path = Path(folder) / f'evidence-{index}.json'
                digest = self.validator.save_evidence(path, report, protocol=protocol,
                                                      expected_protocol_sha256=protocol['protocol_sha256'])
                loaded = self.validator.load_evidence(path, protocol=protocol,
                    expected_protocol_sha256=protocol['protocol_sha256'], evidence_sha256=digest)
                self.assertEqual(loaded, report)
                with self.assertRaises(ValueError):
                    self.validator.require_evidence(loaded, protocol=protocol, actual_binding=self.binding,
                        expected_protocol_sha256=protocol['protocol_sha256'], evidence_sha256=digest, rows=2)

    def test_saved_report_checks_protocol_and_externally_retained_digest(self):
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            digest = self.validator.save_evidence(path, report, protocol=protocol,
                                                  expected_protocol_sha256=protocol['protocol_sha256'])
            loaded = self.validator.load_evidence(path, protocol=protocol,
                                                  expected_protocol_sha256=protocol['protocol_sha256'], evidence_sha256=digest)
            self.assertEqual(loaded, report)
            with self.assertRaises(FileExistsError):
                self.validator.save_evidence(path, report, protocol=protocol,
                                             expected_protocol_sha256=protocol['protocol_sha256'])
            altered = json.loads(path.read_text(encoding='utf-8'))
            altered['status'] = 'failed'
            path.write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.validator.load_evidence(path, protocol=protocol,
                                             expected_protocol_sha256=protocol['protocol_sha256'], evidence_sha256=digest)

    def test_self_hashed_empty_pass_cannot_satisfy_frozen_nonempty_protocol(self):
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        report['groups'] = []
        report['calls'] = {'serial': 0, 'mixed': 0}
        unsigned = {key: value for key, value in report.items() if key != 'evidence_sha256'}
        report['evidence_sha256'] = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'forged.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.validator.load_evidence(path, protocol=protocol,
                                             expected_protocol_sha256=protocol['protocol_sha256'],
                                             evidence_sha256=report['evidence_sha256'])

    def test_evidence_rejects_inconsistent_zero_tolerance_metrics(self):
        protocol = self.protocol()
        report = self.run_protocol(protocol, self.binding, self.groups, self.separable)
        report['groups'][0]['cases'][0]['max_abs_error'] = 1.0
        report['groups'][0]['cases'][0]['max_relative_error'] = 1.0
        unsigned = {key: value for key, value in report.items() if key != 'evidence_sha256'}
        report['evidence_sha256'] = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'forged.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.validator.load_evidence(path, protocol=protocol,
                                             expected_protocol_sha256=protocol['protocol_sha256'],
                                             evidence_sha256=report['evidence_sha256'])


if __name__ == '__main__':
    unittest.main()
