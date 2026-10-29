"""CPU checks for service-owned sampled batch-admission evidence."""

from dataclasses import replace
import unittest

import torch

from looking_glass import batch_validation as validation
from looking_glass.microbatch import BatchProfile, ForwardContract
from looking_glass.resident_worker import ForwardRequest


class CertifiedBatchPolicyTests(unittest.TestCase):
    def fixture(self, *, groups_count=2, coupled=False, repeated=False):
        from looking_glass.certified_batch_policy import CertifiedBatchPolicy, runtime_fingerprint
        runtime = {'fixture': 'cpu-runtime-v1', 'mode': 'eager'}
        contract = ForwardContract('model', 'checkpoint', 'adapter', 'cpu', torch.float32,
                                   runtime_id='sha256:' + runtime_fingerprint(runtime), seed=17, deterministic=True,
                                   batch_separable=True, mode='eval', rng_policy='none',
                                   cache_policy='disabled', mutable_state_policy='none',
                                   tracing=False, intervention=False)
        profile = BatchProfile('profile', 'cert', contract, (('input_ids', torch.int64),), (1, 3),
                               torch.float32, 1, 0.0, 0.0, (2, 4), 4, 1, 4, 12, 96, 48, 144)
        identity = {'model_fingerprint': 'a' * 64, 'checkpoint_sha256': 'b' * 64,
                    'adapter_sha256': 'c' * 64, 'backend': 'cpu-eager'}
        binding = validation.binding_from_profile(profile, **identity)
        groups = []
        for group in range(groups_count):
            index = 0 if repeated else group * 5
            cases = [{'case_id': f'{group}-a', 'owner': 'team-a',
                      'inputs': {'input_ids': torch.tensor([[index + 1, index + 2, index + 3]])}},
                     {'case_id': f'{group}-b', 'owner': 'team-b',
                      'inputs': {'input_ids': torch.tensor([[index + 2, index + 3, index + 4]])}}]
            groups.append({'group_id': f'g-{group}', 'cases': cases})
        discovery = {'input_ids': torch.tensor([[90, 91, 92]])}
        protocol = validation.freeze_protocol(binding, groups,
            discovery_row_hashes=validation.input_row_digests(binding, discovery))
        def execute(inputs, *, request_id, seed):
            values = inputs['input_ids'].to(torch.float32)
            if coupled:
                values = values - values.mean(dim=0, keepdim=True)
            return values.unsqueeze(-1)
        report = validation.run_validation(protocol, binding, groups, execute,
                                           expected_protocol_sha256=protocol['protocol_sha256'])
        options = {'expected_protocol_sha256': protocol['protocol_sha256'],
                   'evidence_sha256': report['evidence_sha256'], 'minimum_groups': 2,
                   'max_atol': 0.0, 'max_rtol': 0.0,
                   'runtime_manifest': runtime, **identity}
        return CertifiedBatchPolicy, profile, protocol, report, options

    def test_passing_independent_evidence_authorizes_only_observed_size(self):
        cls, profile, protocol, report, options = self.fixture()
        policy = cls(profile, protocol, report, **options)
        self.assertEqual(policy.allowed_request_counts, (2,))
        self.assertEqual(policy.tested_rows, (2,))
        self.assertEqual(len(policy.compatibility_key), 32)
        self.assertEqual(policy.coverage(2), (2, 4, 2))
        policy.require_count(2)
        with self.assertRaises(ValueError):
            policy.require_count(4)

    def test_each_row_count_needs_its_own_coverage(self):
        cls, profile, _, _, options = self.fixture()
        identity = {name: options[name] for name in ('model_fingerprint', 'checkpoint_sha256',
                                                     'adapter_sha256', 'backend')}
        binding = validation.binding_from_profile(profile, **identity)
        groups = []
        for group_index, size in enumerate((2, 2, 4)):
            cases = []
            for row in range(size):
                offset = 20 * group_index + 3 * row
                cases.append({'case_id': f'{group_index}-{row}', 'owner': f'team-{row}',
                              'inputs': {'input_ids': torch.tensor([[offset, offset + 1, offset + 2]])}})
            groups.append({'group_id': f'g-{group_index}', 'cases': cases})
        discovery = {'input_ids': torch.tensor([[90, 91, 92]])}
        protocol = validation.freeze_protocol(binding, groups,
            discovery_row_hashes=validation.input_row_digests(binding, discovery))
        def execute(inputs, *, request_id, seed):
            return inputs['input_ids'].float().unsqueeze(-1)
        report = validation.run_validation(protocol, binding, groups, execute,
                                           expected_protocol_sha256=protocol['protocol_sha256'])
        options.update(expected_protocol_sha256=protocol['protocol_sha256'],
                       evidence_sha256=report['evidence_sha256'])
        self.assertEqual(report['tested_rows'], [2, 4])
        policy = cls(profile, protocol, report, **options)
        self.assertEqual(policy.tested_rows, (2,))
        with self.assertRaises(ValueError):
            policy.require_count(4)

    def test_incomplete_or_repeated_groups_do_not_satisfy_minimum(self):
        for kwargs in ({'groups_count': 1}, {'repeated': True}):
            cls, profile, protocol, report, options = self.fixture(**kwargs)
            self.assertEqual(report['status'], 'passed')
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cls(profile, protocol, report, **options)

    def test_failed_coupled_model_or_tampered_evidence_is_rejected(self):
        cls, profile, protocol, report, options = self.fixture(coupled=True)
        self.assertEqual(report['status'], 'failed')
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **options)
        cls, profile, protocol, report, options = self.fixture()
        report['status'] = 'failed'
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **options)
        report['status'] = 'passed'
        options['expected_protocol_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **options)

    def test_thresholds_and_external_digest_validation_fail_closed(self):
        cls, profile, protocol, report, options = self.fixture()
        for value in (True, 0, 1, 65, 1.5):
            with self.subTest(minimum_groups=value), self.assertRaises(ValueError):
                cls(profile, protocol, report, **{**options, 'minimum_groups': value})
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **{**options, 'evidence_sha256': options['expected_protocol_sha256']})
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **{**options, 'evidence_sha256': 'd' * 64})
        for limit in (True, -1, float('inf'), 10**1000):
            with self.subTest(max_atol=repr(limit)[:40]), self.assertRaises(ValueError):
                cls(profile, protocol, report, **{**options, 'max_atol': limit})
        lax_profile = replace(profile, serial_atol=1e-4)
        with self.assertRaisesRegex(ValueError, 'exceed the frozen service limits'):
            cls(lax_profile, protocol, report, **options)

    def test_runtime_digest_uses_a_bounded_exact_json_manifest(self):
        from looking_glass.certified_batch_policy import runtime_fingerprint
        self.assertEqual(runtime_fingerprint({'b': 2, 'a': 1}), runtime_fingerprint({'a': 1, 'b': 2}))
        for bad in ({}, {1: 'value'}, {'nested': {1: 'value'}}, {'nan': float('nan')},
                    {'large': 'x' * (64 * 1024)}):
            with self.subTest(bad=repr(bad)[:100]), self.assertRaises(ValueError):
                runtime_fingerprint(bad)

    def test_live_profile_binding_and_cpu_device_cannot_drift(self):
        cls, profile, protocol, report, options = self.fixture()
        policy = cls(profile, protocol, report, **options)
        identity = {name: options[name] for name in ('model_fingerprint', 'checkpoint_sha256',
                                                     'adapter_sha256', 'backend', 'runtime_manifest')}
        policy.require_live_binding(profile, **identity)
        with self.assertRaises(ValueError):
            policy.require_live_binding(replace(profile, serial_atol=1e-6), **identity)
        cuda_profile = replace(profile, contract=replace(profile.contract, device='cuda:0'))
        with self.assertRaises(ValueError):
            policy.require_live_binding(cuda_profile, **identity)
        with self.assertRaises(ValueError):
            policy.require_live_binding(profile, **{**identity, 'runtime_manifest': {'mode': 'changed'}})
        with self.assertRaises(ValueError):
            cls(profile, protocol, report, **{**options, 'runtime_manifest': {'mode': 'changed'}})

    def test_request_checks_exact_shape_schema_dtype_seed_and_no_grad(self):
        cls, profile, protocol, report, options = self.fixture()
        policy = cls(profile, protocol, report, **options)
        good = ForwardRequest('request', {'input_ids': torch.tensor([[1, 2, 3]])}, seed=17)
        policy.require_request(good)
        variants = [replace(good, seed=16),
                    replace(good, request_id='x' * 257),
                    replace(good, inputs={'input_ids': torch.ones((1, 4), dtype=torch.int64)}),
                    replace(good, inputs={'input_ids': torch.ones((1, 3), dtype=torch.int32)}),
                    replace(good, inputs={'input_ids': torch.ones((1, 3), dtype=torch.int64),
                                          'attention_mask': torch.ones((1, 3), dtype=torch.int64)})]
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                policy.require_request(bad)
        with self.assertRaises(ValueError):
            policy.require_count(True)
        with self.assertRaises(ValueError):
            policy.coverage(True)

    def test_policy_snapshots_evidence_and_does_not_trust_mutating_callers(self):
        cls, profile, protocol, report, options = self.fixture()
        policy = cls(profile, protocol, report, **options)
        key = policy.compatibility_key
        protocol['binding']['seed'] = 99
        report['status'] = 'failed'
        options['model_fingerprint'] = 'e' * 64
        self.assertEqual(policy.allowed_request_counts, (2,))
        self.assertEqual(policy.compatibility_key, key)
        policy.require_count(2)


if __name__ == '__main__':
    unittest.main()
