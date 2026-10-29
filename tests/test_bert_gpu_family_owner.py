"""CUDA-hidden tests for profile-owner resource and state primitives."""

import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch

from looking_glass.bert_gpu_family_owner import (_capture_state, _reserve_family_budget,
                                                  _verify_state, _HOST_CLAIM, _GPU_CLAIM,
                                                  measure_family_owned)
from looking_glass.memory_budget import CudaMemorySnapshot


class FamilyOwnerPrimitiveTests(unittest.TestCase):
    def test_shared_disabled_gate_stops_direct_owner_before_session_or_device(self):
        import looking_glass.bert_gpu_family_owner as subject
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(subject, 'FAMILY_PROFILE_ENABLED', False), \
             patch.object(subject, '_verify_startup') as startup, \
             patch.object(subject.OwnerSession, 'from_environment') as session, \
             patch.object(subject.DeviceLease, 'acquire') as lease, \
             patch.object(subject, '_import_torch') as load_torch:
            self.assertEqual(subject.owner_main(directory, directory, '1'*32, '2'*40, 'unused'), 1)
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                subject.measure_family_owned(None, None, None, directory, {},
                    journal_path='unused', record_memory=lambda *args: None)
            startup.assert_not_called()
            session.assert_not_called()
            lease.assert_not_called()
            load_torch.assert_not_called()

    def test_family_claims_cover_gpu_and_host_until_explicit_release(self):
        gib = 1024 ** 3
        telemetry = lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0)
        custody = {'budget': None, 'memory_lease': None}
        budget, lease = _reserve_family_budget(telemetry, cap=2 * gib, custody=custody)
        self.assertIs(custody['budget'], budget)
        self.assertIs(custody['memory_lease'], lease)
        self.assertIn(lease, budget.active_leases())
        lease.release_trace('gpu', _GPU_CLAIM)
        lease.release_trace('host', _HOST_CLAIM)
        lease.release()
        self.assertEqual(len(budget.active_leases()), 0)

    def test_state_witness_detects_parameter_buffer_and_mode_changes(self):
        model = torch.nn.Linear(3, 2).eval()
        model.register_buffer('persistent_test', torch.tensor([1., 2.]))
        model.register_buffer('transient_test', torch.tensor([3.]), persistent=False)
        witness = _capture_state(model, torch, device='cpu', check_owner=lambda: None,
                                 synchronize=lambda: None)
        self.assertTrue(_verify_state(model, witness, torch, device='cpu',
                                      check_owner=lambda: None, synchronize=lambda: None))
        model.transient_test[0] = 4.
        with self.assertRaisesRegex(RuntimeError, 'model state'):
            _verify_state(model, witness, torch, device='cpu', check_owner=lambda: None,
                          synchronize=lambda: None)
        model.transient_test[0] = 3.
        model.train()
        with self.assertRaisesRegex(RuntimeError, 'model state'):
            _verify_state(model, witness, torch, device='cpu', check_owner=lambda: None,
                          synchronize=lambda: None)

    def test_owner_revocation_prevents_later_state_copy(self):
        model = torch.nn.Linear(3, 2).eval()
        calls = []
        def revoke():
            calls.append('check')
            raise RuntimeError('lease revoked')
        with self.assertRaisesRegex(RuntimeError, 'lease revoked'):
            _capture_state(model, torch, device='cpu', check_owner=revoke,
                           synchronize=lambda: calls.append('sync'))
        self.assertEqual(calls, ['check'])

    def test_allocator_or_device_total_mismatch_stops_before_model_load(self):
        import looking_glass.bert_gpu_family_owner as subject
        gib = 1024 ** 3
        class FakeCuda:
            def __init__(self, backend): self.backend = backend
            def is_initialized(self): return False
            def device_count(self): return 1
            def get_device_properties(self, index):
                return SimpleNamespace(uuid=object(), total_memory=8 * gib)
            def get_allocator_backend(self): return self.backend
            def set_per_process_memory_fraction(self, fraction, index): pass
            def get_per_process_memory_fraction(self, index): return .25
            def reset_peak_memory_stats(self, index): pass
        session, lease = object.__new__(subject.OwnerSession), object.__new__(subject.DeviceLease)
        snapshot = {'device_free_bytes': 6 * gib, 'device_total_bytes': 9 * gib,
                    'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0,
                    'peak_reserved_bytes': 0}
        for backend, message in (('cudaMallocAsync', 'allocator'), ('native', 'totals disagree')):
            with self.subTest(backend=backend), patch.object(subject.OwnerSession, 'check'), \
                 patch.object(subject.OwnerSession, 'verify_runtime_binding'), \
                 patch.object(subject, '_runtime_versions', return_value={}), \
                 patch.object(subject, '_runtime_uuid', return_value='GPU-fixed'), \
                 patch.object(subject, '_guarded_snapshot', return_value=snapshot), \
                 patch.object(subject, '_load_family_model') as load:
                with self.assertRaisesRegex(RuntimeError, message):
                    measure_family_owned(session, lease, SimpleNamespace(cuda=FakeCuda(backend)),
                        'unused', {'rows': []}, journal_path='unused', record_memory=lambda *args: None)
                load.assert_not_called()

    def test_child_roots_first_failure_and_does_not_attest_clean(self):
        import looking_glass.bert_gpu_family_owner as subject
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(device_lease_path='lease', run_id='supervised', identity=object())
            device_lease = object()
            prior = dict(subject._CUSTODY)
            subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
            try:
                with patch.object(subject, '_verify_startup'), patch.object(subject, '_require_pristine_dependencies'), \
                     patch.object(subject.OwnerSession, 'from_environment', return_value=session), \
                     patch.object(subject.OwnerSession, 'check'), \
                     patch.object(subject.OwnerSession, 'attest_clean') as attest, \
                     patch.object(subject.DeviceLease, 'acquire', return_value=device_lease), \
                     patch.object(subject, '_verify_source', return_value={}), \
                     patch.object(subject, '_load_manifest', return_value=({'rows': []}, 'a' * 64)), \
                     patch.object(subject, '_import_torch', side_effect=MemoryError('injected')):
                    result = subject.owner_main(directory, directory, '1' * 32, '2' * 40, 'unused')
                self.assertEqual(result, 1)
                self.assertIs(subject._CUSTODY['session'], session)
                self.assertIs(subject._CUSTODY['device_lease'], device_lease)
                self.assertIsInstance(subject._CUSTODY['failure'], MemoryError)
                self.assertEqual(json.loads((Path(directory) / 'failure.json').read_bytes())['status'], 'quarantined')
                attest.assert_not_called()
            finally:
                subject._CUSTODY.update(prior)

    def test_child_propagates_interrupt_and_roots_a_fresh_diagnostic_interrupt(self):
        import looking_glass.bert_gpu_family_owner as subject
        with tempfile.TemporaryDirectory() as directory:
            prior = dict(subject._CUSTODY)
            try:
                for interrupt in (KeyboardInterrupt('stop'), SystemExit('stop')):
                    with self.subTest(kind=type(interrupt).__name__):
                        subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
                        path = Path(directory) / type(interrupt).__name__
                        path.mkdir()
                        with patch.object(subject, '_verify_startup', side_effect=interrupt):
                            with self.assertRaises(type(interrupt)):
                                subject.owner_main(path, directory, '1' * 32, '2' * 40, 'unused')
                        self.assertIs(subject._CUSTODY['failure'], interrupt)
                        self.assertEqual(json.loads((path / 'failure.json').read_bytes())['failure_kind'], 'interrupted')
                subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
                original, second = RuntimeError('first failure'), KeyboardInterrupt('diagnostic stop')
                with patch.object(subject, '_verify_startup', side_effect=original), \
                     patch.object(subject, 'write_json_new', side_effect=second):
                    with self.assertRaises(KeyboardInterrupt):
                        subject.owner_main(directory, directory, '1' * 32, '2' * 40, 'unused')
                self.assertIs(subject._CUSTODY['failure'], original)
                self.assertIs(subject._CUSTODY['diagnostic_failure'], second)
            finally:
                subject._CUSTODY.update(prior)

    def test_child_writes_final_only_after_candidate_and_budget_release(self):
        import looking_glass.bert_gpu_family_owner as subject
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(device_lease_path='lease', run_id='supervised', identity=object())
            device_lease = object()
            prior = dict(subject._CUSTODY)
            subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
            def fake_measure(*args, record_memory, **kwargs):
                for index in range(287):
                    record_memory('fake', index, {'allocated_bytes': 0})
                subject._CUSTODY['profile'] = {'temporary': True}
                return {'profile': {'forward_count': 140}}
            def release(got_session, got_lease):
                self.assertIs(got_session, session)
                self.assertIs(got_lease, device_lease)
                self.assertTrue((Path(directory) / 'candidate.json').is_file())
                self.assertFalse((Path(directory) / 'final.json').exists())
                self.assertIsNone(subject._CUSTODY['profile'])
            try:
                with patch.object(subject, '_verify_startup'), patch.object(subject, '_require_pristine_dependencies'), \
                     patch.object(subject.OwnerSession, 'from_environment', return_value=session), \
                     patch.object(subject.OwnerSession, 'check'), \
                     patch.object(subject.OwnerSession, 'attest_clean') as attest, \
                     patch.object(subject.DeviceLease, 'acquire', return_value=device_lease), \
                     patch.object(subject, '_verify_source', return_value={}), \
                     patch.object(subject, '_load_manifest', return_value=({'rows': []}, 'a' * 64)), \
                     patch.object(subject, '_import_torch', return_value=object()), \
                     patch.object(subject, 'measure_family_owned', side_effect=fake_measure), \
                     patch.object(subject, '_check_family_custody'), \
                     patch.object(subject, '_release_family_budget', side_effect=release):
                    result = subject.owner_main(directory, directory, '1' * 32, '2' * 40, 'unused')
                self.assertEqual(result, 0)
                self.assertEqual(json.loads((Path(directory) / 'final.json').read_bytes())['active_memory_leases'], 0)
                attest.assert_called_once_with(session, device_lease)
            finally:
                subject._CUSTODY.update(prior)

    def test_session_loss_after_candidate_keeps_all_claims_and_quarantines(self):
        import looking_glass.bert_gpu_family_owner as subject
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(device_lease_path='lease', run_id='supervised', identity=object())
            device_lease = object()
            prior = dict(subject._CUSTODY)
            subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
            lost = [False]
            real_write = subject.write_json_new
            def check(*args):
                if lost[0]:
                    raise RuntimeError('injected supervisor loss')
            def write(path, value):
                result = real_write(path, value)
                if Path(path).name == 'candidate.json':
                    lost[0] = True
                return result
            def fake_measure(*args, record_memory, **kwargs):
                cap = 2 * gib
                subject._CUSTODY['cap'], subject._CUSTODY['commitment'] = cap, cap + subject._HEADROOM
                subject._reserve_family_budget(lambda: CudaMemorySnapshot(6*gib, 8*gib, 0, 0),
                                               cap=cap, custody=subject._CUSTODY)
                for index in range(287):
                    record_memory('fake', index, {'allocated_bytes': 0})
                return {'profile': {'forward_count': 140}}
            try:
                with patch.object(subject, '_verify_startup'), patch.object(subject, '_require_pristine_dependencies'), \
                     patch.object(subject.OwnerSession, 'from_environment', return_value=session), \
                     patch.object(subject.OwnerSession, 'check', side_effect=check), \
                     patch.object(subject.OwnerSession, 'attest_clean') as attest, \
                     patch.object(subject.DeviceLease, 'acquire', return_value=device_lease), \
                     patch.object(subject, '_verify_source', return_value={}), \
                     patch.object(subject, '_load_manifest', return_value=({'rows': []}, 'a' * 64)), \
                     patch.object(subject, '_import_torch', return_value=object()), \
                     patch.object(subject, 'measure_family_owned', side_effect=fake_measure), \
                     patch.object(subject, 'write_json_new', side_effect=write):
                    result = subject.owner_main(directory, directory, '1' * 32, '2' * 40, 'unused')
                self.assertEqual(result, 1)
                budget, memory_lease = subject._CUSTODY['budget'], subject._CUSTODY['memory_lease']
                self.assertEqual(budget.active_leases(), (memory_lease,))
                self.assertEqual(memory_lease.trace_bytes('gpu'), _GPU_CLAIM)
                self.assertEqual(memory_lease.trace_bytes('host'), _HOST_CLAIM)
                self.assertTrue((Path(directory) / 'candidate.json').is_file())
                self.assertFalse((Path(directory) / 'final.json').exists())
                self.assertEqual(json.loads((Path(directory) / 'failure.json').read_bytes())['status'], 'quarantined')
                attest.assert_not_called()
            finally:
                subject._CUSTODY.update(prior)

    def test_session_loss_between_claim_releases_retains_remaining_custody(self):
        import looking_glass.bert_gpu_family_owner as subject
        gib = 1024 ** 3
        session, device_lease = object(), object()
        prior = dict(subject._CUSTODY)
        subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
        subject._CUSTODY.update(session=session, device_lease=device_lease,
                                cap=2*gib, commitment=2*gib+subject._HEADROOM)
        try:
            budget, memory_lease = subject._reserve_family_budget(
                lambda: CudaMemorySnapshot(6*gib, 8*gib, 0, 0), cap=2*gib, custody=subject._CUSTODY)
            checks = [0]
            def check(*args):
                checks[0] += 1
                if checks[0] == 3:
                    raise RuntimeError('injected supervisor loss')
            with patch.object(subject.OwnerSession, 'check', side_effect=check):
                with self.assertRaisesRegex(RuntimeError, 'supervisor loss'):
                    subject._release_family_budget(session, device_lease)
            self.assertEqual(memory_lease.trace_bytes('gpu'), 0)
            self.assertEqual(memory_lease.trace_bytes('host'), _HOST_CLAIM)
            self.assertEqual(budget.active_leases(), (memory_lease,))
        finally:
            subject._CUSTODY.update(prior)

    def test_fake_device_owner_holds_claims_through_all_endpoints(self):
        import looking_glass.bert_gpu_family_owner as subject
        gib, mib = 1024 ** 3, 1024 ** 2
        class FakeCuda:
            def __init__(self):
                self.total, self.allocated, self.reserved, self.peak = 8 * gib, 0, 0, 0
                self.low_free, self.calls = False, []
            def is_initialized(self): return False
            def device_count(self): return 1
            def get_allocator_backend(self): return 'native'
            def get_device_properties(self, index):
                return SimpleNamespace(uuid=object(), total_memory=self.total, name='fake', major=7, minor=5)
            def set_per_process_memory_fraction(self, fraction, index): self.fraction = fraction
            def get_per_process_memory_fraction(self, index): return self.fraction
            def reset_peak_memory_stats(self, index): self.peak = self.reserved
            def synchronize(self, index): self.calls.append('synchronize')
            def mem_get_info(self, index):
                self.calls.append('mem_get_info')
                return (0 if self.low_free else 6 * gib - self.reserved), self.total
            def memory_allocated(self, index): return self.allocated
            def memory_reserved(self, index): return self.reserved
            def max_memory_allocated(self, index): return self.peak
            def max_memory_reserved(self, index): return self.peak
            def manual_seed(self, seed): pass
            def empty_cache(self):
                if self.allocated == 0: self.reserved = 0
            def allocate(self, count):
                self.allocated += count
                self.reserved = max(self.reserved, self.allocated)
                self.peak = max(self.peak, self.reserved)
        cuda = FakeCuda()
        class Device:
            def __str__(self): return 'cuda:0'
        class FakeModel:
            def __init__(self):
                self.weight = SimpleNamespace(device=Device(), dtype='float32', numel=lambda: 66985530)
                self.vocab_projector = SimpleNamespace(weight=self.weight)
                self.training, self.bytes = False, 0
            def to(self, device):
                self.bytes = 256 * mib
                cuda.allocate(self.bytes)
                return self
            def parameters(self): return iter((self.weight,))
            def buffers(self): return iter(())
            def get_input_embeddings(self): return SimpleNamespace(weight=self.weight)
            def __del__(self): cuda.allocated -= self.bytes
        fake_torch = SimpleNamespace(cuda=cuda, float32='float32', __version__='2.8.0+cu128',
            version=SimpleNamespace(cuda='12.8'),
            __future__=SimpleNamespace(get_overwrite_module_params_on_conversion=lambda: False,
                                       get_swap_module_params_on_conversion=lambda: False),
            random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                     cudnn=SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=True)),
            set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None,
            get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
            use_deterministic_algorithms=lambda value: None,
            are_deterministic_algorithms_enabled=lambda: True,
            _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: None))
        session, lease = object.__new__(subject.OwnerSession), object.__new__(subject.DeviceLease)
        prior = dict(subject._CUSTODY)
        subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
        subject._CUSTODY.update(session=session, device_lease=lease)
        stages, observations, mode, stopped_at = [], [], ['normal'], [None]
        def record(stage, index, data):
            stages.append((stage, index))
            observations.append(data)
        def execute(model, manifest, *, checkpoint, **kwargs):
            for number in range(1, 141):
                if number == 2 and mode[0] == 'lost_claims':
                    owned = subject._CUSTODY['memory_lease']
                    owned.release_trace('gpu', _GPU_CLAIM)
                    owned.release_trace('host', _HOST_CLAIM)
                    owned.release()
                    stopped_at[0] = len(cuda.calls)
                if number == 2 and mode[0] == 'low_headroom':
                    cuda.low_free = True
                checkpoint('before', number)
                checkpoint('after', number)
            return {'forward_count': 140}
        def run():
            with patch.object(subject.OwnerSession, 'check'), \
                 patch.object(subject.OwnerSession, 'verify_runtime_binding'), \
                 patch.object(subject, '_runtime_versions', return_value={}), \
                 patch.object(subject, '_runtime_uuid', return_value='GPU-fixed'), \
                 patch.object(subject, '_load_family_model', new=lambda *args: (FakeModel(), {})), \
                 patch.object(subject, '_capture_state', new=lambda *args, **kwargs: SimpleNamespace(sha256='a' * 64)), \
                 patch.object(subject, '_verify_state', new=lambda *args, **kwargs: True), \
                 patch('looking_glass.bert_agreement_runner._has_hooks', new=lambda model: False), \
                 patch('looking_glass.bert_gpu_family_profile.execute_family_profile', new=execute):
                return measure_family_owned(session, lease, fake_torch, 'unused', {'rows': []},
                    journal_path='unused', record_memory=record)
        try:
            result = run()
            self.assertEqual(len(stages), 287)
            self.assertEqual(stages[4:8], [('before', 1), ('after', 1), ('before', 2), ('after', 2)])
            self.assertEqual(result['cleanup']['allocated_bytes'], 0)
            self.assertEqual(result['cleanup']['reserved_bytes'], 0)
            self.assertFalse(result['model_weakref_alive'])
            budget = subject._CUSTODY['budget']
            with patch.object(subject.OwnerSession, 'check'):
                self.assertEqual(budget.snapshot().host_trace_bytes, _HOST_CLAIM)
                self.assertEqual(budget.snapshot().gpu_trace_bytes, _GPU_CLAIM)
                subject._release_family_budget(session, lease)
            self.assertEqual(len(budget.active_leases()), 0)

            for scenario in ('low_headroom', 'lost_claims'):
                subject._CUSTODY.update({key: False if key == 'entered' else None for key in prior})
                subject._CUSTODY.update(session=session, device_lease=lease)
                stages.clear()
                observations.clear()
                cuda.low_free = False
                mode[0] = scenario
                if scenario == 'low_headroom':
                    with self.assertRaisesRegex(RuntimeError, 'insufficient device reserve'):
                        run()
                    self.assertEqual(stages[-1], ('before', 2))
                    self.assertEqual(observations[-1]['device_free_bytes'], 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'custody was lost'):
                        run()
                    self.assertEqual(stages[-1], ('after', 1))
                    self.assertEqual(len(cuda.calls), stopped_at[0])
                subject._CUSTODY['model'] = subject._CUSTODY['converted'] = None
                subject._CUSTODY['witness'] = None
                import gc
                gc.collect()
                cuda.allocated = cuda.reserved = cuda.peak = 0
        finally:
            subject._CUSTODY.update(prior)


if __name__ == '__main__':
    unittest.main()
