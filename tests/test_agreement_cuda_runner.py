"""CPU-only guards and failure semantics; this file does not initialize CUDA."""
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import sys
import tempfile
import weakref
from contextlib import ExitStack
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from looking_glass import resource_profile as rp
from looking_glass.agreement_cuda_runner import (CALIBRATION_KINDS, CAPTURE_POLICY_SHA256,
    CALIBRATION_INPUT_IDS, CALIBRATION_DONOR_IDS, OUTPUT_BYTES, TRACE_BYTES, WARMUP_POLICY_SHA256,
    CudaAgreementRunner, GuardedAgreementMemoryProbe, _validate_profile,
    _validate_warmup_profile, calibration_case_sha256)
from looking_glass.agreement_patch_runner import AgreementCase, NaturalAgreementRunner, StudyFailure
from looking_glass.memory_budget import CudaMemorySnapshot
from looking_glass.pretrained import ASSETS, REVISION


GIB = 1024 ** 3


def workload(**changes):
    fields = {'checkpoint_sha256': ASSETS['model.safetensors'][1], 'checkpoint_revision': REVISION,
              'model_config_sha256': 'a' * 64, 'source_sha256': 'b' * 64,
              'runtime_settings_sha256': 'c' * 64, 'torch_version': '2.8.0+cu128',
              'transformers_version': '4.57.3', 'cuda_runtime_version': '12.8',
              'driver_version': '570.0', 'python_version': '3.11.15', 'device': 'cuda:0',
              'device_uuid': 'GPU-00000000-0000-0000-0000-000000000001', 'device_name': 'test GPU',
              'compute_capability': (7, 5), 'device_total_bytes': 8 * GIB,
              'model_dtype': 'torch.float32', 'compute_dtype': 'torch.float32',
              'attention_backend': 'eager', 'model_mode': 'eval', 'grad_mode': 'inference',
              'autocast': 'disabled', 'use_cache': False, 'compiled': False, 'warmup_state': 'warm',
              'warmup_protocol_sha256': WARMUP_POLICY_SHA256, 'operation': 'forward', 'capture_mode': 'direct',
              'capture_policy_sha256': CAPTURE_POLICY_SHA256, 'batch_size': 1, 'sequence_length': 5,
              'input_dtypes': (('attention_mask', 'torch.int64'), ('input_ids', 'torch.int64')),
              'output_limit_bytes': OUTPUT_BYTES, 'max_gpu_trace_bytes': TRACE_BYTES,
              'max_host_trace_bytes': TRACE_BYTES}
    fields.update(changes)
    fields['model_id'] = rp.model_runtime_fingerprint(fields)
    return rp.WorkloadKey(**fields)


def profile(key, *, kinds=CALIBRATION_KINDS, seed=20261028, margin_bytes=512 * 1024 ** 2, margin_bps=2000):
    before = CudaMemorySnapshot(6 * GIB, 8 * GIB, 400 * 1024 ** 2, 500 * 1024 ** 2)
    after = CudaMemorySnapshot(6 * GIB, 8 * GIB, 410 * 1024 ** 2, 520 * 1024 ** 2)
    samples = tuple(rp.MemorySample(key.fingerprint(), kind, calibration_case_sha256(kind), seed,
                                    'completed', before, after, 450 * 1024 ** 2, 540 * 1024 ** 2)
                    for kind in kinds)
    evidence = rp.encode_evidence(key, samples)
    provenance = rp.ProfileProvenance('resource-profile-v1', 'e' * 64,
        hashlib.sha256(evidence).hexdigest(), '2026-10-28T00:00:00Z',
        'cooperative-exclusive', 'synchronize-reset-synchronize')
    return rp.MeasuredResourceProfile.build(key, samples, provenance,
                                            margin_bytes=margin_bytes, margin_bps=margin_bps)


def cold_profile(key, *, seed=20261028, case_sha256=WARMUP_POLICY_SHA256):
    cold = replace(key, warmup_state='cold')
    before = CudaMemorySnapshot(6 * GIB, 8 * GIB, 400 * 1024 ** 2, 500 * 1024 ** 2)
    after = CudaMemorySnapshot(6 * GIB, 8 * GIB, 410 * 1024 ** 2, 520 * 1024 ** 2)
    samples = (rp.MemorySample(cold.fingerprint(), 'warmup', case_sha256, seed, 'completed',
                               before, after, 450 * 1024 ** 2, 540 * 1024 ** 2),)
    evidence = rp.encode_evidence(cold, samples)
    provenance = rp.ProfileProvenance('resource-profile-v1', 'e' * 64,
        hashlib.sha256(evidence).hexdigest(), '2026-10-28T00:00:00Z',
        'cooperative-exclusive', 'synchronize-reset-synchronize')
    return rp.MeasuredResourceProfile.build(cold, samples, provenance,
                                            margin_bytes=512 * 1024 ** 2, margin_bps=2000)


class CudaAgreementGuardTests(unittest.TestCase):
    def test_calibration_policy_matches_reviewed_spec(self):
        from looking_glass import agreement_calibration_spec as spec
        from looking_glass.agreement_patch_runner import _RNG_SEED
        self.assertEqual(CALIBRATION_KINDS, spec.SAMPLE_KINDS)
        self.assertEqual(CALIBRATION_INPUT_IDS, spec.TARGET_INPUT_IDS)
        self.assertEqual(CALIBRATION_DONOR_IDS, spec.DONOR_INPUT_IDS)
        self.assertEqual(_RNG_SEED, spec.CALIBRATION_SEED)
        self.assertEqual(TRACE_BYTES, spec.MIN_TRACE_ALLOWANCE_BYTES)
        key = workload()
        warm = profile(key)
        spec.validate_profile_requirements(sample_kinds=CALIBRATION_KINDS, seed=_RNG_SEED,
            gpu_trace_bytes=TRACE_BYTES, host_trace_bytes=TRACE_BYTES,
            margin_bytes=warm.margin_bytes, margin_bps=warm.margin_bps)

    def test_state_and_rng_work_require_fresh_headroom_first(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._device_rng_before = torch.zeros(1, dtype=torch.uint8)
        runner._check_headroom = mock.Mock(side_effect=RuntimeError('pressure'))
        runner._sync = mock.Mock()
        with mock.patch.object(torch.cuda, 'get_rng_state') as get_rng:
            with self.assertRaisesRegex(RuntimeError, 'pressure'):
                runner._model_digest()
            # Start from a fresh fake because a failed state check is sticky.
            runner = object.__new__(CudaAgreementRunner)
            runner._device_rng_before = torch.zeros(1, dtype=torch.uint8)
            runner._check_headroom = mock.Mock(side_effect=RuntimeError('pressure'))
            with self.assertRaisesRegex(RuntimeError, 'pressure'):
                runner._seed_device_rng()
            with self.assertRaisesRegex(RuntimeError, 'pressure'):
                runner._restore_device_rng()
        get_rng.assert_not_called()

    def test_failed_rng_seed_quarantines_before_any_restore_attempt(self):
        # CPU fakes model a saved CUDA RNG state followed by a failing read.
        # Unwinding may call _restore_device_rng, but it must not touch CUDA.
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._device_rng_before = None
        def check_owner(*, allow_quarantined=False, allow_pending=False):
            if runner.quarantined and not (allow_quarantined or allow_pending):
                raise StudyFailure('owner quarantined')
        runner._check_owner = check_owner
        runner._check_headroom = check_owner
        generator = mock.Mock()
        initial = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (mock.patch.object(torch.cuda, 'get_rng_state',
                                side_effect=[initial, RuntimeError('RNG read failed')]) as get_rng,
              mock.patch.object(torch.cuda, 'default_generators', (generator,)),
              mock.patch.object(torch.cuda, 'set_rng_state') as set_rng):
            with self.assertRaisesRegex(RuntimeError, 'RNG read failed'):
                runner._seed_device_rng()
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)
            with self.assertRaises(StudyFailure):
                runner._restore_device_rng()
            self.assertEqual(get_rng.call_count, 2)
            set_rng.assert_not_called()

    def test_successful_rng_seed_clears_pending_and_restores(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._device_rng_before = None
        def check_owner(*, allow_quarantined=False, allow_pending=False):
            if runner._quarantine_latched and not allow_quarantined:
                raise StudyFailure('owner quarantined')
            if runner._device_drain_uncertain and not allow_pending:
                raise StudyFailure('device operation pending')
        runner._check_owner = check_owner
        runner._check_headroom = check_owner
        initial = torch.tensor([1, 2, 3], dtype=torch.uint8)
        seeded = torch.tensor([4, 5, 6], dtype=torch.uint8)
        generator = mock.Mock()
        with (mock.patch.object(torch.cuda, 'get_rng_state', side_effect=[initial, seeded, initial]),
              mock.patch.object(torch.cuda, 'default_generators', (generator,)),
              mock.patch.object(torch.cuda, 'set_rng_state') as set_rng):
            result = runner._seed_device_rng()
            self.assertFalse(runner.quarantined)
            self.assertFalse(runner._device_drain_uncertain)
            self.assertEqual(result['seeded_state_sha256'], hashlib.sha256(bytes([4, 5, 6])).hexdigest())
            runner._restore_device_rng()
            set_rng.assert_called_once_with(initial, 0)
        self.assertFalse(runner.quarantined)

    def test_each_rng_seed_failure_boundary_latches_before_cleanup(self):
        for fault in ('headroom', 'first_get', 'owner1', 'seed', 'owner2', 'second_get', 'owner3', 'hash'):
            with self.subTest(fault=fault):
                runner = object.__new__(CudaAgreementRunner)
                runner.quarantined = False
                runner._device_drain_uncertain = False
                runner._device_rng_before = None
                events, gets, checks = [], [0], [0]
                def fail():
                    events.append('fault')
                    raise torch.OutOfMemoryError('simulated boundary failure')
                def check_owner(*, allow_quarantined=False, allow_pending=False):
                    if runner._quarantine_latched and not allow_quarantined:
                        raise StudyFailure('owner quarantined')
                    if allow_pending:
                        checks[0] += 1
                        if fault == 'owner' + str(checks[0]):
                            fail()
                    elif runner._device_drain_uncertain:
                        raise StudyFailure('device operation pending')
                runner._check_owner = check_owner
                def check_headroom():
                    check_owner()
                    if fault == 'headroom':
                        fail()
                runner._check_headroom = check_headroom
                def get_rng(*args):
                    gets[0] += 1
                    events.append('get')
                    if fault == ('first_get' if gets[0] == 1 else 'second_get'):
                        fail()
                    if fault == 'hash' and gets[0] == 2:
                        return SimpleNamespace(numpy=fail)
                    return torch.tensor([1], dtype=torch.uint8)
                def seed(*args):
                    events.append('seed')
                    if fault == 'seed':
                        fail()
                with (mock.patch.object(torch.cuda, 'get_rng_state', side_effect=get_rng),
                      mock.patch.object(torch.cuda, 'set_rng_state', side_effect=lambda *_: events.append('set')),
                      mock.patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=seed),))):
                    with self.assertRaises(torch.OutOfMemoryError):
                        runner._seed_device_rng()
                    self.assertTrue(runner.quarantined)
                    self.assertTrue(runner._device_drain_uncertain)
                    if runner._device_rng_before is not None:
                        with self.assertRaises(StudyFailure):
                            runner._restore_device_rng()
                self.assertEqual(events[events.index('fault') + 1:], [])

    def test_failed_rng_restore_latches_before_caller_cleanup(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._device_rng_before = torch.tensor([1], dtype=torch.uint8)
        runner._check_headroom = lambda: None
        runner._check_owner = lambda **kwargs: None
        with (mock.patch.object(torch.cuda, 'set_rng_state',
                                side_effect=torch.OutOfMemoryError('restore failed')),
              mock.patch.object(torch.cuda, 'get_rng_state') as get_rng):
            with self.assertRaisesRegex(torch.OutOfMemoryError, 'restore failed'):
                runner._restore_device_rng()
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)
            get_rng.assert_not_called()

    def test_interrupt_in_rng_restore_failure_handler_keeps_pending_latch(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._device_rng_before = torch.tensor([1], dtype=torch.uint8)
        runner._check_headroom = lambda: None
        runner._check_owner = lambda **kwargs: None
        source, start = inspect.getsourcelines(CudaAgreementRunner._restore_device_rng)
        handler = start + next(index for index, line in enumerate(source)
                               if line.strip() == 'except BaseException:')
        target = handler + 1
        fired = []
        def trace(frame, event, arg):
            if (frame.f_code is CudaAgreementRunner._restore_device_rng.__code__
                    and event == 'line' and frame.f_lineno == target):
                fired.append(True)
                raise KeyboardInterrupt('restore handler interrupted')
            return trace
        with (mock.patch.object(torch.cuda, 'set_rng_state',
                                side_effect=torch.OutOfMemoryError('restore failed')),
              mock.patch.object(torch.cuda, 'get_rng_state') as get_rng):
            previous = sys.gettrace()
            try:
                sys.settrace(trace)
                with self.assertRaisesRegex(KeyboardInterrupt, 'restore handler interrupted'):
                    runner._restore_device_rng()
            finally:
                sys.settrace(previous)
            self.assertEqual(fired, [True])
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)
            get_rng.assert_not_called()

    def test_each_rng_restore_failure_boundary_blocks_another_device_call(self):
        for fault in ('headroom', 'set', 'owner1', 'get', 'owner2', 'mismatch'):
            with self.subTest(fault=fault):
                runner = object.__new__(CudaAgreementRunner)
                runner.quarantined = False
                runner._device_drain_uncertain = False
                runner._device_rng_before = torch.tensor([1], dtype=torch.uint8)
                events, checks = [], [0]
                def fail():
                    events.append('fault')
                    raise KeyboardInterrupt('simulated restore interruption')
                def check_owner(*, allow_quarantined=False, allow_pending=False):
                    if runner._quarantine_latched and not allow_quarantined:
                        raise StudyFailure('owner quarantined')
                    if allow_pending:
                        checks[0] += 1
                        if fault == 'owner' + str(checks[0]):
                            fail()
                    elif runner._device_drain_uncertain:
                        raise StudyFailure('device operation pending')
                runner._check_owner = check_owner
                def check_headroom():
                    check_owner()
                    if fault == 'headroom':
                        fail()
                runner._check_headroom = check_headroom
                def set_rng(*args):
                    events.append('set')
                    if fault == 'set':
                        fail()
                def get_rng(*args):
                    events.append('get')
                    if fault == 'get':
                        fail()
                    if fault == 'mismatch':
                        events.append('fault')
                        return torch.tensor([2], dtype=torch.uint8)
                    return torch.tensor([1], dtype=torch.uint8)
                with (mock.patch.object(torch.cuda, 'set_rng_state', side_effect=set_rng),
                      mock.patch.object(torch.cuda, 'get_rng_state', side_effect=get_rng)):
                    with self.assertRaises((KeyboardInterrupt, StudyFailure)):
                        runner._restore_device_rng()
                    self.assertTrue(runner.quarantined)
                    self.assertTrue(runner._device_drain_uncertain)
                    with self.assertRaises(StudyFailure):
                        runner._restore_device_rng()
                self.assertEqual(events[events.index('fault') + 1:], [])

    def test_failed_or_cancelled_calibration_retains_custody_without_rng_restore(self):
        from looking_glass.agreement_cuda_calibration import CudaAgreementCalibrator
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import _model_contract, _state_hash
        from looking_glass.model_ownership import ModelLease
        spec = importlib.util.spec_from_file_location('rng_custody_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        for mode in ('warmup', 'clean_capture'):
            for fault in ('first_get', 'seed', 'second_get', 'cancel_after_seed', 'cancel_before', 'post_seed_oom'):
                with self.subTest(mode=mode, fault=fault), tempfile.TemporaryDirectory() as temporary:
                    model = fixture.ToyModel().eval()
                    journal = AgreementJournal(Path(temporary) / 'attempt', {'test': 'cpu-fake'})
                    runner = object.__new__(CudaAgreementCalibrator)
                    NaturalAgreementRunner.__init__(runner, model, journal, expected_quartets=1)
                    runner._warmup_completed, runner._completed = mode != 'warmup', False
                    runner._calibration_index, runner._calibration_in_callback = 0, False
                    runner._device_rng_before, runner._device_drain_uncertain = None, False
                    runner._execution_contract = lambda: _model_contract(model)
                    runner._model_digest = lambda: _state_hash(model)
                    runner._check_owner = lambda **kwargs: None
                    if fault == 'post_seed_oom':
                        def fail_forward(*args):
                            events.append('fault')
                            raise torch.OutOfMemoryError('simulated post-seed failure')
                        runner._forward = fail_forward
                    def check_headroom():
                        if runner.quarantined:
                            raise StudyFailure('owner unsafe')
                    runner._check_headroom = check_headroom
                    device, memory, resident = object(), object(), object()
                    runner._device_lease, runner._memory_lease, runner._resident_lease = device, memory, resident
                    events, calls = [], [0]
                    def get_rng(*args):
                        calls[0] += 1
                        events.append('get')
                        if (fault == 'first_get' and calls[0] == 1) or (fault == 'second_get' and calls[0] == 2):
                            events.append('fault')
                            raise torch.OutOfMemoryError('simulated RNG failure')
                        if fault == 'cancel_after_seed' and calls[0] == 2:
                            runner.cancel_event.cancel()
                            events.append('fault')
                        return torch.tensor([0], dtype=torch.uint8)
                    def seed(*args):
                        events.append('seed')
                        if fault == 'seed':
                            events.append('fault')
                            raise torch.OutOfMemoryError('simulated RNG failure')
                    def set_rng(*args):
                        events.append('set')
                    if fault == 'cancel_before':
                        runner.cancel_event.cancel()
                        events.append('fault')
                    previous_rng = torch.random.get_rng_state().clone()
                    with (mock.patch.object(torch.cuda, 'get_rng_state', side_effect=get_rng),
                          mock.patch.object(torch.cuda, 'set_rng_state', side_effect=set_rng),
                          mock.patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=seed),)),
                          mock.patch.object(torch.cuda, 'synchronize') as synchronize):
                        with self.assertRaises((torch.OutOfMemoryError, StudyFailure)):
                            runner.warmup() if mode == 'warmup' else runner.calibrate_action(mode)
                        synchronize.assert_not_called()
                    self.assertIn('fault', events)
                    self.assertEqual(events[events.index('fault') + 1:], [])
                    self.assertTrue(runner.quarantined)
                    self.assertTrue(runner._device_drain_uncertain)
                    self.assertIsNotNone(runner._lease)
                    self.assertTrue(runner._lease._quarantined)
                    with self.assertRaisesRegex(RuntimeError, 'already leased'):
                        ModelLease(model).acquire()
                    self.assertIs(runner._device_lease, device)
                    self.assertIs(runner._memory_lease, memory)
                    self.assertIs(runner._resident_lease, resident)
                    self.assertTrue(torch.equal(previous_rng, torch.random.get_rng_state()))
                    self.assertFalse(torch.cuda.is_initialized())

    def test_cancelled_study_does_not_restore_device_rng_or_release_model(self):
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import _model_contract, _state_hash
        from looking_glass.model_ownership import ModelLease
        spec = importlib.util.spec_from_file_location('rng_study_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        model = fixture.ToyModel().eval()
        with tempfile.TemporaryDirectory() as temporary:
            journal = AgreementJournal(Path(temporary) / 'attempt', {'test': 'cpu-fake'})
            runner = object.__new__(CudaAgreementRunner)
            NaturalAgreementRunner.__init__(runner, model, journal, expected_quartets=1)
            runner._warmup_completed, runner._device_drain_uncertain = True, False
            runner._device_rng_before = None
            runner._execution_contract = lambda: _model_contract(model)
            runner._model_digest = lambda: _state_hash(model)
            runner._runtime_metadata = lambda: {'device': 'cpu-fake'}
            runner._check_owner = lambda **kwargs: None
            def check_headroom():
                if runner.quarantined:
                    raise StudyFailure('owner unsafe')
            runner._check_headroom = check_headroom
            events, calls = [], [0]
            def get_rng(*args):
                calls[0] += 1
                events.append('get')
                if calls[0] == 2:
                    runner.cancel_event.cancel()
                    events.append('cancel')
                return torch.tensor([0], dtype=torch.uint8)
            def set_rng(*args):
                events.append('set')
            previous_rng = torch.random.get_rng_state().clone()
            with (mock.patch.object(torch.cuda, 'get_rng_state', side_effect=get_rng),
                  mock.patch.object(torch.cuda, 'set_rng_state', side_effect=set_rng),
                  mock.patch.object(torch.cuda, 'default_generators', (SimpleNamespace(manual_seed=lambda *_: None),))):
                with self.assertRaises(StudyFailure):
                    NaturalAgreementRunner.run(runner, fixture.cases())
            self.assertEqual(events, ['get', 'get', 'cancel'])
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)
            self.assertIsNotNone(runner._lease)
            with self.assertRaisesRegex(RuntimeError, 'already leased'):
                ModelLease(model).acquire()
            self.assertEqual(model.forward_calls, 0)
            self.assertTrue(torch.equal(previous_rng, torch.random.get_rng_state()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_warmup_state_witness_rejects_value_or_identity_drift(self):
        from looking_glass.agreement_patch_runner import _state_identity
        for drift in ('value', 'replacement'):
            with self.subTest(drift=drift):
                model = torch.nn.Linear(3, 2).eval()
                runner = object.__new__(CudaAgreementRunner)
                runner.model, runner.device = model, 'cpu'
                runner._warmup_completed = False
                runner._device_drain_uncertain = False
                runner._check_headroom = lambda: None
                runner._check_owner = lambda **kwargs: None
                runner._sync = lambda **kwargs: None
                runner._warmup_state_sha256 = runner._model_digest()
                runner._warmup_state_identity = _state_identity(model)
                runner._warmup_completed = True
                self.assertEqual(runner._model_digest(), runner._warmup_state_sha256)
                if drift == 'value':
                    with torch.no_grad():
                        model.weight[0, 0] += 1
                else:
                    model.weight = torch.nn.Parameter(model.weight.detach().clone())
                with self.assertRaisesRegex(StudyFailure, 'changed since same-owner warmup'):
                    runner._model_digest()
                self.assertTrue(runner.quarantined)

    def test_cuda_contract_rejects_cross_phase_attention_flag_drift(self):
        spec = importlib.util.spec_from_file_location('toy_runtime_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        model = fixture.ToyModel().eval()
        model.config.n_embd, model.config.vocab_size = 768, 50257
        model.transformer.h[1].attn.scale_attn_weights = True
        runner = object.__new__(CudaAgreementRunner)
        runner.model, runner.device, runner._expected_key = model, 'cpu', workload()
        runner._warmup_completed = False
        runner._check_owner = lambda **kwargs: None
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            # This CPU-only test isolates attention-configuration drift, not
            # execution under the historical pinned CUDA runtime.
            with (mock.patch.object(torch, '__version__', runner._expected_key.torch_version),
                  mock.patch.object(torch.version, 'cuda', runner._expected_key.cuda_runtime_version)):
                runner._warmup_execution_contract = runner._execution_contract()
                runner._warmup_completed = True
                model.transformer.h[1].attn.scale_attn_weights = False
                with self.assertRaisesRegex(StudyFailure, 'configuration changed since same-owner warmup'):
                    runner._execution_contract()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.use_deterministic_algorithms(deterministic, warn_only=warn)

    def _fake_owner(self, temporary, *, minimum_incremental_bytes=0):
        from looking_glass import agreement_cuda_runner as module
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.device_identity import DeviceIdentity
        from looking_glass.device_lease import DeviceLease
        from looking_glass.memory_budget import MemoryBudget
        from looking_glass.supervisor import OwnerSession
        key = workload()
        warm, cold = profile(key), cold_profile(key)
        identity = object.__new__(DeviceIdentity)
        object.__setattr__(identity, 'gpu_uuid', key.device_uuid)
        session = object.__new__(OwnerSession)
        session._identity = identity
        session._runtime_verified = True
        device_lease = object.__new__(DeviceLease)
        probe = GuardedAgreementMemoryProbe(torch.cuda, device='cuda:0', device_uuid=key.device_uuid,
                                            runtime_binding=lambda: (key.device_uuid, 1),
                                            session=session, device_lease=device_lease)
        budget = MemoryBudget(probe.snapshot, max_gpu_trace_bytes=TRACE_BYTES,
                              max_host_trace_bytes=TRACE_BYTES)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(OwnerSession, 'check'))
        stack.enter_context(mock.patch.object(OwnerSession, 'verify_runtime_binding'))
        stack.enter_context(mock.patch.object(DeviceLease, 'check'))
        stack.enter_context(mock.patch.object(module, '_runtime_uuid', return_value=key.device_uuid))
        stack.enter_context(mock.patch.object(torch.cuda, 'mem_get_info', return_value=(6 * GIB, 8 * GIB)))
        stack.enter_context(mock.patch.object(torch.cuda, 'memory_allocated', return_value=400 * 1024 ** 2))
        stack.enter_context(mock.patch.object(torch.cuda, 'memory_reserved', return_value=500 * 1024 ** 2))
        stack.enter_context(mock.patch.object(torch.cuda, 'is_initialized', return_value=True))
        count = stack.enter_context(mock.patch.object(torch.cuda, 'device_count', return_value=1))
        props = SimpleNamespace(uuid='mock', total_memory=8 * GIB, name='test GPU', major=7, minor=5)
        stack.enter_context(mock.patch.object(torch.cuda, 'get_device_properties', return_value=props))
        stack.enter_context(mock.patch.object(torch.cuda, 'get_allocator_backend', return_value='native'))
        stack.enter_context(mock.patch.object(torch.cuda, 'get_per_process_memory_fraction', return_value=0.25))
        memory_lease = budget.acquire(max(warm.estimated_incremental_bytes, cold.estimated_incremental_bytes,
                                          minimum_incremental_bytes),
                                      max_gpu_trace_bytes=TRACE_BYTES, max_host_trace_bytes=TRACE_BYTES)
        memory_lease.reserve_trace('gpu', TRACE_BYTES)
        memory_lease.reserve_trace('host', TRACE_BYTES)
        self.addCleanup(module._owner_gates.pop, device_lease, None)
        def journal(name):
            return AgreementJournal(Path(temporary) / name, {'test': 'cpu-fake'})
        return module, key, warm, cold, session, device_lease, memory_lease, probe, count, journal

    def test_telemetry_stops_before_following_query_if_owner_is_revoked(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            from looking_glass.supervisor import OwnerSession
            calls = []
            def revoked_after_free(owner, lease):
                calls.append('guard')
                if 'free' in calls:
                    raise RuntimeError('owner revoked')
            def free(device):
                calls.append('free')
                return 6 * GIB, 8 * GIB
            with mock.patch.object(OwnerSession, 'check', side_effect=revoked_after_free), \
                 mock.patch.object(torch.cuda, 'mem_get_info', side_effect=free), \
                 mock.patch.object(torch.cuda, 'memory_allocated') as allocated, \
                 mock.patch.object(torch.cuda, 'memory_reserved') as reserved:
                with self.assertRaisesRegex(RuntimeError, 'owner revoked'):
                    probe.snapshot()
            allocated.assert_not_called()
            reserved.assert_not_called()

    def test_telemetry_stops_between_queries_when_cancelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            runner = CudaAgreementRunner(None, journal('cancelled-telemetry'), session=session,
                device_lease=device, memory_lease=memory, memory_probe=probe, profile=warm,
                warmup_profile=cold, expected_key=key)
            def cancel_after_first_query(*args):
                runner.cancel_event.cancel()
                return 6 * GIB, 8 * GIB
            with (mock.patch.object(torch.cuda, 'mem_get_info', side_effect=cancel_after_first_query),
                  mock.patch.object(torch.cuda, 'memory_allocated') as allocated,
                  mock.patch.object(torch.cuda, 'memory_reserved') as reserved):
                with self.assertRaises(StudyFailure):
                    probe.snapshot()
            allocated.assert_not_called()
            reserved.assert_not_called()
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)

    def test_profile_requires_exact_policy_order_seed_and_margin(self):
        key = workload()
        _validate_profile(profile(key), key)
        for candidate in (profile(key, kinds=CALIBRATION_KINDS[:-1]),
                          profile(key, kinds=tuple(reversed(CALIBRATION_KINDS))),
                          profile(key, seed=17), profile(key, margin_bytes=1),
                          profile(key, margin_bps=1999)):
            with self.subTest(candidate=candidate.profile_id), self.assertRaises(ValueError):
                _validate_profile(candidate, key)
        wrong = workload(sequence_length=4)
        with self.assertRaises(ValueError):
            _validate_profile(profile(wrong), wrong)
        self.assertEqual(len({calibration_case_sha256(kind) for kind in CALIBRATION_KINDS}), 13)

    def test_separate_cold_profile_is_required_for_warmup(self):
        key = workload()
        _validate_warmup_profile(cold_profile(key), key)
        for candidate in (profile(key), cold_profile(key, seed=17), cold_profile(key, case_sha256='f' * 64)):
            with self.subTest(candidate=candidate.profile_id), self.assertRaises(ValueError):
                _validate_warmup_profile(candidate, key)

    def test_unverified_sync_is_never_retried(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._check_owner = mock.Mock()
        with mock.patch.object(torch.cuda, 'synchronize', side_effect=RuntimeError('drain failed')) as sync:
            with self.assertRaisesRegex(RuntimeError, 'drain failed'):
                runner._after_forward()
            runner._failed_forward(RuntimeError('original'))
            with self.assertRaises(StudyFailure):
                runner._sync(allow_quarantined=True)
        self.assertEqual(sync.call_count, 1)
        self.assertTrue(runner.quarantined)
        self.assertTrue(runner._device_drain_uncertain)

    def test_interrupt_after_pending_latch_is_publicly_quarantined(self):
        runner = object.__new__(CudaAgreementRunner)
        runner.quarantined = False
        runner._device_drain_uncertain = False
        runner._check_owner = mock.Mock()
        lines, start = inspect.getsourcelines(CudaAgreementRunner._sync)
        pending_index = next(index for index, line in enumerate(lines)
                             if 'self._device_drain_uncertain = True' in line)
        interrupt_line = start + next(index for index in range(pending_index + 1, len(lines))
                                      if lines[index].strip() == 'try:')
        def trace(frame, event, arg):
            if frame.f_code is CudaAgreementRunner._sync.__code__ and event == 'line' and frame.f_lineno == interrupt_line:
                raise KeyboardInterrupt('after pending latch')
            return trace
        with mock.patch.object(torch.cuda, 'synchronize') as sync:
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    runner._sync()
            finally:
                sys.settrace(None)
            self.assertTrue(runner.quarantined)
            self.assertTrue(runner._device_drain_uncertain)
            runner._failed_forward(KeyboardInterrupt())
            self.assertEqual(sync.call_count, 0)

    def test_assigned_input_and_source_survive_interrupt_before_root_store(self):
        from types import SimpleNamespace
        def interrupt_at(method, needle, invoke):
            lines, start = inspect.getsourcelines(method)
            target = start + next(i for i, line in enumerate(lines) if needle in line)
            fired = []
            def trace(frame, event, arg):
                if frame.f_code is method.__code__ and event == 'line' and frame.f_lineno == target and not fired:
                    fired.append(True)
                    raise KeyboardInterrupt('before custody store')
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    invoke()
            finally:
                sys.settrace(None)
            self.assertEqual(fired, [True])
        for name in ('ids', 'mask'):
            with self.subTest(name=name):
                runner = object.__new__(CudaAgreementRunner)
                runner.device, runner._forward_roots = 'cpu', {'ids': None, 'mask': None}
                runner._check_owner = lambda **kwargs: None
                with mock.patch.object(torch.Tensor, 'to', lambda tensor, *args, **kwargs: tensor.clone()):
                    interrupt_at(CudaAgreementRunner._make_inputs,
                        f"self._forward_roots['{name}'] = {name}",
                        lambda: runner._make_inputs(SimpleNamespace(input_ids=(1, 2, 3, 4, 5))))
                self.assertIsInstance(runner._forward_roots[name], torch.Tensor)
        runner = object.__new__(CudaAgreementRunner)
        runner.device = 'cpu'
        runner._forward_roots = {'sources': None, 'source_fallback': None}
        runner._check_owner = lambda **kwargs: None
        with mock.patch.object(torch.Tensor, 'to', lambda tensor, *args, **kwargs: tensor.clone()):
            interrupt_at(CudaAgreementRunner._prepare_steps, "item['value'] = device_source",
                         lambda: runner._prepare_steps([{'value': torch.zeros(768)}]))
        self.assertIsInstance(runner._forward_roots['source_fallback'], torch.Tensor)

    def test_assigned_model_output_is_rooted_before_failed_drain(self):
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import _model_contract, _state_hash
        spec = importlib.util.spec_from_file_location('toy_output_custody_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        class FakeOwner(CudaAgreementRunner):
            def __init__(self, model, journal):
                NaturalAgreementRunner.__init__(self, model, journal, expected_quartets=1)
                self._device_drain_uncertain = False
                self._forward_roots = {name: None for name in
                    ('ids', 'mask', 'inputs', 'sources', 'source_fallback', 'steps', 'plan', 'output', 'captures', 'offload')}
            def _execution_contract(self): return _model_contract(self.model)
            def _model_digest(self): return _state_hash(self.model)
            def _prepare_steps(self, steps): return steps
            def _make_inputs(self, case): return NaturalAgreementRunner._make_inputs(self, case)
            def _check_owner(self, **kwargs): pass
            def _before_forward(self): pass
            def _seed_device_rng(self): return None
            def _restore_device_rng(self): pass
            def _runtime_metadata(self): return {'device': 'cpu', 'mock': True}
            def _sync(self, **kwargs): raise RuntimeError('simulated failed drain')
        lines, start = inspect.getsourcelines(NaturalAgreementRunner._forward)
        target = start + next(i for i, line in enumerate(lines) if "self._root_forward('output', output)" in line)
        fired = []
        def trace(frame, event, arg):
            if frame.f_code is NaturalAgreementRunner._forward.__code__ and event == 'line' and frame.f_lineno == target and not fired:
                fired.append(True)
                raise KeyboardInterrupt('before output custody store')
            return trace
        with tempfile.TemporaryDirectory() as temporary:
            journal = AgreementJournal(Path(temporary) / 'attempt', {'test': 'cpu-fake'})
            runner = FakeOwner(fixture.ToyModel().eval(), journal)
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    NaturalAgreementRunner.run(runner, fixture.cases())
            finally:
                sys.settrace(None)
            self.assertEqual(fired, [True])
            self.assertTrue(runner.quarantined)
            self.assertIsInstance(runner._forward_roots['output'].logits, torch.Tensor)

    def test_plan_source_and_capture_clones_survive_failed_drain_or_exit(self):
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import _CANDIDATES, _model_contract, _payload, _steps_for_action
        from looking_glass.distilgpt2_intervention_dev import _hook_state
        from looking_glass.model_ownership import ModelLease
        spec = importlib.util.spec_from_file_location('toy_plan_custody_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        class FakeOwner(CudaAgreementRunner):
            def __init__(self, model, journal):
                NaturalAgreementRunner.__init__(self, model, journal, expected_quartets=1)
                self._device_drain_uncertain = False
                self._forward_roots = {name: None for name in
                    ('ids', 'mask', 'inputs', 'sources', 'source_fallback', 'steps', 'plan',
                     'plan_sources', 'plan_captures', 'output', 'captures', 'offload')}
                self.source_refs = []
            def _execution_contract(self): return _model_contract(self.model)
            def _prepare_steps(self, steps): return steps
            def _make_inputs(self, case): return NaturalAgreementRunner._make_inputs(self, case)
            def _check_owner(self, **kwargs): pass
            def _before_forward(self): pass
            def _encode_payload(self, tensor): return _payload(tensor)
            def _sync(self, **kwargs):
                self._device_drain_uncertain = True
                raise RuntimeError('simulated failed drain')
            def _new_plan(self, steps):
                plan = super()._new_plan(steps)
                self.source_refs = [weakref.ref(tensor) for tensor in plan._sources.values()]
                return plan
        for failure in ('drain', 'plan_exit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                model = fixture.ToyModel().eval()
                if failure == 'plan_exit':
                    original = model.forward
                    def fail_after_forward(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError('simulated forward failure')
                    model.forward = fail_after_forward
                journal = AgreementJournal(Path(temporary) / 'attempt', {'test': 'cpu-fake'})
                runner = FakeOwner(model, journal)
                runner._baseline_contract = _model_contract(model)
                runner._baseline_hooks = _hook_state(model)
                runner._lease = ModelLease(model)
                runner._lease.acquire()
                steps = _steps_for_action(fixture.cases()[0], _CANDIDATES[0], 'patch', torch.zeros(8))
                with self.assertRaises(RuntimeError):
                    runner._forward(fixture.cases()[0], steps)
                self.assertTrue(runner.quarantined)
                self.assertEqual(runner._forward_roots['plan']._sources, {})
                if failure == 'plan_exit':
                    with self.assertRaises(RuntimeError):
                        _ = runner._forward_roots['plan'].captures
                capture_refs = [weakref.ref(tensor) for tensor in runner._forward_roots['plan']._captures.values()]
                self.assertTrue(runner.source_refs and all(ref() is not None for ref in runner.source_refs))
                self.assertTrue(capture_refs and all(ref() is not None for ref in capture_refs))
                self.assertEqual(tuple(ref() for ref in runner.source_refs), runner._forward_roots['plan_sources'])
                self.assertIs(runner._forward_roots['plan']._captures, runner._forward_roots['plan_captures'])

    def test_rejected_cuda_observations_are_not_accessed_after_failure(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError('device observation was touched')
        runner = object.__new__(CudaAgreementRunner)
        runner.journal = mock.Mock()
        runner._forwards_recorded = 0
        runner._failure_evidence_error = None
        runner._record_rejected(SimpleNamespace(case_id='case'), Untouchable(), Untouchable(),
                                Untouchable(), RuntimeError('failure'))
        args = runner.journal.record.call_args.args
        self.assertEqual(args[0]['kind'], 'rejected_forward')
        self.assertIn('known rooted buffers', args[0]['unavailable'][0])
        self.assertEqual(args[1], {})

    def test_invalid_profile_is_rejected_before_any_cuda_call(self):
        with mock.patch.object(torch.cuda, 'device_count', side_effect=AssertionError('CUDA called')):
            with self.assertRaises(ValueError):
                CudaAgreementRunner(None, None, session=None, device_lease=None,
                                    memory_lease=None, memory_probe=None, profile=None,
                                    warmup_profile=None, expected_key=None)

    def test_owner_gate_rejects_second_runner_without_another_cuda_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            runner = CudaAgreementRunner(None, journal('first'), session=session, device_lease=device,
                                         memory_lease=memory, memory_probe=probe, profile=warm,
                                         warmup_profile=cold, expected_key=key)
            self.assertIs(module._owner_gates[device][1], runner)
            with self.assertRaisesRegex(ValueError, 'already has'):
                CudaAgreementRunner(None, journal('second'), session=session, device_lease=device,
                                    memory_lease=memory, memory_probe=probe, profile=warm,
                                    warmup_profile=cold, expected_key=key)
            self.assertEqual(count.call_count, 1)

    def test_post_gate_constructor_failure_roots_and_quarantines_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            count.side_effect = RuntimeError('simulated device failure')
            with self.assertRaisesRegex(RuntimeError, 'device failure'):
                CudaAgreementRunner(None, journal('attempt'), session=session, device_lease=device,
                                    memory_lease=memory, memory_probe=probe, profile=warm,
                                    warmup_profile=cold, expected_key=key)
            self.assertIn(device, module._owner_gates)
            self.assertTrue(module._owner_gates[device][1].quarantined)
            self.assertFalse(memory.released)

    def test_calibration_owner_requires_provisional_commitment_and_refuses_study(self):
        from looking_glass.agreement_cuda_calibration import (CudaAgreementCalibrator, PROVISIONAL_JOB_BYTES,
                                                             PROVISIONAL_RESIDENT_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            resident = memory._budget.acquire(PROVISIONAL_RESIDENT_BYTES, max_gpu_trace_bytes=0,
                                               max_host_trace_bytes=0)
            with self.assertRaisesRegex(ValueError, 'incremental commitment'):
                CudaAgreementCalibrator(None, journal('too-small'), session=session, device_lease=device,
                                        memory_lease=memory, resident_lease=resident, memory_probe=probe,
                                        expected_key=key)
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(
                temporary, minimum_incremental_bytes=PROVISIONAL_JOB_BYTES)
            resident = memory._budget.acquire(PROVISIONAL_RESIDENT_BYTES, max_gpu_trace_bytes=0,
                                               max_host_trace_bytes=0)
            runner = CudaAgreementCalibrator(None, journal('calibration'), session=session, device_lease=device,
                                             memory_lease=memory, resident_lease=resident,
                                             memory_probe=probe, expected_key=key)
            with self.assertRaisesRegex(StudyFailure, 'cannot run a study'):
                runner.run([])
            runner._warmup_completed = True
            with self.assertRaisesRegex(StudyFailure, 'out of order'):
                runner.calibrate_action('self_patch')
            self.assertTrue(runner.quarantined)

    def test_telemetry_stops_if_either_bound_memory_lease_is_revoked(self):
        from looking_glass.agreement_cuda_calibration import (CudaAgreementCalibrator, PROVISIONAL_JOB_BYTES,
                                                             PROVISIONAL_RESIDENT_BYTES)
        for removed in ('job', 'resident'):
            with self.subTest(removed=removed), tempfile.TemporaryDirectory() as temporary:
                module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(
                    temporary, minimum_incremental_bytes=PROVISIONAL_JOB_BYTES)
                budget = memory._budget
                resident = budget.acquire(PROVISIONAL_RESIDENT_BYTES, max_gpu_trace_bytes=0,
                                          max_host_trace_bytes=0)
                runner = CudaAgreementCalibrator(None, journal('calibration'), session=session,
                    device_lease=device, memory_lease=memory, resident_lease=resident,
                    memory_probe=probe, expected_key=key)
                previous = budget._leases
                def free(device_index):
                    budget._leases = budget._leases - {memory if removed == 'job' else resident}
                    return 6 * GIB, 8 * GIB
                try:
                    with mock.patch.object(torch.cuda, 'mem_get_info', side_effect=free), \
                         mock.patch.object(torch.cuda, 'memory_allocated') as allocated, \
                         mock.patch.object(torch.cuda, 'memory_reserved') as reserved:
                        with self.assertRaises((ValueError, StudyFailure)):
                            runner._check_headroom()
                        allocated.assert_not_called()
                        reserved.assert_not_called()
                    self.assertTrue(runner.quarantined)
                finally:
                    budget._leases = previous

    def test_allocator_change_rejects_before_telemetry(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            runner = CudaAgreementRunner(None, journal('attempt'), session=session, device_lease=device,
                                         memory_lease=memory, memory_probe=probe, profile=warm,
                                         warmup_profile=cold, expected_key=key)
            with mock.patch.object(torch.cuda, 'get_allocator_backend', return_value='changed'), \
                 mock.patch.object(torch.cuda, 'mem_get_info') as free:
                with self.assertRaisesRegex(StudyFailure, 'allocator'):
                    runner._check_headroom()
                free.assert_not_called()

    def test_calibration_runs_exact_synthetic_actions_without_publishing_values(self):
        from looking_glass import agreement_cuda_runner as cuda_module
        from looking_glass import agreement_cuda_calibration as calibration
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import _model_contract, _payload, _state_hash
        spec = importlib.util.spec_from_file_location('toy_calibration_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        model = fixture.ToyModel().eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'test': 'synthetic-only'})
            runner = object.__new__(calibration.CudaAgreementCalibrator)
            NaturalAgreementRunner.__init__(runner, model, journal, expected_quartets=1)
            runner._warmup_completed = runner._completed = False
            runner._device_drain_uncertain = False
            runner._calibration_index, runner._calibration_donors = 0, None
            runner._calibration_in_callback = False
            runner._forward_roots = {name: None for name in
                ('ids', 'mask', 'inputs', 'sources', 'source_fallback', 'steps', 'plan', 'output', 'captures', 'offload')}
            runner._execution_contract = lambda: _model_contract(model)
            runner._model_digest = lambda: _state_hash(model)
            runner._encode_payload = _payload
            runner._make_inputs = NaturalAgreementRunner._make_inputs.__get__(runner)
            runner._prepare_steps = lambda steps: steps
            runner._check_owner = lambda **kwargs: None
            runner._before_forward = runner._after_forward = lambda: None
            runner._sync = lambda **kwargs: None
            runner._seed_device_rng = lambda: None
            runner._restore_device_rng = lambda: None
            target, donor = (1, 11, 2, 3, 4), (1, 12, 2, 3, 4)
            with mock.patch.object(cuda_module, 'CALIBRATION_INPUT_IDS', target), \
                 mock.patch.object(cuda_module, 'CALIBRATION_DONOR_IDS', donor), \
                 mock.patch.object(calibration, 'CALIBRATION_INPUT_IDS', target), \
                 mock.patch.object(calibration, 'CALIBRATION_DONOR_IDS', donor):
                runner.warmup()
                records = [runner.calibrate_action(kind) for kind in CALIBRATION_KINDS]
            self.assertEqual([record['sample_id'] for record in records], list(CALIBRATION_KINDS))
            self.assertTrue(all(record['capture_or_landing_checked'] for record in records))
            self.assertIsNone(runner._calibration_donors)
            self.assertTrue(runner._completed)
            self.assertEqual(model.forward_calls, 14)
            import json
            events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 14)
            self.assertTrue(all(event['tensors'] == {} for event in events))
            self.assertTrue(all(event['event']['kind'] == 'gpu_calibration_action' for event in events[1:]))

    def test_calibrator_rejects_unbound_base_study_before_forward(self):
        from looking_glass.agreement_cuda_calibration import CudaAgreementCalibrator
        from looking_glass.agreement_journal import AgreementJournal
        spec = importlib.util.spec_from_file_location('toy_no_study_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        model = fixture.ToyModel().eval()
        with tempfile.TemporaryDirectory() as temporary:
            journal = AgreementJournal(Path(temporary) / 'attempt', {'test': 'synthetic-only'})
            runner = object.__new__(CudaAgreementCalibrator)
            NaturalAgreementRunner.__init__(runner, model, journal, expected_quartets=1)
            runner._calibration_in_callback = False
            runner._device_drain_uncertain = False
            with self.assertRaisesRegex(StudyFailure, 'cannot run a study'):
                NaturalAgreementRunner.run(runner, fixture.cases())
            self.assertEqual(model.forward_calls, 0)

    def test_late_bad_length_is_rejected_before_any_study_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            runner = CudaAgreementRunner(None, journal('attempt'), session=session, device_lease=device,
                                         memory_lease=memory, memory_probe=probe, profile=warm,
                                         warmup_profile=cold, expected_key=key, expected_quartets=2)
            runner._warmup_completed = True
            cases = []
            for family in range(2):
                for index, (lemma, number) in enumerate((('a', 'singular'), ('a', 'plural'),
                                                          ('b', 'singular'), ('b', 'plural'))):
                    ids = (464, 100 + index, 200 + family, 300, 400)
                    if family == 1 and index == 3:
                        ids += (500,)
                    cases.append(AgreementCase(f'{family}-{index}', f'q{family}', lemma, number, ids, 1))
            with mock.patch.object(NaturalAgreementRunner, 'run', side_effect=AssertionError('forward started')) as run:
                with self.assertRaises(StudyFailure):
                    runner.run(cases)
            self.assertEqual(run.call_count, 0)
            self.assertTrue(runner.quarantined)

    def test_heldout_is_rejected_before_unverified_low_level_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, key, warm, cold, session, device, memory, probe, count, journal = self._fake_owner(temporary)
            runner = CudaAgreementRunner(None, journal('heldout'), session=session, device_lease=device,
                                         memory_lease=memory, memory_probe=probe, profile=warm,
                                         warmup_profile=cold, expected_key=key, expected_quartets=1)
            runner._warmup_completed = True
            cases = [AgreementCase(str(i), 'q', lemma, number, (464, 100+i, 200, 300, 400), 1)
                     for i, (lemma, number) in enumerate((('a', 'singular'), ('a', 'plural'),
                                                          ('b', 'singular'), ('b', 'plural')))]
            frozen, binding = object(), object()
            with mock.patch.object(NaturalAgreementRunner, 'run', return_value={'status': 'complete'}) as base:
                with self.assertRaisesRegex(StudyFailure, 'held-out execution is not yet supported'):
                    runner.run(cases, mode='heldout', frozen_predictions=frozen, heldout_binding=binding)
            base.assert_not_called()
            metadata = runner._runtime_metadata()
            self.assertEqual(metadata['profile_id'], warm.profile_id)
            self.assertEqual(metadata['warmup_profile_id'], cold.profile_id)
            self.assertEqual(metadata['warmup_policy_sha256'], WARMUP_POLICY_SHA256)

    def test_warmup_executes_ten_capture_path_once_and_discards_values(self):
        from looking_glass import agreement_cuda_runner as cuda_module
        from looking_glass.agreement_journal import AgreementJournal
        from looking_glass.agreement_patch_runner import NaturalAgreementRunner, _model_contract, _payload, _state_hash
        spec = importlib.util.spec_from_file_location('toy_agreement_fixture',
                    Path(__file__).with_name('test_agreement_patch_runner.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        model = fixture.ToyModel().eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'test': 'synthetic-only'})
            runner = object.__new__(CudaAgreementRunner)
            NaturalAgreementRunner.__init__(runner, model, journal, expected_quartets=1)
            runner._warmup_completed = False
            runner._device_drain_uncertain = False
            runner._forward_roots = {name: None for name in
                                     ('ids', 'mask', 'inputs', 'sources', 'steps', 'plan', 'output', 'captures')}
            runner._execution_contract = lambda: _model_contract(model)
            runner._model_digest = lambda: _state_hash(model)
            runner._encode_payload = _payload
            runner._make_inputs = NaturalAgreementRunner._make_inputs.__get__(runner)
            runner._prepare_steps = lambda steps: steps
            runner._before_forward = lambda: None
            runner._after_forward = lambda: None
            runner._check_owner = lambda **kwargs: None
            runner._sync = lambda **kwargs: None
            runner._seed_device_rng = lambda: None
            runner._restore_device_rng = lambda: None
            prior_rng = torch.random.get_rng_state().clone()
            with mock.patch.object(cuda_module, 'CALIBRATION_INPUT_IDS', (1, 11, 2, 3, 4)):
                warmup = runner.warmup()
            self.assertEqual(model.forward_calls, 1)
            self.assertTrue(runner._warmup_completed)
            self.assertEqual(runner._forwards, 0)
            self.assertTrue(torch.equal(prior_rng, torch.random.get_rng_state()))
            self.assertEqual(warmup['sample_id'], 'warmup')
            self.assertEqual(warmup['seed'], 20261028)
            self.assertEqual(warmup['policy_sha256'], WARMUP_POLICY_SHA256)
            self.assertEqual(len(warmup['model_state_sha256']), 64)
            events = (path / 'events.jsonl').read_text().splitlines()
            self.assertEqual(len(events), 1)
            import json
            event = json.loads(events[0])
            self.assertEqual(event['event']['kind'], 'gpu_warmup')
            self.assertEqual(len(event['event']['plan_metadata']['steps']), 10)
            self.assertEqual(event['tensors'], {})


if __name__ == '__main__':
    unittest.main()
