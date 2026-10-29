"""CPU-only protocol and supervised-owner boundary tests."""
import builtins
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from looking_glass import grouped_gpu_dev as dev


def synthetic_runtime():
    return {'schema': 'grouped-gpu-runtime-v1', 'physical_gpu_uuid': 'fake', 'driver': '1.2',
        'name': 'fake', 'total_bytes': 8 * dev._GIB, 'compute_capability': [8, 6],
        'python': 'synthetic', 'platform': 'synthetic', 'torch': '2.8.0+cu128', 'cuda_build': '12.8',
        'cudnn': 90000, 'transformers': '4.57.3', 'transformers_gpt2_source_sha256': 'a' * 64,
        'torch_threads': 1, 'torch_interop_threads': 1, 'allocator_backend': 'native',
        'allocator_fraction': .25, 'matmul_tf32': False, 'cudnn_tf32': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True, 'deterministic_algorithms': True,
        'flash_sdp': False, 'memory_efficient_sdp': False, 'math_sdp': True, 'cpu_autocast': False,
        'cuda_autocast': False, 'cublas_workspace_config': ':4096:8', 'float32_matmul_precision': 'highest',
        'conversion_overwrite': False, 'conversion_swap': False, 'attention': 'eager',
        'cache': 'disabled', 'dtype': 'float32'}


class GroupedGpuDevelopmentTests(unittest.TestCase):
    def tearDown(self):
        dev._ROOTS[:] = [None] * len(dev._ROOTS)

    def test_import_does_not_import_cuda_capable_dependencies(self):
        repo = str(Path(__file__).resolve().parents[1])
        code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                "import looking_glass.grouped_gpu_dev;"
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, repo], check=True,
                       capture_output=True, text=True, timeout=30)

    def test_fixed_protocol_rejects_changed_input_even_with_new_hash(self):
        path = Path(__file__).resolve().parents[1] / dev._PROTOCOL
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        value = dev._protocol(path, digest)
        self.assertEqual(value['combined_shape'], [2, 4])
        self.assertNotEqual(value['policy_groups'][0], value['broker_holdout_groups'][0])
        value['broker_holdout_groups'][0][0][1] += 1
        changed = json.dumps(value).encode()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'changed.json'
            target.write_bytes(changed)
            with self.assertRaises(ValueError):
                dev._protocol(target, hashlib.sha256(changed).hexdigest())
            with self.assertRaises(ValueError):
                dev._protocol(target, digest)
            value['broker_holdout_groups'][0][0][1] -= 1
            value['control_host_ledger']['max_requests'] = True
            changed = json.dumps(value).encode()
            target.write_bytes(changed)
            with self.assertRaises(ValueError):
                dev._protocol(target, hashlib.sha256(changed).hexdigest())

    def test_isolated_bootstrap_uses_explicit_site_not_sourceless_checkout_shadow(self):
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'grouped_gpu_dev_bootstrap.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, site, prefix = root / 'checkout', root / 'explicit-site', root / 'prefix'
            (checkout / 'scripts').mkdir(parents=True)
            site.mkdir()
            prefix.mkdir()
            script = checkout / 'scripts' / source.name
            shutil.copyfile(source, script)
            (checkout / 'torch.pyc').write_bytes(b'ignored sourceless shadow')
            (site / 'torch.py').write_text('raise RuntimeError("must not execute")', encoding='utf-8')
            command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                       str(script), '--site-packages', str(site), '--check-only']
            completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(Path(json.loads(completed.stdout)['torch_origin']), site / 'torch.py')
            (site / 'torch.py').unlink()
            (site / 'torch.pyc').write_bytes(b'unsupported sourceless dependency')
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('sourceless dependency', rejected.stderr)
            nested = subprocess.run([sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                str(script), '--site-packages', str(checkout), '--check-only'],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(nested.returncode, 0)
            self.assertIn('separate from the checkout', nested.stderr)

    def test_owner_requires_verified_teardown_before_attestation(self):
        session = SimpleNamespace(device_lease_path='unused', identity=object(), run_id='owner')
        lease = object()
        real_import = builtins.__import__
        events = []
        def importing(name, *args, **kwargs):
            if name in ('torch', 'transformers'):
                events.append('import-' + name)
                return SimpleNamespace()
            return real_import(name, *args, **kwargs)
        def write(path, value):
            events.append(Path(path).name)
        def run(*args, **kwargs):
            events.append('run')
            return {'status': 'passed'}
        with patch.object(dev, 'sys', SimpleNamespace(modules={})), \
             patch('builtins.__import__', side_effect=importing), \
             patch.object(dev.OwnerSession, 'from_environment', return_value=session), \
             patch.object(dev.DeviceLease, 'acquire', return_value=lease), \
             patch.object(dev.OwnerSession, 'check', side_effect=lambda *a: events.append('check')), \
             patch.object(dev.OwnerSession, 'attest_clean', side_effect=lambda *a: events.append('attest')), \
             patch.object(dev, '_require_bootstrap'), \
             patch.object(dev, '_require_supervisor'), \
             patch.object(dev, '_require_gpu_boundary'), \
             patch.object(dev, '_configure', side_effect=lambda *a: events.append('configure')), \
             patch.object(dev, '_bind_device', side_effect=lambda *a: events.append('bind')), \
             patch.object(dev, '_verify_sources', return_value={dev._PROTOCOL: 'a' * 64}), \
             patch.object(dev, '_protocol', return_value={}), \
             patch.object(dev, '_write_json', side_effect=write), \
             patch.object(dev, '_run', side_effect=run):
            try:
                self.assertEqual(dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver'), 0)
                self.assertLess(events.index('check'), events.index('import-torch'))
                self.assertLess(events.index('bind'), events.index('import-transformers'))
                self.assertLess(events.index('run'), events.index('candidate.json'))
                self.assertLess(events.index('candidate.json'), events.index('attest'))
                self.assertIs(dev._ROOTS[0], session)
                self.assertIs(dev._ROOTS[1], lease)
            finally:
                dev._ROOTS[:] = [None] * len(dev._ROOTS)

    def test_owner_failure_keeps_lease_rooted_and_does_not_attest(self):
        session = SimpleNamespace(device_lease_path='unused', identity=object(), run_id='owner')
        lease = object()
        real_import = builtins.__import__
        writes = []
        def importing(name, *args, **kwargs):
            if name in ('torch', 'transformers'):
                return SimpleNamespace()
            return real_import(name, *args, **kwargs)
        with patch.object(dev, 'sys', SimpleNamespace(modules={})), \
             patch('builtins.__import__', side_effect=importing), \
             patch.object(dev.OwnerSession, 'from_environment', return_value=session), \
             patch.object(dev.DeviceLease, 'acquire', return_value=lease), \
             patch.object(dev.OwnerSession, 'check'), \
             patch.object(dev.OwnerSession, 'attest_clean') as attest, \
             patch.object(dev, '_require_bootstrap'), \
             patch.object(dev, '_require_supervisor'), \
             patch.object(dev, '_require_gpu_boundary'), \
             patch.object(dev, '_configure'), \
             patch.object(dev, '_bind_device'), \
             patch.object(dev, '_verify_sources', return_value={dev._PROTOCOL: 'a' * 64}), \
             patch.object(dev, '_protocol', return_value={}), \
             patch.object(dev, '_write_json', side_effect=lambda path, value: writes.append((Path(path).name, value))), \
             patch.object(dev, '_run', side_effect=MemoryError('private device failure')):
            try:
                self.assertEqual(dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver'), 1)
                attest.assert_not_called()
                self.assertIs(dev._ROOTS[1], lease)
                self.assertIsInstance(dev._ROOTS[13], MemoryError)
                failure = next(value for name, value in writes if name == 'failure.json')
                self.assertEqual(failure['error_type'], 'MemoryError')
                self.assertNotIn('private device failure', repr(failure))
            finally:
                dev._ROOTS[:] = [None] * len(dev._ROOTS)

    def test_owner_interrupt_is_rooted_and_propagated_without_attestation(self):
        session = SimpleNamespace(device_lease_path='unused', identity=object(), run_id='owner')
        real_import = builtins.__import__
        def importing(name, *args, **kwargs):
            return SimpleNamespace() if name in ('torch', 'transformers') else real_import(name, *args, **kwargs)
        with patch.object(dev, 'sys', SimpleNamespace(modules={})), \
             patch('builtins.__import__', side_effect=importing), \
             patch.object(dev, '_require_bootstrap'), patch.object(dev, '_require_supervisor'), \
             patch.object(dev, '_require_gpu_boundary'), \
             patch.object(dev.OwnerSession, 'from_environment', return_value=session), \
             patch.object(dev.DeviceLease, 'acquire', return_value=object()), \
             patch.object(dev.OwnerSession, 'check'), \
             patch.object(dev.OwnerSession, 'attest_clean') as attest, \
             patch.object(dev, '_verify_sources', return_value={dev._PROTOCOL: 'a' * 64}), \
             patch.object(dev, '_protocol', return_value={}), patch.object(dev, '_write_json'), \
             patch.object(dev, '_configure'), patch.object(dev, '_bind_device'), \
             patch.object(dev, '_run', side_effect=KeyboardInterrupt('cancelled')):
            with self.assertRaises(KeyboardInterrupt):
                dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver')
            attest.assert_not_called()
            self.assertIsInstance(dev._ROOTS[13], KeyboardInterrupt)

    def test_new_interrupt_while_writing_owner_failure_takes_precedence(self):
        session = SimpleNamespace(device_lease_path='unused', identity=object(), run_id='owner')
        real_import = builtins.__import__
        def importing(name, *args, **kwargs):
            return SimpleNamespace() if name in ('torch', 'transformers') else real_import(name, *args, **kwargs)
        def writing(path, value):
            if Path(path).name == 'failure.json':
                raise KeyboardInterrupt('write interrupted')
        with patch.object(dev, 'sys', SimpleNamespace(modules={})), \
             patch('builtins.__import__', side_effect=importing), \
             patch.object(dev, '_require_bootstrap'), patch.object(dev, '_require_supervisor'), \
             patch.object(dev, '_require_gpu_boundary'), \
             patch.object(dev.OwnerSession, 'from_environment', return_value=session), \
             patch.object(dev.DeviceLease, 'acquire', return_value=object()), \
             patch.object(dev.OwnerSession, 'check'), \
             patch.object(dev.OwnerSession, 'attest_clean') as attest, \
             patch.object(dev, '_verify_sources', return_value={dev._PROTOCOL: 'a' * 64}), \
             patch.object(dev, '_protocol', return_value={}), patch.object(dev, '_write_json', side_effect=writing), \
             patch.object(dev, '_configure'), patch.object(dev, '_bind_device'), \
             patch.object(dev, '_run', side_effect=RuntimeError('initial failure')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver')
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertIsInstance(dev._ROOTS[13], RuntimeError)
            self.assertIsInstance(dev._ROOTS[17], KeyboardInterrupt)
            attest.assert_not_called()

    def test_failed_device_call_is_rooted_and_prohibits_later_calls(self):
        calls = []
        class Payload:
            pass
        payload = Payload()
        reference = weakref.ref(payload)
        def failed_operation(owned=payload):
            calls.append('failed')
            raise RuntimeError('simulated failed drain')
        with patch.object(dev.OwnerSession, 'check') as check:
            try:
                dev._cuda(object(), object(), failed_operation)
                self.fail('failure was not raised')
            except RuntimeError:
                pass
            del payload, failed_operation
            self.assertIsNotNone(reference())
            with self.assertRaisesRegex(RuntimeError, 'prohibited'):
                dev._cuda(object(), object(), lambda: calls.append('must-not-run'))
            self.assertEqual(calls, ['failed'])
            self.assertEqual(check.call_count, 1)
            self.assertIsInstance(dev._ROOTS[15], RuntimeError)

    def test_owner_loss_during_completed_operation_prevents_publication(self):
        class Payload:
            pass
        owned = Payload()
        reference = weakref.ref(owned)
        operations = []
        def completed(owned=owned):
            operations.append('completed')
            return owned
        with patch.object(dev.OwnerSession, 'check', side_effect=[None, RuntimeError('owner lost')]):
            try:
                dev._cuda(object(), object(), completed)
                self.fail('owner loss was not reported')
            except RuntimeError:
                pass
            del owned, completed
            self.assertIsNotNone(reference())
            with self.assertRaisesRegex(RuntimeError, 'prohibited'):
                dev._cuda(object(), object(), lambda: operations.append('must-not-run'))
            self.assertEqual(operations, ['completed'])
            self.assertIsInstance(dev._ROOTS[15], RuntimeError)

    def test_nested_failure_latch_blocks_return_even_if_callback_swallows_it(self):
        original = RuntimeError('nested failure')
        def operation():
            dev._ROOTS[15] = original
            return 'must-not-publish'
        with patch.object(dev.OwnerSession, 'check'):
            with self.assertRaisesRegex(RuntimeError, 'nested uncertain failure') as raised:
                dev._cuda(object(), object(), operation)
        self.assertIs(raised.exception.__cause__, original)
        self.assertIs(dev._ROOTS[15], original)
        self.assertIs(dev._ROOTS[18], raised.exception)

    def test_nested_diagnostic_oom_keeps_original_failure_rooted(self):
        class Payload:
            pass
        payload = Payload()
        reference = weakref.ref(payload)
        def original_failure(owned=payload):
            raise ValueError('original device failure')
        try:
            original_failure()
        except ValueError as exc:
            original = exc
        def swallowed_operation():
            dev._ROOTS[15] = original
            return 'must-not-publish'
        with patch.object(dev.OwnerSession, 'check'), \
             patch.object(dev, 'RuntimeError', side_effect=MemoryError('diagnostic allocation'), create=True):
            try:
                dev._cuda(object(), object(), swallowed_operation)
                self.fail('diagnostic failure was not raised')
            except MemoryError:
                pass
        del payload, original_failure, original, swallowed_operation
        self.assertIsNotNone(reference())
        self.assertIsInstance(dev._ROOTS[15], ValueError)
        self.assertIsInstance(dev._ROOTS[18], MemoryError)

    def test_preloaded_hub_or_transformer_submodule_rejects_before_lease(self):
        for name in ('huggingface_hub.constants', 'transformers.utils', 'torch._utils'):
            with self.subTest(name=name), \
                 patch.object(dev, 'sys', SimpleNamespace(modules={name: object()})), \
                 patch.object(dev, '_require_bootstrap'), patch.object(dev, '_require_supervisor'), \
                 patch.object(dev, '_require_gpu_boundary'), \
                 patch.object(dev.OwnerSession, 'from_environment') as session, \
                 patch.object(dev.DeviceLease, 'acquire') as acquire, \
                 patch.object(dev, '_write_json'):
                self.assertEqual(dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver'), 1)
                session.assert_not_called()
                acquire.assert_not_called()
                dev._ROOTS[:] = [None] * len(dev._ROOTS)

    def test_unexpected_conversion_result_is_rooted(self):
        class Payload:
            pass
        unexpected = Payload()
        reference = weakref.ref(unexpected)
        model = SimpleNamespace(to=lambda **kwargs: unexpected)
        with patch.object(dev.OwnerSession, 'check'):
            with self.assertRaisesRegex(RuntimeError, 'another object'):
                dev._convert_model(object(), object(), model, device='cpu', dtype='float32')
        del model, unexpected
        self.assertIsNotNone(reference())
        self.assertIs(dev._ROOTS[16], reference())

    def test_wrong_runtime_binding_prevents_transformers_import(self):
        session = SimpleNamespace(device_lease_path='unused', identity=object(), run_id='owner')
        imports = []
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(uuid='wrong')))
        real_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name in ('torch', 'transformers'):
                imports.append(name)
                return fake_torch if name == 'torch' else SimpleNamespace()
            return real_import(name, *args, **kwargs)
        with patch.object(dev, 'sys', SimpleNamespace(modules={})), \
             patch('builtins.__import__', side_effect=importing), \
             patch.object(dev, '_require_bootstrap'), \
             patch.object(dev, '_require_supervisor'), \
             patch.object(dev, '_require_gpu_boundary'), \
             patch.object(dev.OwnerSession, 'from_environment', return_value=session), \
             patch.object(dev.DeviceLease, 'acquire', return_value=object()), \
             patch.object(dev.OwnerSession, 'check'), \
             patch.object(dev.OwnerSession, 'attest_clean') as attest, \
             patch.object(dev, '_verify_sources', return_value={dev._PROTOCOL: 'a' * 64}), \
             patch.object(dev, '_protocol', return_value={}), \
             patch.object(dev, '_write_json'), \
             patch.object(dev, '_configure'), \
             patch.object(dev, '_runtime_uuid', return_value='wrong'), \
             patch.object(dev.OwnerSession, 'verify_runtime_binding', side_effect=RuntimeError('wrong UUID')):
            self.assertEqual(dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver'), 1)
            self.assertEqual(imports, ['torch'])
            attest.assert_not_called()

    def test_supervisor_compatibility_gate_rejects_unreviewed_version(self):
        from looking_glass import supervisor
        with patch.object(supervisor, 'SUPERVISOR_LAUNCH_CUSTODY_CONTRACT', None, create=True):
            with self.assertRaisesRegex(RuntimeError, 'reviewed gated'):
                dev._require_supervisor()

    def test_gpu_launch_is_disabled_before_any_owner_or_device_action(self):
        with patch.object(dev, '_require_bootstrap'), patch.object(dev, '_require_supervisor'), \
             patch.object(dev.OwnerSession, 'from_environment') as session, \
             patch.object(dev.DeviceLease, 'acquire') as lease, patch.object(dev, '_write_json'):
            with self.assertRaisesRegex(RuntimeError, 'session-bound GPU worker'):
                dev.run_development(workspace_root='unused', cache_dir='unused', output='unused',
                                    site_packages='unused', smi='unused')
            self.assertEqual(dev._owner_main('run', 'cache', 'id', 'commit', 'a' * 64, 'driver'), 1)
            session.assert_not_called()
            lease.assert_not_called()

    def test_routes_reject_swapped_outputs_and_indistinguishable_references(self):
        import torch
        from looking_glass import batch_validation
        self.assertTrue(torch.set_flush_denormal(False))
        serial = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
        distinct, _ = dev._separation(batch_validation, serial, 1e-6, 1e-5)
        self.assertTrue(distinct)
        self.assertEqual(dev._route_comparison(batch_validation, serial, serial[0], 0, 1e-6, 1e-5)['status'], 'matched')
        self.assertEqual(dev._route_comparison(batch_validation, serial, serial[1], 0, 1e-6, 1e-5)['status'], 'mismatch')
        distinct, _ = dev._separation(batch_validation, [serial[0], serial[0] + 1e-8], 1e-6, 1e-5)
        self.assertFalse(distinct)

    def test_serial_reference_rejects_wrong_shape_dtype_and_nonfinite(self):
        import torch
        from looking_glass import batch_validation
        valid = torch.zeros((1, 4, 50257), dtype=torch.float32)
        dev._check_serial_reference(batch_validation, valid, 50257)
        bad = [torch.zeros((1, 4, 1), dtype=torch.float32), valid.double(), valid.clone()]
        bad[-1][0, 0, 0] = float('nan')
        for reference in bad:
            with self.subTest(shape=reference.shape, dtype=reference.dtype):
                with self.assertRaisesRegex(RuntimeError, 'serial reference'):
                    dev._check_serial_reference(batch_validation, reference, 50257)

    def test_ok_delivery_must_not_claim_failure_or_cancellation(self):
        handle = SimpleNamespace(_job=SimpleNamespace(job_id='job-1'))
        broker = SimpleNamespace(lookup=lambda owner, request_id: handle)
        validation = SimpleNamespace(_valid_output=lambda *args: True)
        case = {'owner': 'team-a'}
        fields = dict(status='ok', cancel_requested=False, error_code=None, team_id='team-a',
                      request_id='request-1', job_id='job-1', logits=object())
        dev._check_delivery(validation, broker, case, handle, SimpleNamespace(**fields), 'request-1', 50257)
        for field, value in [('cancel_requested', True), ('error_code', 'unexpected')]:
            with self.subTest(field=field):
                changed = dict(fields, **{field: value})
                with self.assertRaisesRegex(RuntimeError, 'routing mismatch'):
                    dev._check_delivery(validation, broker, case, handle, SimpleNamespace(**changed),
                                        'request-1', 50257)

    def test_public_comparison_rejects_inconsistent_diagnostics(self):
        valid = {'status': 'matched', 'max_abs_error': 0.0, 'absolute_error_overflow': False,
                 'max_relative_error': 0.0, 'relative_error_nonfinite_reason': None}
        self.assertEqual(dev._public_comparison(valid), valid)
        for changes in ({'status': 'mismatch'}, {'max_relative_error': 1.0},
                        {'max_abs_error': -1}, {'max_abs_error': None},
                        {'relative_error_nonfinite_reason': 'zero_reference'}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                dev._public_comparison({**valid, **changes})

    def test_memory_series_rejects_changed_total_and_lower_post_peak(self):
        base = {'device_free_bytes': 6 * dev._GIB, 'device_total_bytes': 8 * dev._GIB,
                'allocated_bytes': 400, 'reserved_bytes': 500,
                'peak_allocated_bytes': 400, 'peak_reserved_bytes': 500}
        with tempfile.TemporaryDirectory() as directory:
            records = []
            dev._record_memory(Path(directory), records, 'first', base, 8 * dev._GIB)
            changed = dict(base, device_total_bytes=7 * dev._GIB)
            with self.assertRaisesRegex(RuntimeError, 'total changed'):
                dev._record_memory(Path(directory), records, 'changed', changed, 8 * dev._GIB)
            self.assertEqual(len(records), 2)
            self.assertTrue((Path(directory) / 'memory-001.json').is_file())
            impossible = dict(base, device_free_bytes=9 * dev._GIB)
            with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                dev._record_memory(Path(directory), records, 'invalid', impossible, 8 * dev._GIB)
            self.assertEqual(json.loads((Path(directory) / 'memory-002.json').read_bytes())['stage'], 'invalid')
            impossible = dict(base, peak_reserved_bytes=9 * dev._GIB)
            with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                dev._record_memory(Path(directory), records, 'invalid-peak', impossible, 8 * dev._GIB)
            self.assertEqual(json.loads((Path(directory) / 'memory-003.json').read_bytes())['stage'], 'invalid-peak')
            impossible = dict(base, peak_reserved_bytes=3 * dev._GIB)
            with self.assertRaisesRegex(RuntimeError, 'fixed native cap'):
                dev._record_memory(Path(directory), records, 'invalid-cap', impossible, 8 * dev._GIB)
            self.assertEqual(json.loads((Path(directory) / 'memory-004.json').read_bytes())['stage'], 'invalid-cap')
        lower_peak = dict(base, allocated_bytes=300, reserved_bytes=300,
                          peak_allocated_bytes=300, peak_reserved_bytes=300)
        with self.assertRaisesRegex(RuntimeError, 'below the reset baseline'):
            dev._check_memory_pair(base, lower_peak)
        with self.assertRaisesRegex(RuntimeError, 'reset baseline'):
            dev._check_peak_reset(dict(base, peak_allocated_bytes=401))

    def test_runtime_contract_rejects_wrong_dtype_flags_and_cap(self):
        protocol = {'allocator_fraction': .25}
        dev._validate_runtime(synthetic_runtime(), protocol)
        for key, value in [('dtype', 'bfloat16'), ('cuda_autocast', True),
                           ('deterministic_algorithms', False), ('allocator_fraction', .5),
                           ('torch_threads', True)]:
            runtime = synthetic_runtime()
            runtime[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                dev._validate_runtime(runtime, protocol)

    def _synthetic_owner(self, *, swap=False, low_before_copy=False, indistinguishable=False,
                         peak_regression=False, promotion=False):
        import gc
        import torch
        self.assertTrue(torch.set_flush_denormal(False))
        from torch import nn
        from looking_glass import cuda_cleanup, pretrained
        from looking_glass.host_budget import HostClaim
        conversions = []
        references = []
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False, _attn_implementation='eager', vocab_size=50257)
                self.embedding = SimpleNamespace(weight=None)
            def get_input_embeddings(self):
                return self.embedding
            def get_output_embeddings(self):
                return self.embedding
            def to(self, *args, **kwargs):
                conversions.append('called')
                return super().to(*args, **kwargs)
            def forward(self, input_ids, attention_mask):
                if indistinguishable and input_ids[0, 0].item() >= 17:
                    input_ids = torch.zeros_like(input_ids)
                if swap and input_ids.shape[0] == 2 and input_ids[0, 0].item() >= 17:
                    input_ids = input_ids.flip(0)
                return {'logits': input_ids.float().unsqueeze(-1).expand(-1, -1, 50257).contiguous()}
        def load(**kwargs):
            return SimpleNamespace(model=FakeModel().eval(), metadata={
                'parameter_count': 0, 'asset_sha256': {name: digest
                    for name, (_, digest) in pretrained.ASSETS.items()}})
        properties = SimpleNamespace(uuid='fake', total_memory=8 * dev._GIB,
                                     name='fake', major=8, minor=6)
        protocol_path = Path(__file__).resolve().parents[1] / dev._PROTOCOL
        raw = protocol_path.read_bytes()
        protocol = dev._protocol(protocol_path, hashlib.sha256(raw).hexdigest())
        protocol['policy_groups'] = [[[1, 2, 3, 4], [5, 6, 7, 8]],
                                     [[9, 10, 11, 12], [13, 14, 15, 16]]]
        protocol['broker_holdout_groups'] = [[[17, 18, 19, 20], [21, 22, 23, 24]],
                                             [[25, 26, 27, 28], [29, 30, 31, 32]]]
        readings = []
        def memory_info(*args):
            readings.append('sample')
            free = 2 * dev._GIB if low_before_copy and len(readings) >= 3 else 6 * dev._GIB
            return free, 8 * dev._GIB
        peak_readings = []
        def peak_allocated(*args):
            index = len(peak_readings) + 1
            value = ({5: 400, 6: 500, 7: 450}.get(index, 0) if peak_regression else 0)
            peak_readings.append(value)
            return value
        def peak_reserved(*args):
            return peak_readings[-1]
        original_check = dev._check_serial_reference
        def observe_reference(module, value, width):
            references.append(weakref.ref(value))
            return original_check(module, value, width)
        original_release = HostClaim.release
        def check_release(claim):
            if claim.info().stage == 'references-and-scratch':
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
            return original_release(claim)
        patches = [
            patch.object(dev.OwnerSession, 'check'),
            patch.object(dev.OwnerSession, 'verify_runtime_binding'),
            patch.object(dev, '_configure', side_effect=lambda *args: torch.set_flush_denormal(False)),
            patch.object(dev, '_live_runtime', return_value=synthetic_runtime()),
            patch.object(dev, '_runtime_uuid', return_value='fake'),
            patch.object(pretrained, 'load_distilgpt2', side_effect=load),
            patch.object(cuda_cleanup, 'clear_cublas_after_drain',
                         return_value={'before': {'allocated_bytes': 0, 'reserved_bytes': 0},
                                       'after': {'allocated_bytes': 0, 'reserved_bytes': 0},
                                       'api': 'PyTorch 2.8.0+cu128 private _cuda_clearCublasWorkspaces'}),
            patch.object(torch.cuda, 'is_available', return_value=True),
            patch.object(torch.cuda, 'device_count', return_value=1),
            patch.object(torch.cuda, 'get_device_properties', return_value=properties),
            patch.object(torch.cuda, 'get_allocator_backend', return_value='native'),
            patch.object(torch.cuda, 'set_per_process_memory_fraction'),
            patch.object(torch.cuda, 'get_per_process_memory_fraction', return_value=.25),
            patch.object(torch.cuda, 'manual_seed'),
            patch.object(torch.cuda, 'synchronize'),
            patch.object(torch.cuda, 'reset_peak_memory_stats'),
            patch.object(torch.cuda, 'mem_get_info', side_effect=memory_info),
            patch.object(torch.cuda, 'memory_allocated', side_effect=lambda *a: 400
                         if peak_regression and len(peak_readings) >= 4 else 0),
            patch.object(torch.cuda, 'memory_reserved', side_effect=lambda *a: 400
                         if peak_regression and len(peak_readings) >= 4 else 0),
            patch.object(torch.cuda, 'max_memory_allocated', side_effect=peak_allocated),
            patch.object(torch.cuda, 'max_memory_reserved', side_effect=peak_reserved),
        ]
        if indistinguishable:
            patches += [patch.object(dev, '_check_serial_reference', new=observe_reference),
                        patch.object(HostClaim, 'release', new=check_release)]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if low_before_copy:
                with self.assertRaisesRegex(RuntimeError, 'before model copy'):
                    dev._run(object(), object(), torch, SimpleNamespace(), cache=Path(directory),
                        protocol=protocol, run_dir=Path(directory), driver='synthetic',
                        source_hashes={'synthetic': 'a' * 64}, run_id='synthetic', _worker_device='cpu')
                self.assertEqual(conversions, [])
                self.assertIsNotNone(dev._ROOTS[3])
                self.assertIsNotNone(dev._ROOTS[4])
                return None
            if peak_regression:
                with self.assertRaisesRegex(RuntimeError, 'sampled forward failed'):
                    dev._run(object(), object(), torch, SimpleNamespace(), cache=Path(directory),
                        protocol=protocol, run_dir=Path(directory), driver='synthetic',
                        source_hashes={'synthetic': 'a' * 64}, run_id='synthetic', _worker_device='cpu')
                records = [json.loads(path.read_bytes()) for path in Path(directory).glob('memory-*.json')]
                self.assertTrue(any(record['peak_allocated_bytes'] == 450 for record in records))
                self.assertIsNotNone(dev._ROOTS[15])
                return None
            result = dev._run(object(), object(), torch, SimpleNamespace(), cache=Path(directory),
                protocol=protocol, run_dir=Path(directory), driver='synthetic',
                source_hashes={'synthetic': 'a' * 64}, run_id='synthetic', _worker_device='cpu')
            if promotion:
                run_dir = Path(directory)
                supervised = SimpleNamespace(run_id='supervisor', pid=123)
                identity = {'run_id': 'synthetic', 'supervisor_run_id': 'supervisor', 'pid': 123}
                raw_protocol = json.dumps(protocol, sort_keys=True).encode('utf-8')
                protocol_sha = hashlib.sha256(raw_protocol).hexdigest()
                assets = [{'name': name, 'bytes': size, 'sha256': digest}
                          for name, (size, digest) in sorted(pretrained.ASSETS.items())]
                records = {
                    'owner.json': {'schema': 'grouped-gpu-owner-v1', **identity},
                    'candidate.json': {'schema': 'grouped-gpu-candidate-v1', **identity, 'result': result},
                    'supervision.json': {'schema': 'grouped-gpu-supervision-v1', **identity,
                                         'exit_code': 0, 'clean': True, 'timed_out': False},
                    'protocol-source.json': {'schema': 'grouped-gpu-protocol-source-v1', 'sha256': protocol_sha,
                                             'protocol_utf8': raw_protocol.decode('utf-8')},
                    'asset-verification.json': {'schema': 'grouped-gpu-assets-v1', 'run_id': 'synthetic',
                        'model': protocol['model'], 'revision': protocol['revision'], 'verified': True,
                        'assets': assets},
                    'private-selected-device.json': {'schema': 'grouped-gpu-selected-device-v1',
                                                     'run_id': 'synthetic', 'physical_gpu_uuid': 'fake'},
                }
                for name, value in records.items():
                    dev._write_json(run_dir / name, value)
                inventory, promoted, _ = dev._verify_promotion(run_dir, 'synthetic', supervised, 'fake',
                    protocol, protocol_sha, assets, {'synthetic': 'a' * 64}, _expected_device='cpu')
                self.assertEqual(promoted['status'], 'passed')
                self.assertIn('private-runtime.json', inventory)
                self.assertTrue(all(set(item) == {'bytes', 'sha256'} for item in inventory.values()))
                private = json.loads(json.dumps(promoted))
                private['unexpected_private_field'] = 'private-sentinel'
                private['holdout'][0]['comparisons'][0]['own_reference']['private'] = 'private-sentinel'
                self.assertNotIn('private-sentinel', json.dumps(dev._public_measurement(private)))
                observation_path = run_dir / 'observations.json'
                observation = json.loads(observation_path.read_bytes())
                candidate_path = run_dir / 'candidate.json'
                candidate = json.loads(candidate_path.read_bytes())
                index = next(i for i, item in enumerate(observation['memory'])
                             if item.get('admission_kind') == 'precopy')
                original_memory = dict(observation['memory'][index])
                impossible = dict(original_memory, device_free_bytes=0, admissible_estimate_bytes=0)
                observation['memory'][index] = impossible
                candidate['result']['memory'][index] = impossible
                observation_path.write_text(json.dumps(observation), encoding='utf-8')
                candidate_path.write_text(json.dumps(candidate), encoding='utf-8')
                memory_path = run_dir / f'memory-{index:03d}.json'
                memory_path.write_text(json.dumps(impossible), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, 'incomplete or inconsistent'):
                    dev._verify_promotion(run_dir, 'synthetic', supervised, 'fake', protocol, protocol_sha,
                                          assets, {'synthetic': 'a' * 64}, _expected_device='cpu')
                observation['memory'][index] = original_memory
                candidate['result']['memory'][index] = original_memory
                candidate_path.write_text(json.dumps(candidate), encoding='utf-8')
                memory_path.write_text(json.dumps(original_memory), encoding='utf-8')
                observation['holdout'][0]['comparisons'][0]['job_id'] = 'tampered'
                observation_path.write_text(json.dumps(observation), encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, 'incomplete or inconsistent'):
                    dev._verify_promotion(run_dir, 'synthetic', supervised, 'fake', protocol, protocol_sha,
                                          assets, {'synthetic': 'a' * 64}, _expected_device='cpu')
        return result

    def test_low_free_memory_after_cpu_load_prevents_device_copy(self):
        self._synthetic_owner(low_before_copy=True)

    def test_intermediate_peak_drop_is_recorded_and_quarantines_owner(self):
        self._synthetic_owner(peak_regression=True)

    def test_uninformative_rows_are_disposed_before_control_claim(self):
        result = self._synthetic_owner(indistinguishable=True)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(all(group['status'] == 'uninformative' for group in result['holdout']))

    def test_synthetic_swapped_group_rows_fail_end_to_end(self):
        result = self._synthetic_owner(swap=True)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(all(case['status'] == 'mismatch'
                            for group in result['holdout'] for case in group['comparisons']))

    def test_synthetic_owner_runs_real_cpu_worker_and_broker_chain(self):
        result = self._synthetic_owner(promotion=True)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['holdout']), 2)
        for group in result['holdout']:
            self.assertEqual(group['entry_shapes'], [[2, 4]])
            self.assertEqual([case['status'] for case in group['comparisons']], ['matched', 'matched'])
            self.assertEqual(len({case['job_id'] for case in group['comparisons']}), 2)
        admission = [item for item in result['memory'] if item['stage'] == 'admission']
        self.assertGreater(len(admission), 0)
        self.assertTrue(all(item['reserve_bytes'] >= dev._GIB for item in admission))


if __name__ == '__main__':
    unittest.main()
