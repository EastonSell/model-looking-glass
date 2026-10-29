import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from looking_glass import bert_gpu_family_controller as subject


class ClosedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = 'a' * 32
        self.commit = 'b' * 40
        self.supervision = SimpleNamespace(run_id='supervisor-id', pid=31415)
        self.source = {'example.py': 'c' * 64}
        self.cases = [{'row_id': 'example', 'value': 1.0}]
        self.checked = {'cases': self.cases, 'site_means': {'block': 0.5}, 'controls': [{}]}
        self.event_hash = 'd' * 64
        total = 8 * subject._GIB
        cap = int(.25 * total)
        idle = {'device_free_bytes': 6 * subject._GIB, 'device_total_bytes': total,
                'allocated_bytes': 0, 'reserved_bytes': 0,
                'peak_allocated_bytes': 64 * subject._MIB, 'peak_reserved_bytes': 128 * subject._MIB}
        active = dict(idle, allocated_bytes=64 * subject._MIB, reserved_bytes=128 * subject._MIB)
        stages = [(name, None) for name in ('before_model_load', 'before_model_copy',
                  'after_model_copy', 'after_initial_state')]
        for index in range(1, 141):
            stages.extend((('before', index), ('after', index)))
        stages.extend((name, None) for name in ('after_final_state',
                      'cleanup_before_workspace_clear', 'after_cleanup'))
        self.memory = [{'stage': stage, 'index': index, 'peak_scope': subject._PEAK_SCOPE,
                        'memory': dict(idle if stage == 'after_cleanup' else active)}
                       for stage, index in stages]
        config = {'device': 'cuda:0', 'threads': 1, 'interop_threads': 1,
            'deterministic_algorithms': True, 'matmul_tf32': False, 'cudnn_tf32': False,
            'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'overwrite_module_params_on_conversion': False, 'swap_module_params_on_conversion': False}
        metadata = {'model': subject.MODEL_ID, 'revision': subject.REVISION,
            'assets': {name: {'size': size, 'sha256': digest}
                       for name, (size, digest) in subject.ASSETS.items()},
            'model_class': 'DistilBertForMaskedLM', 'tokenizer_class': 'DistilBertTokenizerFast',
            'transformers_version': '4.57.3', 'tokenizers_version': '0.22.2',
            'safetensors_version': '0.8.0', 'attention_implementation': 'eager',
            'parameter_count': 66985530, 'cpu_tied_weight_verified': True}
        profile = {'forward_count': 140, 'journal_sha256': self.event_hash, 'record_count': 328,
            'expected_device': 'cuda:0', 'live_cases': self.cases, 'audited_cases': self.cases,
            'site_means': self.checked['site_means'], 'control_count': 1}
        self.measurement = {'profile': profile, 'model_metadata': metadata,
            'model_state_sha256': 'e' * 64, 'execution_config': config,
            'device_name': 'NVIDIA GeForce RTX 2070 SUPER', 'device_total_bytes': total,
            'allocator_backend': 'native', 'allocator_fraction': .25, 'allocator_cap_bytes': cap,
            'whole_run_commitment_bytes': cap + subject._HEADROOM,
            'gpu_trace_claim_bytes': 65536, 'host_trace_claim_bytes': 64 * subject._MIB,
            'peak_scope': subject._PEAK_SCOPE, 'runtime_uuid_verified': True,
            'cleanup': dict(idle), 'model_weakref_alive': False,
            'budget_active_before_candidate_write': 1, 'torch_version': '2.8.0+cu128',
            'cuda_build': '12.8', 'runtime_versions': dict(subject._RUNTIME), 'seed': 17,
            'cublas_workspace_config': ':4096:8', 'memory_record_count': 287,
            'manifest_sha256': subject._MANIFEST_SHA, 'source_sha256': self.source}
        identity = {'run_id': self.run_id, 'supervisor_run_id': self.supervision.run_id,
                    'pid': self.supervision.pid, 'source_commit': self.commit}
        self.owner = {'schema': 'bert-gpu-family-owner-v1', 'manifest_sha256': subject._MANIFEST_SHA,
                      **identity}
        self.candidate = {'schema': 'bert-gpu-family-candidate-v1', 'measurement': self.measurement,
                          **identity}
        self.final = {'schema': 'bert-gpu-family-final-v1', 'active_memory_leases': 0, **identity}

    def write_records(self):
        raw = b''.join((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode()
                       for row in self.memory)
        (self.root / 'memory.jsonl').write_bytes(raw)
        self.measurement['memory_journal_sha256'] = hashlib.sha256(raw).hexdigest()
        (self.root / 'owner.json').write_text(json.dumps(self.owner), encoding='utf-8')
        raw_candidate = json.dumps(self.candidate).encode()
        (self.root / 'candidate.json').write_bytes(raw_candidate)
        self.final['candidate_sha256'] = hashlib.sha256(raw_candidate).hexdigest()
        (self.root / 'final.json').write_text(json.dumps(self.final), encoding='utf-8')

    def validate(self):
        with patch.object(subject, '_read_profile_journal', return_value=([{}] * 328, self.event_hash)):
            return subject._validate_bundle(self.root, run_id=self.run_id, supervision=self.supervision,
                source_commit=self.commit, source_hashes=self.source, manifest={},
                manifest_sha=subject._MANIFEST_SHA, audit=lambda *args, **kwargs: self.checked)

    def test_consistent_complete_bundle_passes(self):
        self.write_records()
        self.assertEqual(self.validate()['profile']['forward_count'], 140)

    def test_foreign_process_or_unreleased_claim_cannot_promote(self):
        for name, change in (('pid', lambda: self.owner.__setitem__('pid', 999)),
                             ('lease', lambda: self.final.__setitem__('active_memory_leases', False)),
                             ('source', lambda: self.candidate.__setitem__('source_commit', 'f' * 40))):
            with self.subTest(name=name):
                self.setUp()
                change()
                self.write_records()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_changed_candidate_or_unsafe_cleanup_cannot_promote(self):
        self.write_records()
        self.final['candidate_sha256'] = 'f' * 64
        (self.root / 'final.json').write_text(json.dumps(self.final), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate()
        self.write_records()
        self.memory[-1]['memory']['allocated_bytes'] = 1
        self.measurement['cleanup']['allocated_bytes'] = 1
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_memory_order_invalid_counters_and_peaks_are_rejected(self):
        for mutation in ('order', 'counter', 'peak'):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == 'order':
                    self.memory[10]['index'] = 8
                elif mutation == 'counter':
                    self.memory[10]['memory']['device_free_bytes'] = False
                else:
                    self.memory[10]['memory']['peak_reserved_bytes'] = 0
                self.write_records()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_live_summary_type_difference_or_journal_mismatch_is_rejected(self):
        self.measurement['profile']['live_cases'] = [{'row_id': 'example', 'value': True}]
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()
        self.measurement['profile']['live_cases'] = self.cases
        self.measurement['profile']['journal_sha256'] = 'f' * 64
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_nonfinite_and_incomplete_records_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":1e400}', b'{"a":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                subject._json(raw)
        (self.root / 'memory.jsonl').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            subject._read_memory(self.root / 'memory.jsonl')
        for raw in (b'\n' * 100000, b'{}\r\n' * 287):
            (self.root / 'memory.jsonl').write_bytes(raw)
            with self.assertRaises(ValueError):
                subject._read_memory(self.root / 'memory.jsonl')

    def test_source_or_manifest_mismatch_cannot_promote(self):
        self.measurement['source_sha256'] = {'example.py': 'f' * 64}
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()
        self.measurement['source_sha256'] = self.source
        self.owner['manifest_sha256'] = 'f' * 64
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()

    def test_metadata_version_must_match_frozen_runtime(self):
        for key in ('tokenizers_version', 'safetensors_version'):
            with self.subTest(key=key):
                self.setUp()
                self.measurement['model_metadata'][key] = 'incorrect'
                self.write_records()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_public_projection_excludes_unknown_private_fields(self):
        self.measurement['extra_private_path'] = 'private-path'
        self.measurement['model_metadata']['extra_private_path'] = 'private-path'
        self.measurement['cleanup']['extra_private_path'] = 'private-path'
        self.memory[-1]['memory']['extra_private_path'] = 'private-path'
        self.write_records()
        projected = subject._public_measurement(self.validate())
        self.assertNotIn('private-path', json.dumps(projected))

    def test_runtime_identity_cannot_alias_an_integer_seed(self):
        self.measurement['seed'] = True
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()

    def test_model_dependency_versions_must_match_runtime_identity(self):
        self.measurement['model_metadata']['tokenizers_version'] = 'private-sentinel'
        self.write_records()
        with self.assertRaises(ValueError):
            self.validate()

    def test_mixed_terminal_or_uncertain_partial_is_incomplete(self):
        required = ('started.json', 'supervision.json', 'owner.json', 'candidate.json',
                    'final.json', 'events.jsonl', 'memory.jsonl')
        inventory = [{'name': name, 'size': 1, 'sha256': 'a' * 64} for name in required]
        subject._validate_complete_inventory(inventory)
        for extra in ('failure.json', 'interrupted.json', 'failure.json.partial'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                subject._validate_complete_inventory(inventory + [{'name': extra, 'size': 1,
                                                                  'sha256': 'b' * 64}])
        # A retained same-bytes partial is a documented completed hard link.
        subject._validate_complete_inventory(inventory + [{'name': 'final.json.partial', 'size': 1,
                                                           'sha256': 'a' * 64}])


class SupervisionTests(unittest.TestCase):
    def test_unclean_child_is_quarantined_and_cannot_promote(self):
        self.check_child(clean=False)

    def test_clean_child_requires_bundle_audit(self):
        self.check_child(clean=True)

    def test_changed_start_record_cannot_promote(self):
        self.check_child(clean=True, alter_start=True)

    def test_changed_journal_after_audit_cannot_promote(self):
        self.check_child(clean=True, alter_after_audit=True)

    def check_child(self, *, clean, alter_start=False, alter_after_audit=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'work').mkdir()
            cache = root / 'cache'
            cache.mkdir()
            site = root / 'site'
            site.mkdir()
            output = root / 'result.json'
            commit = 'a' * 40
            hashes = {path: 'b' * 64 for path in subject.SOURCE_PATHS}
            supervision = SimpleNamespace(run_id='supervisor', pid=123,
                exit_code=0 if clean else 1, clean=clean, timed_out=False, reason='test')
            identity = SimpleNamespace(lock_path=root / 'lock', marker_path=root / 'marker')
            def fake_launch(command, **_kwargs):
                run_dir = Path(command[command.index('--owner') + 1])
                if alter_start:
                    started = json.loads((run_dir / 'started.json').read_text())
                    started['source_commit'] = 'f' * 40
                    (run_dir / 'started.json').write_text(json.dumps(started), encoding='utf-8')
                if clean:
                    for name in ('owner.json', 'candidate.json', 'final.json', 'events.jsonl', 'memory.jsonl'):
                        (run_dir / name).write_bytes(b'checked evidence')
                return supervision
            launch = Mock(side_effect=fake_launch)
            def fake_validate(run_dir, **kwargs):
                for name in ('owner.json', 'candidate.json', 'final.json', 'events.jsonl', 'memory.jsonl'):
                    kwargs['verified_hashes'][name] = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                if alter_after_audit:
                    (run_dir / 'events.jsonl').write_bytes(b'changed after validation')
                return {'profile': 'audited'}
            validate = Mock(side_effect=fake_validate)
            snapshot = ('GPU-test', {'timestamp_utc': 'test'})
            with patch.object(subject._family_owner, 'FAMILY_PROFILE_ENABLED', True), \
                 patch.object(subject, '_workspace_root', return_value=root), \
                 patch.object(subject, '_verify_startup'), \
                 patch.object(subject, '_require_pristine_dependencies'), \
                 patch.object(subject.subprocess, 'run', return_value=SimpleNamespace(stdout=commit)), \
                 patch.object(subject, '_verify_source', return_value=hashes), \
                 patch.object(subject, '_load_manifest', return_value=({}, subject._MANIFEST_SHA)), \
                 patch.object(subject, '_verify_offline_cache'), \
                 patch.object(subject, '_smi_snapshot', return_value=snapshot), \
                 patch.object(subject, '_child_environment', return_value={'CUDA_VISIBLE_DEVICES': 'GPU-test'}), \
                 patch.object(subject, 'resolve_device_identity', return_value=identity), \
                 patch.object(subject, 'run_supervised', launch), \
                 patch.object(subject, '_owner_presence_after_exit',
                              return_value={'query': 'ok', 'owner_pid_listed': False}), \
                 patch.object(subject, '_validate_bundle', validate), \
                 patch.object(subject, '_public_measurement', side_effect=lambda data: data):
                result = subject.run_profile(workspace_root=root, cache_dir=cache, output=output,
                    site_packages=site, smi='fake-smi', git_exe=sys.executable)
            self.assertEqual(result['status'], 'incomplete' if alter_start or alter_after_audit else
                             'success' if clean else 'quarantined')
            self.assertEqual(validate.call_count, int(clean and not alter_start))
            self.assertEqual(launch.call_count, 1)
            self.assertFalse(launch.call_args.kwargs['inherit_environment'])
            self.assertEqual(json.loads(output.read_text())['status'], result['status'])

    def test_disabled_gate_and_wrong_shared_root_stop_before_telemetry(self):
        with patch.object(subject._family_owner, 'FAMILY_PROFILE_ENABLED', False), \
             patch.object(subject, '_smi_snapshot') as query, self.assertRaises(RuntimeError):
            subject.run_profile(workspace_root='unused', cache_dir='unused', output='unused',
                                site_packages='unused', smi='unused', git_exe='unused')
        query.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in ('work/teams/branch', 'work/frozen/snapshot', 'work/primary'):
                repo = root / relative
                repo.mkdir(parents=True)
                self.assertEqual(subject._workspace_root(repo, root), root.resolve())
                with self.assertRaises(ValueError):
                    subject._workspace_root(repo, repo)

    def test_interruption_is_propagated_before_any_post_exit_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'work').mkdir()
            cache = root / 'cache'
            cache.mkdir()
            site = root / 'site'
            site.mkdir()
            commit = 'a' * 40
            hashes = {path: 'b' * 64 for path in subject.SOURCE_PATHS}
            snapshot = ('GPU-test', {'timestamp_utc': 'test'})
            identity = SimpleNamespace(lock_path=root / 'lock', marker_path=root / 'marker')
            with patch.object(subject._family_owner, 'FAMILY_PROFILE_ENABLED', True), \
                 patch.object(subject, '_workspace_root', return_value=root), \
                 patch.object(subject, '_verify_startup'), \
                 patch.object(subject, '_require_pristine_dependencies'), \
                 patch.object(subject.subprocess, 'run', return_value=SimpleNamespace(stdout=commit)), \
                 patch.object(subject, '_verify_source', return_value=hashes), \
                 patch.object(subject, '_load_manifest', return_value=({}, subject._MANIFEST_SHA)), \
                 patch.object(subject, '_verify_offline_cache'), \
                 patch.object(subject, '_smi_snapshot', return_value=snapshot) as query, \
                 patch.object(subject, '_child_environment', return_value={}), \
                 patch.object(subject, 'resolve_device_identity', return_value=identity), \
                 patch.object(subject.secrets, 'token_hex', return_value='c' * 32), \
                 patch.object(subject, 'run_supervised', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    subject.run_profile(workspace_root=root, cache_dir=cache, output=root / 'result.json',
                        site_packages=site, smi='fake-smi', git_exe=sys.executable)
            self.assertEqual(query.call_count, 1)
            self.assertFalse((root / 'result.json').exists())
            interrupted = root / 'work' / ('bert-gpu-family-' + 'c' * 32) / 'interrupted.json'
            self.assertEqual(json.loads(interrupted.read_text())['status'], 'unverified')


class BootstrapTests(unittest.TestCase):
    def test_checkout_cannot_be_used_as_dependency_site(self):
        checkout = Path(__file__).resolve().parents[1]
        for overlap in (checkout, checkout.parent):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as temp:
                prefix = Path(temp) / 'prefix'
                prefix.mkdir()
                result = subprocess.run([sys.executable, '-I', '-S', '-B', '-X',
                    'pycache_prefix=' + str(prefix), str(checkout / 'scripts' / 'bert_gpu_family_bootstrap.py'),
                    '--bootstrap-site', str(overlap), '--help'], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('dependency directory', result.stderr)

    def test_sourceless_dependency_never_executes(self):
        source_bootstrap = Path(__file__).resolve().parents[1] / 'scripts' / 'bert_gpu_family_bootstrap.py'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts, package, site, prefix = (root / name for name in ('checkout/scripts',
                'checkout/looking_glass', 'site', 'prefix'))
            for path in (scripts, package, site, prefix):
                path.mkdir(parents=True)
            shutil.copyfile(source_bootstrap, scripts / source_bootstrap.name)
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'bert_gpu_family_controller.py').write_text('import torch\n', encoding='utf-8')
            sentinel = root / 'executed'
            fake = site / 'torch.py'
            fake.write_text('from pathlib import Path\nPath(' + repr(str(sentinel)) +
                            ').write_text("executed")\n', encoding='utf-8')
            py_compile.compile(str(fake), cfile=str(site / 'torch.pyc'), doraise=True)
            fake.unlink()
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-X',
                'pycache_prefix=' + str(prefix), str(scripts / source_bootstrap.name),
                '--bootstrap-site', str(site)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('sourceless dependency bytecode', result.stderr)
            self.assertFalse(sentinel.exists())


if __name__ == '__main__':
    unittest.main()
