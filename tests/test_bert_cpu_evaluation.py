"""CPU-only integration checks for the still-disabled BERT evaluation path."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass import bert_agreement_study as study
from looking_glass.bert_agreement_cases import build_manifest, manifest_digest
from looking_glass.bert_agreement_exposures_v3 import canonical_snapshot_bytes
from looking_glass.bert_evaluation_ledger import (EvaluationLedger, read_record, root_identity,
                                                  verify_reservation, write_record)


class CpuEvaluationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bert-evaluation-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo, self.cache = self.root / 'repo', self.root / 'cache'
        self.ledger_root, self.custody = self.root / 'ledger', self.root / 'custody'
        self.controller_custody = self.root / 'controller-custody'
        for directory in (self.repo, self.cache, self.ledger_root, self.custody, self.controller_custody):
            directory.mkdir()
        (self.repo / 'experiments').mkdir()
        _, ledger_sha = root_identity(self.ledger_root)
        cpu = study._cpu_study_binding(self.custody)
        self.manifest = build_manifest()
        self.manifest_sha = manifest_digest()
        exposure = canonical_snapshot_bytes()
        self.profile = {'schema': 'bert-agreement-cpu-evaluation-v1', 'ledger_root_sha256': ledger_sha,
                        'custody_root_sha256': cpu['custody_root_sha256'], 'manifest_sha256': self.manifest_sha,
                        'controller_custody_root_sha256': study._cpu_study_binding(self.controller_custody)['custody_root_sha256'],
                        'exposure_sha256': hashlib.sha256(exposure).hexdigest(), 'evaluation_rows': 64,
                        'maximum_attempts': 32}
        self.profile_path = self.repo / 'experiments/bert_agreement_cpu_evaluation_v1.json'
        self.exposure_path = self.repo / 'experiments/bert_agreement_exposures_v3.json'
        self.profile_path.write_bytes(json.dumps(self.profile, sort_keys=True).encode('ascii') + b'\n')
        self.exposure_path.write_bytes(exposure)
        self.hashes = {'experiments/bert_agreement_cpu_evaluation_v1.json':
                       hashlib.sha256(self.profile_path.read_bytes()).hexdigest(),
                       'experiments/bert_agreement_exposures_v3.json': hashlib.sha256(exposure).hexdigest(),
                       'docs/BERT_AGREEMENT_V1.md': 'c' * 64}

    def binding(self, manifest=None):
        return study._evaluation_binding(self.repo, 'a' * 40, self.hashes, manifest or self.manifest,
                                         self.manifest_sha, self.custody, self.ledger_root, self.controller_custody)

    def test_binding_uses_canonical_manifest_and_pinned_exposure(self):
        bound = self.binding()
        self.assertEqual(len(bound['rows']), 64)
        self.assertEqual(bound['ledger_root_sha256'], self.profile['ledger_root_sha256'])
        changed = json.loads(json.dumps(self.manifest))
        changed['rows'][-1]['input_ids'][2] += 1
        with self.assertRaises(RuntimeError):
            self.binding(changed)
        self.exposure_path.write_bytes(self.exposure_path.read_bytes() + b' ')
        with self.assertRaises(RuntimeError):
            self.binding()
        self.hashes['experiments/bert_agreement_exposures_v3.json'] = hashlib.sha256(self.exposure_path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            self.binding()

    def test_changed_launcher_is_rejected_even_when_git_hides_it(self):
        name = 'scripts/bert_agreement_cpu_evaluation.py'
        self.assertIn(name, study.SOURCE_PATHS)
        subprocess.run(['git', 'init'], cwd=self.repo, check=True, capture_output=True)
        path = self.repo / name
        path.parent.mkdir()
        path.write_bytes(b'print(1)\n')
        subprocess.run(['git', 'add', name], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-m', 'fixture'], cwd=self.repo, check=True, capture_output=True)
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        study.verify_git_sources(self.repo, commit, (name,))
        subprocess.run(['git', 'update-index', '--assume-unchanged', name], cwd=self.repo,
                       check=True, capture_output=True)
        path.write_bytes(b'print(2)\n')
        with self.assertRaises(RuntimeError):
            study.verify_git_sources(self.repo, commit, (name,))

    def test_draft_rejects_evaluation_before_reserving_or_spawning(self):
        output = self.ledger_root / ('1' * 32)
        with patch.object(study, '_require_bootstrap'), patch.object(study, 'EVALUATION_ENABLED', False):
            with self.assertRaises(RuntimeError):
                study.run_cpu_controller(repo=self.repo, cache=self.cache, output_dir=output, commit='a' * 40,
                                         split='evaluation', custody_root=self.custody, ledger_root=self.ledger_root,
                                         controller_custody_root=self.controller_custody)
        self.assertFalse(output.exists())

    def test_ungranted_direct_worker_fails_before_manifest_or_model(self):
        output = self.ledger_root / ('2' * 32)
        output.mkdir()
        with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout'), \
             patch.object(study, 'EVALUATION_ENABLED', True), patch.object(study, '_checked_manifest') as checked, \
             patch.dict('os.environ', {}, clear=True):
            result = study._owner_main(repo=self.repo, cache=self.cache, run_dir=output, commit='a' * 40,
                                       git_exe='git', split='evaluation', family=None, run_id='2' * 32,
                                       custody_root=self.custody, ledger_root=self.ledger_root,
                                       controller_custody_root=self.controller_custody)
        self.assertEqual(result, 1)
        checked.assert_not_called()
        self.assertEqual(read_record(output / 'owner-failure.json')[0]['stage'], 'custody_claim')
        self.assertFalse((output / 'events.jsonl').exists())

    def test_ungranted_controller_fails_before_preflight_or_reservation(self):
        output = self.ledger_root / ('5' * 32)
        with patch.object(study, '_require_bootstrap'), patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.object(study, '_checked_manifest') as checked, patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(Exception):
                study.run_cpu_controller(repo=self.repo, cache=self.cache, output_dir=output, commit='a' * 40,
                                         split='evaluation', custody_root=self.custody, ledger_root=self.ledger_root,
                                         controller_custody_root=self.controller_custody)
        checked.assert_not_called()
        self.assertFalse(output.exists())

    def test_controller_keeps_reservation_locked_and_exit_is_recorded_after_return(self):
        bound = self.binding()
        run_id = '3' * 32
        output = self.ledger_root / run_id
        def fake_run(**kwargs):
            context = kwargs['ledger_context']
            context['ledger']._check()
            reservation, digest = verify_reservation(output, run_id, bound)
            self.assertEqual(digest, context['reservation_sha256'])
            self.assertEqual(reservation['prior_attempts'], [])
            write_record(output / 'result.json', {'schema': 'bert-agreement-controller-result-v1',
                         'run_id': run_id, 'source_commit': 'a' * 40, 'status': 'incomplete',
                         'owner_state': 'not_started'})
            return {'status': 'incomplete', 'owner_state': 'not_started'}
        session = SimpleNamespace(run_id='f' * 32, check=lambda lease: None, attest_clean=lambda lease: None)
        with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout', return_value=self.repo), \
             patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.object(study, '_checked_manifest', return_value=(self.hashes, self.manifest, self.manifest_sha)), \
             patch.object(study, '_run_cpu_controller', side_effect=fake_run), \
             patch('looking_glass.cpu_study_supervision.claim_cpu_study', return_value=(session, object())), \
             patch('shutil.which', return_value='git'):
            result = study.run_cpu_controller(repo=self.repo, cache=self.cache, output_dir=output, commit='a' * 40,
                                               split='evaluation', custody_root=self.custody, ledger_root=self.ledger_root,
                                               controller_custody_root=self.controller_custody)
            self.assertEqual(result['status'], 'incomplete')
            study.record_cpu_evaluation_exit(repo=self.repo, commit='a' * 40, output_dir=output, run_id=run_id,
                                             supervision={'scope': 'cooperative-cpu-controller-custody-only',
                                                          'run_id': 'f' * 32, 'pid': os.getpid(),
                                                          'exit_code': 1, 'clean': True, 'timed_out': False,
                                                          'reason': 'synthetic test'},
                                             custody_root=self.custody, ledger_root=self.ledger_root,
                                             controller_custody_root=self.controller_custody)
        ledger = EvaluationLedger.acquire(self.ledger_root, bound['ledger_root_sha256'], bound)
        prior, complete = ledger.prior_attempts()
        self.assertEqual(prior[0]['run_id'], run_id)
        self.assertEqual(prior[0]['status'], 'incomplete')
        self.assertFalse(complete)
        ledger.release()

    def test_known_inner_quarantine_rejects_before_reservation_and_attests_no_start(self):
        output = self.ledger_root / ('7' * 32)
        attestations, lease = [], object()
        session = SimpleNamespace(run_id='f' * 32, check=lambda value: None,
                                  attest_clean=lambda value: attestations.append(value))
        with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout', return_value=self.repo), \
             patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.object(study, '_checked_manifest', return_value=(self.hashes, self.manifest, self.manifest_sha)), \
             patch.object(study, '_run_cpu_controller') as run, \
             patch('looking_glass.cpu_study_supervision.claim_cpu_study', return_value=(session, lease)), \
             patch('shutil.which', return_value='git'):
            for name in ('cpu-study.unsafe.json', 'cpu-study.unsafe.json.attestation'):
                with self.subTest(name=name):
                    path = self.custody / name
                    path.write_bytes(b'{}\n')
                    with self.assertRaisesRegex(RuntimeError, 'no evaluation attempt was reserved'):
                        study.run_cpu_controller(repo=self.repo, cache=self.cache, output_dir=output, commit='a' * 40,
                                                 split='evaluation', custody_root=self.custody,
                                                 ledger_root=self.ledger_root,
                                                 controller_custody_root=self.controller_custody)
                    path.unlink()
            run.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(attestations, [lease, lease])
        binding = self.binding()
        ledger = EvaluationLedger.acquire(self.ledger_root, binding['ledger_root_sha256'], binding)
        self.assertEqual(ledger.prior_attempts(), ([], False))
        ledger.release()

    def test_launcher_waits_for_controller_then_records_actual_exit(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/bert_agreement_cpu_evaluation.py'
        spec = importlib.util.spec_from_file_location('bert_cpu_eval_launcher_test', script)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        actual_repo = script.parent.parent
        site = self.root / 'site'
        site.mkdir()
        from looking_glass.supervisor import SupervisionResult
        commands, records = [], []
        def fake_run(command, **kwargs):
            commands.append((command, kwargs))
            Path(command[command.index('--output') + 1]).mkdir()
            return SupervisionResult('6' * 32, 123, 0, True, False, 'synthetic clean')
        args = ['--repo', str(actual_repo), '--cache', str(self.cache), '--site', str(site),
                '--commit', 'a' * 40, '--custody-root', str(self.custody), '--ledger-root', str(self.ledger_root),
                '--controller-custody-root', str(self.controller_custody),
                '--work-dir', str(self.root / 'launcher-work')]
        with patch.object(launcher, '_require_isolation'), \
             patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.dict(os.environ, {'LOOKING_GLASS_BERT_SITE': str(site)}), \
             patch('looking_glass.cpu_study_supervision.run_cpu_study', side_effect=fake_run), \
             patch.object(study, 'record_cpu_evaluation_exit', side_effect=lambda **kwargs: records.append(kwargs)), \
             patch.object(launcher.secrets, 'token_hex', return_value='4' * 32):
            self.assertEqual(launcher.main(args), 0)
        self.assertEqual(len(commands), 1)
        self.assertEqual(len(records), 1)
        command, options = commands[0]
        self.assertEqual(options['study_root'], self.controller_custody)
        self.assertEqual(options['cwd'], self.root / 'launcher-work')
        self.assertEqual(options['env']['TEMP'], str(self.root / 'launcher-work' / 'tmp'))
        self.assertNotIn('SYSTEMDRIVE', options['env'])
        self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertIn('--controller-custody-root', command)
        self.assertEqual(records[0]['run_id'], '4' * 32)
        self.assertEqual(records[0]['supervision']['pid'], 123)
        self.assertEqual(records[0]['supervision']['exit_code'], 0)
        self.assertEqual(records[0]['controller_custody_root'], self.controller_custody)

    def test_launcher_refuses_draft_and_never_records_unclean_supervision(self):
        from looking_glass.supervisor import SupervisionResult
        script = Path(__file__).resolve().parents[1] / 'scripts/bert_agreement_cpu_evaluation.py'
        spec = importlib.util.spec_from_file_location('bert_cpu_eval_launcher_failure_test', script)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        site = self.root / 'site'
        site.mkdir()
        def arguments(name):
            return ['--repo', str(script.parent.parent), '--cache', str(self.cache), '--site', str(site),
                    '--commit', 'a' * 40, '--custody-root', str(self.custody), '--ledger-root', str(self.ledger_root),
                    '--controller-custody-root', str(self.controller_custody), '--work-dir', str(self.root / name)]
        with patch.object(launcher, '_require_isolation'), patch.object(study, 'EVALUATION_ENABLED', False), \
             patch('looking_glass.cpu_study_supervision.run_cpu_study') as run:
            with self.assertRaises(RuntimeError):
                launcher.main(arguments('disabled'))
            run.assert_not_called()
            self.assertFalse((self.root / 'disabled').exists())
        with patch.object(launcher, '_require_isolation'), patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.dict(os.environ, {'LOOKING_GLASS_BERT_SITE': str(site)}), \
             patch('looking_glass.cpu_study_supervision.run_cpu_study',
                   return_value=SupervisionResult('6' * 32, 123, 1, False, True, 'timeout')), \
             patch.object(study, 'record_cpu_evaluation_exit') as record:
            self.assertEqual(launcher.main(arguments('timeout')), 1)
            record.assert_not_called()
        with patch.object(launcher, '_require_isolation'), patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.dict(os.environ, {'LOOKING_GLASS_BERT_SITE': str(site)}), \
             patch('looking_glass.cpu_study_supervision.run_cpu_study',
                   return_value=SupervisionResult('6' * 32, 123, 1, True, False, 'clean no-start')), \
             patch.object(study, 'record_cpu_evaluation_exit') as record:
            self.assertEqual(launcher.main(arguments('no-start')), 1)
            record.assert_not_called()
        nested = arguments('unused')
        nested[-1] = str(site / 'nested-work')
        with patch.object(launcher, '_require_isolation'), patch.object(study, 'EVALUATION_ENABLED', True), \
             patch.dict(os.environ, {'LOOKING_GLASS_BERT_SITE': str(site)}), \
             patch('looking_glass.cpu_study_supervision.run_cpu_study') as run:
            with self.assertRaises(ValueError):
                launcher.main(nested)
            run.assert_not_called()
        self.assertFalse((site / 'nested-work').exists())

    def test_real_bootstrap_launcher_entry_rejects_the_disabled_gate(self):
        site, prefix = self.root / 'site', self.root / 'prefix'
        site.mkdir()
        prefix.mkdir()
        repo = Path(__file__).resolve().parents[1]
        work = self.root / 'launcher-work'
        command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                   str(repo / 'scripts/bert_agreement_bootstrap.py'), '--bootstrap-site', str(site), '--launcher',
                   '--repo', str(repo), '--cache', str(self.cache), '--site', str(site), '--commit', 'a' * 40,
                   '--custody-root', str(self.custody), '--controller-custody-root', str(self.controller_custody),
                   '--ledger-root', str(self.ledger_root), '--work-dir', str(work)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
        child = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn('evaluation remains disabled', child.stderr)
        self.assertFalse(work.exists())
        self.assertFalse((self.controller_custody / 'cpu-study.unsafe.json').exists())

    def test_bootstrap_rejects_checkout_as_dependency_directory(self):
        prefix = self.root / 'prefix'
        prefix.mkdir()
        repo = Path(__file__).resolve().parents[1]
        command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                   str(repo / 'scripts/bert_agreement_bootstrap.py'), '--bootstrap-site', str(repo), '--check-only']
        child = subprocess.run(command, cwd=self.root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
                               capture_output=True, text=True, timeout=30)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn('dependency directory must be separate', child.stderr)


if __name__ == '__main__':
    unittest.main()
