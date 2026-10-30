"""CPU-only failure and custody tests for the BERT evaluation ledger."""
import json
import hashlib
import importlib.metadata
from functools import wraps
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from looking_glass.bert_evaluation_ledger import (
    EvaluationLedger, read_record, record_controller_exit as _record_controller_exit, root_identity,
    reservation_binding, verify_reservation, write_record,
)
from looking_glass.bert_agreement_cases import build_manifest, manifest_digest


_ISOLATED_TEST = r'''
import importlib.util, sys, unittest
assert 'transformers' not in sys.modules
spec = importlib.util.spec_from_file_location('ledger_test_isolated', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'transformers' not in sys.modules
suite = unittest.TestSuite([module.EvaluationLedgerTest(sys.argv[2])])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
'''


def requires_clean_model_dependencies(test):
    """Keep the real controller's model-dependency guard in a fresh process."""
    @wraps(test)
    def run(self):
        if 'transformers' not in sys.modules:
            return test(self)
        source = Path(__file__).resolve()
        result = subprocess.run([sys.executable, '-B', '-c', _ISOLATED_TEST,
                                 str(source), test.__name__], cwd=source.parents[1],
                                capture_output=True, text=True, timeout=90,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return run


def record_controller_exit(root, expected_root_sha256, run_id, binding, exit_code):
    reservation, _ = read_record(Path(root) / run_id / 'reservation.json')
    supervision = {'scope': 'cooperative-cpu-controller-custody-only',
                   'run_id': reservation['controller_supervision_run_id'],
                   'pid': reservation['controller_pid'], 'exit_code': exit_code, 'clean': True,
                   'timed_out': False, 'reason': 'synthetic test'}
    return _record_controller_exit(root, expected_root_sha256, run_id, binding, supervision)


class EvaluationLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bert-ledger-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root, self.identity = root_identity(Path(self.tmp.name).resolve())
        from looking_glass import bert_agreement_study
        manifest = build_manifest()
        rows = [{'row_id': row['row_id'], 'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}
                for row in manifest['rows'] if row['split'] == 'evaluation']
        self.binding = {
            'source_commit': 'a' * 40, 'source_sha256': {'source.py': 'b' * 64},
            'protocol_sha256': 'c' * 64, 'manifest_sha256': manifest_digest(),
            'exposure_sha256': 'e' * 64, 'ledger_root_sha256': self.identity,
            'model': {'model': bert_agreement_study.MODEL_ID, 'revision': bert_agreement_study.REVISION,
                      'assets': {name: {'size': size, 'sha256': digest}
                                 for name, (size, digest) in bert_agreement_study.ASSETS.items()}},
            'cpu_study': {'contract': 'cooperative-cpu-study-v1', 'custody_root_sha256': '0' * 64},
            'controller_cpu_study': {'contract': 'cooperative-cpu-study-v1', 'custody_root_sha256': '1' * 64},
            'rows': rows,
        }
        self.real_audit = bert_agreement_study.audit_event_journal
        self.audit_patch = patch.object(bert_agreement_study, 'audit_event_journal',
                                        side_effect=lambda *a, **kw: {'sha256': kw['expected_sha256'],
                                                                     'families': [], 'summary': {}})
        self.audit_mock = self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)

        # Ledger state-machine fixtures describe a recorded pinned-runtime audit;
        # they are not evidence that the host interpreter reproduced that runtime.
        self.real_runtime_audit = bert_agreement_study.audit_cpu_evaluation_journal
        def synthetic_recorded_audit(manifest, path, expected_sha256):
            if 'transformers' in sys.modules:
                raise RuntimeError('controller journal audit must not import model dependencies')
            audited = bert_agreement_study.audit_event_journal(
                manifest, path, split='evaluation', family=None, mask_id=103,
                hidden_size=768, expected_sha256=expected_sha256)
            return audited, {'python': platform.python_version(), 'torch': '2.8.0+cu128',
                'cuda_initialized_before_after': False,
                'denormals_preserved_calling_thread': True, 'model_forwarded': False}
        runtime_patch = patch.object(bert_agreement_study, 'audit_cpu_evaluation_journal',
                                     side_effect=synthetic_recorded_audit)
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def acquire(self, binding=None, timeout=5):
        ledger = EvaluationLedger.acquire(self.root, self.identity, binding or self.binding, timeout=timeout)
        def release_if_open():
            if not ledger._closed:
                ledger.release()
        self.addCleanup(release_if_open)
        return ledger

    @staticmethod
    def run_id(number):
        return format(number, '032x')

    def test_prior_audit_runs_with_model_dependencies_in_parent(self):
        # Collection of unrelated tests may import the model package. The
        # controller's production guard still requires a clean process.
        code = r'''
import importlib.util, sys, types, unittest
sys.modules['transformers'] = types.ModuleType('transformers')
spec = importlib.util.spec_from_file_location('ledger_test_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.TestSuite([module.EvaluationLedgerTest(
    'test_completed_prior_audit_can_enter_the_real_replay_controller')])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve())],
                                capture_output=True, text=True, timeout=60,
                                cwd=Path(__file__).resolve().parents[1],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_prior_audit_rejects_model_dependencies_in_its_own_process(self):
        audit_cpu_evaluation_journal = self.real_runtime_audit
        with patch.dict(sys.modules, {'transformers': object()}):
            with self.assertRaisesRegex(RuntimeError, 'must not import model dependencies'):
                audit_cpu_evaluation_journal(None, None, None)

    def test_real_audit_rejects_unsupported_runtime_before_reading_journal(self):
        import torch
        with patch.dict(sys.modules):
            sys.modules.pop('transformers', None)
            with patch.object(torch, '__version__', 'unsupported-test-runtime'), \
                 patch('looking_glass.bert_agreement_study.audit_event_journal') as read:
                with self.assertRaisesRegex(RuntimeError, 'runtime or CUDA state differs'):
                    self.real_runtime_audit(None, None, None)
                read.assert_not_called()

    def test_failed_test_body_releases_its_owned_ledger(self):
        class FailedBody(EvaluationLedgerTest):
            def runTest(inner):
                inner.test_ledger = inner.acquire()
                raise AssertionError('injected test failure')

        case = FailedBody()
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(len(result.failures), 1)
        self.assertFalse(result.errors, result.errors)
        self.assertTrue(case.test_ledger._closed)
        self.assertTrue(case.test_ledger.lease._released)

    def terminal(self, directory, run_id, status, owner_state='reaped'):
        value = {'schema': 'bert-agreement-controller-result-v1', 'run_id': run_id,
                 'source_commit': 'a' * 40, 'status': status, 'owner_state': owner_state}
        if status == 'complete':
            reservation, reservation_sha = verify_reservation(directory, run_id, self.binding)
            link = reservation_binding(reservation, reservation_sha)
            write_record(directory / 'started.json', {'schema': 'bert-agreement-controller-start-v1',
                         'run_id': run_id, 'source_commit': 'a' * 40, 'split': 'evaluation', 'family': None,
                         'started_utc': 'test-time'})
            write_record(directory / 'preflight.json', {'schema': 'bert-agreement-preflight-v1', 'run_id': run_id,
                         'source_commit': 'a' * 40, 'source_sha256': self.binding['source_sha256'],
                         'manifest_sha256': self.binding['manifest_sha256'], 'split': 'evaluation', 'family': None,
                         'cpu_study': self.binding['cpu_study'], 'evaluation_ledger': link})
            write_record(directory / 'owner-started.json', {'schema': 'bert-agreement-owner-start-v1',
                         'run_id': run_id, 'pid': 123, 'source_commit': 'a' * 40,
                         'manifest_sha256': self.binding['manifest_sha256'], 'split': 'evaluation', 'family': None,
                         'cpu_study': self.binding['cpu_study'], 'evaluation_ledger': link,
                         'cpu_owner_session_run_id': 'e' * 32})
            (directory / 'events.jsonl').write_bytes(b'{}\n')
            journal_sha = hashlib.sha256((directory / 'events.jsonl').read_bytes()).hexdigest()
            audited = {'sha256': journal_sha, 'families': [], 'summary': {}}
            runtime = {'python': platform.python_version(), 'platform': 'Windows', 'machine': 'unknown',
                       'torch': '2.8.0+cu128', 'transformers': '4.57.3', 'numpy': '2.4.6',
                       'intraop_threads': 1, 'interop_threads': 1, 'deterministic_algorithms': True,
                       'denormals_preserved_calling_thread': True, 'pytorch_cuda_initialized': False,
                       'python_hash_seed': 'isolated interpreter; OS-randomized', 'seed': 2026102810}
            model = dict(self.binding['model'], model_class='DistilBertForMaskedLM',
                         tokenizer_class='DistilBertTokenizerFast', torch_version='2.8.0+cu128',
                         transformers_version='4.57.3', dtype='float32', device='cpu',
                         attention_implementation='eager', parameter_count=66985530, pytorch_cuda_initialized=False)
            candidate = {'schema': 'bert-agreement-owner-v1', 'status': 'complete', 'run_id': run_id,
                         'pid': 123, 'source_commit': 'a' * 40, 'source_sha256': self.binding['source_sha256'],
                         'manifest_sha256': self.binding['manifest_sha256'], 'split': 'evaluation', 'family': None,
                         'journal_sha256': journal_sha, 'cpu_study': self.binding['cpu_study'],
                         'evaluation_ledger': link, 'runtime': runtime, 'model': model,
                         'cpu_owner_session_run_id': 'e' * 32,
                         'model_state_sha256_before_after': 'a' * 64, 'audited': audited}
            write_record(directory / 'owner-candidate.json', candidate)
            names = ('started.json', 'preflight.json', 'owner-started.json', 'owner-candidate.json', 'events.jsonl')
            names += ('reservation.json',)
            artifacts = {}
            for name in names:
                data = (directory / name).read_bytes()
                artifacts[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            value.update({'split': 'evaluation', 'family': None, 'source_sha256': self.binding['source_sha256'],
                          'manifest_sha256': self.binding['manifest_sha256'], 'cpu_study': self.binding['cpu_study'],
                          'owner_pid': 123, 'owner_exit_code': 0, 'model': model,
                          'runtime': runtime, 'model_state_sha256_before_after': 'a' * 64,
                          'evaluation_ledger': link, 'families': [], 'summary': {},
                          'evaluation_interpretation': {
                              'designation': 'replay' if link['prior_complete'] else 'first-complete-primary',
                              'prior_attempt_may_have_exposed_rows': link['prior_may_have_exposed'],
                              'unknown_history_and_pretraining': True},
                          'controller_audit_runtime': {'python': platform.python_version(), 'torch': '2.8.0+cu128',
                              'cuda_initialized_before_after': False,
                              'denormals_preserved_calling_thread': True, 'model_forwarded': False},
                          'cpu_supervision': {'scope': 'cooperative-cpu-study-custody-only',
                                              'clean': True, 'timed_out': False, 'pid': 123, 'exit_code': 0,
                                              'run_id': 'e' * 32},
                          'journal_sha256': artifacts['events.jsonl']['sha256'], 'artifacts': artifacts})
        write_record(directory / 'result.json', value)

    @requires_clean_model_dependencies
    def test_reservation_blocks_retry_until_separate_exit_and_preserves_order(self):
        first, second = self.run_id(1), self.run_id(2)
        ledger = self.acquire()
        directory, digest, prior_complete = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.assertFalse(prior_complete)
        record, verified = verify_reservation(directory, first, self.binding)
        self.assertEqual(verified, digest)
        self.assertEqual(record['prior_attempts'], [])
        self.assertFalse(record['prior_may_have_exposed'])
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        self.terminal(directory, first, 'complete')
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        ledger = self.acquire()
        chain, complete = ledger.prior_attempts()
        self.assertEqual([item['run_id'] for item in chain], [first])
        self.assertTrue(complete)
        next_dir, _, prior_complete = ledger.reserve(second, controller_supervision_run_id='f'*32)
        self.assertTrue(prior_complete)
        record, _ = verify_reservation(next_dir, second, self.binding)
        self.assertEqual(record['prior_attempts'], chain)
        self.assertTrue(record['prior_may_have_exposed'])
        self.terminal(next_dir, second, 'incomplete')
        ledger.release()
        record_controller_exit(self.root, self.identity, second, self.binding, 1)
        ledger = self.acquire()
        chain, complete = ledger.prior_attempts()
        self.assertEqual([item['run_id'] for item in chain], [first, second])
        self.assertEqual([item['status'] for item in chain], ['complete', 'incomplete'])
        self.assertTrue(complete)
        ledger.release()

    @requires_clean_model_dependencies
    def test_completed_prior_audit_can_enter_the_real_replay_controller(self):
        """A read-only prior audit may import Torch after the outer claim."""
        from looking_glass import bert_agreement_study as study
        first, second = self.run_id(27), self.run_id(28)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        with tempfile.TemporaryDirectory(prefix='bert-replay-roots-', dir=self.root.parent) as other:
            other = Path(other).resolve()
            repo, cache, owner_root, controller_root = (other / name for name in
                                                          ('repo', 'cache', 'owner', 'controller'))
            for path in (repo, cache, owner_root, controller_root):
                path.mkdir()
            manifest = build_manifest()
            def audit(_manifest, _path, sha):
                import torch
                self.assertFalse(torch.cuda.is_initialized())
                self.assertTrue(torch.set_flush_denormal(False))
                return ({'sha256': sha, 'families': [], 'summary': {}},
                        {'python': platform.python_version(), 'torch': '2.8.0+cu128',
                         'cuda_initialized_before_after': False,
                         'denormals_preserved_calling_thread': True, 'model_forwarded': False})
            attestations = []
            session = SimpleNamespace(run_id='f' * 32, check=lambda lease: None,
                                      attest_clean=lambda lease: attestations.append(lease))
            lease = object()
            with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout', return_value=repo), \
                 patch.object(study, 'EVALUATION_ENABLED', True), \
                 patch.object(study, '_checked_manifest', return_value=(self.binding['source_sha256'], manifest,
                                                                         self.binding['manifest_sha256'])), \
                 patch.object(study, '_evaluation_binding', return_value=self.binding), \
                 patch.object(study, 'audit_cpu_evaluation_journal', side_effect=audit), \
                 patch('looking_glass.cpu_study_supervision.claim_cpu_study', return_value=(session, lease)), \
                 patch('shutil.which', return_value='git'), \
                 patch.dict(os.environ, {'LOOKING_GLASS_BERT_SITE': ''}):
                result = study.run_cpu_controller(repo=repo, cache=cache, output_dir=self.root / second,
                         commit=self.binding['source_commit'], split='evaluation', custody_root=owner_root,
                         ledger_root=self.root, controller_custody_root=controller_root)
            self.assertEqual(result['status'], 'incomplete')
            # Recorded audit metadata does not authorize this host's runtime.
            # The real controller must reject unsupported Torch at startup.
            expected_stage = ('preflight' if str(sys.modules['torch'].__version__) == '2.8.0+cu128'
                              else 'startup')
            self.assertEqual(result['stage'], expected_stage)
            self.assertEqual(result['owner_state'], 'not_started')
            self.assertEqual(attestations, [lease])
            self.assertTrue((self.root / second / 'result.json').is_file())
            reservation, _ = verify_reservation(self.root / second, second, self.binding)
            self.assertTrue(reservation['prior_complete'])
            self.assertEqual(len(reservation['prior_audits']), 1)
            self.assertNotIn('transformers', sys.modules)
            self.assertFalse(sys.modules['torch'].cuda.is_initialized())

    def test_nonzero_exit_after_complete_publication_is_recorded_as_incomplete(self):
        first = self.run_id(7)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        write_record(directory / 'controller-failure.json', {'run_id': first, 'status': 'incomplete'})
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 1)
        ledger = self.acquire()
        chain, completed = ledger.prior_attempts()
        self.assertFalse(completed)
        self.assertEqual(chain[0]['status'], 'incomplete')
        self.assertIsNotNone(chain[0]['failure_sha256'])
        _, _, prior_complete = ledger.reserve(self.run_id(8), controller_supervision_run_id='f'*32)
        self.assertFalse(prior_complete)
        ledger.release()

    @requires_clean_model_dependencies
    def test_first_complete_after_partial_is_labeled_as_a_repeat_exposure(self):
        first, second = self.run_id(23), self.run_id(24)
        ledger = self.acquire()
        first_dir, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(first_dir, first, 'incomplete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 1)
        ledger = self.acquire()
        second_dir, _, prior_complete = ledger.reserve(second, controller_supervision_run_id='f'*32)
        self.assertFalse(prior_complete)
        self.terminal(second_dir, second, 'complete')
        interpretation = read_record(second_dir / 'result.json')[0]['evaluation_interpretation']
        self.assertEqual(interpretation['designation'], 'first-complete-primary')
        self.assertTrue(interpretation['prior_attempt_may_have_exposed_rows'])
        ledger.release()
        record_controller_exit(self.root, self.identity, second, self.binding, 0)
        ledger = self.acquire()
        _, _, prior_complete = ledger.reserve(self.run_id(25), controller_supervision_run_id='f'*32)
        self.assertTrue(prior_complete)
        ledger.release()

    def test_success_exit_with_failure_sidecar_is_not_accepted(self):
        first = self.run_id(9)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        write_record(directory / 'controller-failure.json', {'run_id': first, 'status': 'incomplete'})
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_exit_observation_is_preserved_even_when_terminal_is_missing(self):
        first = self.run_id(10)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 2)
        self.assertEqual(read_record(directory / 'controller-exit.json')[0]['exit_code'], 2)
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_other_threads_cannot_reserve_using_the_same_session(self):
        ledger = self.acquire()
        outcomes = []
        def attempt(number):
            try:
                ledger.reserve(self.run_id(number), controller_supervision_run_id='f'*32)
            except RuntimeError:
                outcomes.append('rejected')
        threads = [threading.Thread(target=attempt, args=(number,)) for number in (11, 12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(outcomes, ['rejected', 'rejected'])
        self.assertEqual(ledger.prior_attempts(), ([], False))
        ledger.release()

    def test_completed_artifact_tampering_cannot_be_reported_as_complete(self):
        first = self.run_id(13)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        (directory / 'events.jsonl').write_bytes(b'{"changed":true}\n')
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_cross_run_owner_artifact_is_rejected_even_with_matching_file_hash(self):
        first = self.run_id(17)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        owner_path = directory / 'owner-started.json'
        owner = read_record(owner_path)[0]
        owner['run_id'] = self.run_id(18)
        raw = json.dumps(owner, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
        owner_path.write_bytes(raw)
        terminal_path = directory / 'result.json'
        terminal = read_record(terminal_path)[0]
        terminal['artifacts']['owner-started.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        terminal_path.write_bytes(json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n')
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_matching_tampered_runtime_is_still_rejected(self):
        first = self.run_id(22)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        candidate_path = directory / 'owner-candidate.json'
        candidate = read_record(candidate_path)[0]
        candidate['runtime']['intraop_threads'] = 2
        raw = json.dumps(candidate, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
        candidate_path.write_bytes(raw)
        terminal_path = directory / 'result.json'
        terminal = read_record(terminal_path)[0]
        terminal['runtime']['intraop_threads'] = 2
        terminal['artifacts']['owner-candidate.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        terminal_path.write_bytes(json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n')
        ledger = self.acquire()
        with self.assertRaises(RuntimeError):
            ledger.prior_attempts()
        ledger.release()

    def test_rehashed_owner_evidence_cannot_change_its_supervisor_session(self):
        first = self.run_id(30)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        terminal_path = directory / 'result.json'
        terminal = read_record(terminal_path)[0]
        for name in ('owner-started.json', 'owner-candidate.json'):
            path = directory / name
            value = read_record(path)[0]
            value['cpu_owner_session_run_id'] = 'd' * 32
            raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
            path.write_bytes(raw)
            terminal['artifacts'][name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        terminal_path.write_bytes(json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n')
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    @unittest.skipUnless(importlib.metadata.version('torch') == '2.8.0+cu128',
                         'real historical journal audit requires torch 2.8.0+cu128')
    @requires_clean_model_dependencies
    def test_malformed_journal_is_rejected_by_the_real_closed_journal_auditor(self):
        import torch
        if str(torch.__version__) != '2.8.0+cu128':
            self.skipTest('real historical journal audit requires torch 2.8.0+cu128; host is ' + str(torch.__version__))
        first = self.run_id(19)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        ledger = self.acquire()
        with patch('looking_glass.bert_agreement_study.audit_event_journal', self.real_audit), \
             patch('looking_glass.bert_agreement_study.audit_cpu_evaluation_journal', self.real_runtime_audit):
            with self.assertRaises(ValueError):
                ledger.prior_attempts()
        ledger.release()

    @requires_clean_model_dependencies
    def test_malformed_journal_rejected_with_synthetic_runtime_record(self):
        """Exercise the real closed-journal parser independently of runtime admission."""
        first = self.run_id(19)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        ledger = self.acquire()
        with patch('looking_glass.bert_agreement_study.audit_event_journal', self.real_audit):
            with self.assertRaises(ValueError):
                ledger.prior_attempts()
        ledger.release()


    @requires_clean_model_dependencies
    def test_replay_prior_audit_does_not_call_a_model_forward(self):
        import torch
        first = self.run_id(20)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'complete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 0)
        ledger = self.acquire()
        with patch.object(torch.nn.Module, 'forward', side_effect=AssertionError('model forward is forbidden')):
            chain, completed = ledger.prior_attempts()
            self.assertTrue(completed)
            self.assertEqual(chain[0]['run_id'], first)
            self.assertTrue(self.audit_mock.called)
            second_dir, _, _ = ledger.reserve(self.run_id(21), controller_supervision_run_id='f'*32)
            reservation, _ = verify_reservation(second_dir, self.run_id(21), self.binding)
            self.assertFalse(reservation['prior_audits'][0]['model_forwarded'])
            self.assertFalse(reservation['prior_audits'][0]['cuda_initialized_before_after'])
        ledger.release()

    def test_later_reservation_commits_prior_terminal_and_exit_digests(self):
        first, second = self.run_id(14), self.run_id(15)
        ledger = self.acquire()
        first_dir, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(first_dir, first, 'incomplete')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 1)
        ledger = self.acquire()
        second_dir, _, _ = ledger.reserve(second, controller_supervision_run_id='f'*32)
        self.terminal(second_dir, second, 'incomplete')
        ledger.release()
        record_controller_exit(self.root, self.identity, second, self.binding, 1)
        exit_path = first_dir / 'controller-exit.json'
        original = read_record(exit_path)[0]
        original['exit_code'] = 2
        exit_path.write_bytes(json.dumps(original, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n')
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_interrupted_unlock_closes_old_session_before_competing_owner(self):
        code = r'''
import os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from looking_glass.bert_evaluation_ledger import EvaluationLedger, root_identity
from looking_glass import device_lease
root, digest = root_identity(Path(sys.argv[1]).resolve())
rows = [{'row_id':'evaluation-'+str(i), 'input_ids':[101,i]+[1]*8, 'attention_mask':[1]*10} for i in range(64)]
binding = {'source_commit':'a'*40, 'source_sha256':{'x':'b'*64}, 'protocol_sha256':'c'*64,
           'manifest_sha256':'d'*64, 'exposure_sha256':'e'*64, 'ledger_root_sha256':digest,
           'model':{}, 'cpu_study':{}, 'controller_cpu_study':{}, 'rows':rows}
ledger = EvaluationLedger.acquire(root, digest, binding)
real = device_lease._unlock
def interrupted(fd):
    real(fd)
    raise KeyboardInterrupt('after unlock')
try:
    with patch.object(device_lease, '_unlock', side_effect=interrupted):
        ledger.release()
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('unlock interruption was not observed')
try:
    ledger.reserve('1'*32, controller_supervision_run_id='f'*32)
except RuntimeError:
    pass
else:
    raise AssertionError('stale session remained usable')
peer = "from looking_glass.device_lease import DeviceLease; import sys; x=DeviceLease.acquire(sys.argv[1],timeout=2); x.release(); print('acquired')"
child = subprocess.run([sys.executable,'-c',peer,str(root/'bert-evaluation.lock')],capture_output=True,text=True,timeout=10,check=True)
assert child.stdout.strip() == 'acquired'
print('pass')
'''
        case = self.root / 'isolated'
        case.mkdir()
        result = subprocess.run([sys.executable, '-c', code, str(case)], capture_output=True,
                                text=True, timeout=30, check=True)
        self.assertEqual(result.stdout.strip(), 'pass')

    def test_incomplete_attempt_is_not_fresh_and_uncertain_owner_blocks_retry(self):
        first = self.run_id(3)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        self.terminal(directory, first, 'incomplete', owner_state='possibly_active')
        ledger.release()
        record_controller_exit(self.root, self.identity, first, self.binding, 1)
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_interruption_after_reservation_link_cannot_silently_retry(self):
        import os
        ledger = self.acquire()
        actual_link = os.link
        def interrupted(source, destination):
            actual_link(source, destination)
            raise KeyboardInterrupt('after the atomic link')
        with patch('looking_glass.bert_evaluation_ledger.os.link', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                ledger.reserve(self.run_id(4), controller_supervision_run_id='f'*32)
        ledger.release()
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()

    def test_changed_binding_wrong_root_and_boolean_exit_are_rejected(self):
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(self.run_id(5), controller_supervision_run_id='f'*32)
        changed = json.loads(json.dumps(self.binding))
        changed['rows'][0]['input_ids'][1] += 1
        with self.assertRaises(ValueError):
            verify_reservation(directory, self.run_id(5), changed)
        ledger.release()
        with self.assertRaises(ValueError):
            EvaluationLedger.acquire(self.root, 'f' * 64, self.binding)
        with self.assertRaises(ValueError):
            record_controller_exit(self.root, self.identity, self.run_id(5), self.binding, True)

    def test_controller_observation_must_match_the_reserved_supervisor_and_pid(self):
        run_id = self.run_id(29)
        ledger = self.acquire()
        _, _, _ = ledger.reserve(run_id, controller_supervision_run_id='f'*32)
        ledger.release()
        base = {'scope': 'cooperative-cpu-controller-custody-only', 'run_id': 'f' * 32,
                'pid': os.getpid(), 'exit_code': 1, 'clean': True, 'timed_out': False, 'reason': 'test'}
        for changed in ({'run_id': 'e' * 32}, {'pid': os.getpid() + 1}, {'clean': False}, {'timed_out': True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _record_controller_exit(self.root, self.identity, run_id, self.binding, dict(base, **changed))
        self.assertFalse((self.root / run_id / 'controller-exit.json').exists())
        _record_controller_exit(self.root, self.identity, run_id, self.binding, base)

    def test_binding_is_owned_and_strictly_typed(self):
        ledger = self.acquire()
        altered = ledger.binding
        altered['rows'][0]['input_ids'][1] = 999
        directory, _, _ = ledger.reserve(self.run_id(6), controller_supervision_run_id='f'*32)
        verify_reservation(directory, self.run_id(6), self.binding)
        ledger.release()
        bad = json.loads(json.dumps(self.binding))
        bad['rows'][0]['attention_mask'][0] = True
        with self.assertRaises(ValueError):
            self.acquire(bad)

    def test_cross_process_ledger_lock_excludes_another_controller(self):
        ledger = self.acquire()
        code = '''
import sys
from looking_glass.device_lease import DeviceLease, LeaseTimeoutError
try:
    lease = DeviceLease.acquire(sys.argv[1], timeout=0)
except LeaseTimeoutError:
    print('busy')
else:
    lease.release()
    print('acquired')
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.root / 'bert-evaluation.lock')],
                                capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(result.stdout.strip(), 'busy')
        ledger.release()

    def test_direct_child_checks_reserving_parent_and_held_lock(self):
        first = self.run_id(16)
        ledger = self.acquire()
        directory, _, _ = ledger.reserve(first, controller_supervision_run_id='f'*32)
        code = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from looking_glass.bert_evaluation_ledger import verify_owner_claim
try:
    verify_owner_claim(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5], json.loads(sys.argv[6]),
                       supervisor_pid=int(sys.argv[7]), supervisor_birth=sys.argv[8])
except ValueError:
    print('rejected')
else:
    print('accepted')
'''
        repo = Path(__file__).resolve().parents[1]
        reservation, _ = read_record(directory / 'reservation.json')
        args = [sys._base_executable, '-c', code, str(repo), str(self.root), self.identity,
                str(directory), first, json.dumps(self.binding), str(os.getpid()), reservation['controller_birth']]
        child = subprocess.run(args, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(child.stdout.strip(), 'accepted')
        ledger.release()
        child = subprocess.run(args, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(child.stdout.strip(), 'rejected')
        # Lock contention does not identify its owner. A separate process can
        # hold the lock, so the trusted controller must retain its own session.
        holder_code = r'''
import sys
sys.path.insert(0, sys.argv[1])
from looking_glass.device_lease import DeviceLease
lease = DeviceLease.acquire(sys.argv[2], timeout=5)
print('ready', flush=True)
sys.stdin.readline()
lease.release()
'''
        holder = subprocess.Popen([sys._base_executable, '-c', holder_code, str(repo),
                                   str(self.root / 'bert-evaluation.lock')], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), 'ready')
            child = subprocess.run(args, capture_output=True, text=True, timeout=20, check=True)
            self.assertEqual(child.stdout.strip(), 'accepted')
        finally:
            holder.communicate(input='\n', timeout=20)

    def test_duplicate_keys_and_unknown_root_entries_fail_closed(self):
        file = self.root / 'invalid.json'
        file.write_bytes(b'{"a":1,"a":2}\n')
        with self.assertRaises(ValueError):
            read_record(file)
        ledger = self.acquire()
        with self.assertRaises(ValueError):
            ledger.prior_attempts()
        ledger.release()


if __name__ == '__main__':
    unittest.main()
