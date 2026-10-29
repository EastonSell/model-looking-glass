"""Evidence durability controls for the DistilBERT agreement study."""

import json
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from looking_glass.bert_agreement_study import (EventJournal, capture_model_state, _owner_main,
                                               aggregate_families, stage_verified_assets, verify_model_state,
                                               read_journal, run_cpu_controller, validate_owner_result, verify_git_sources,
                                               verify_tokenizer_rows, _checked_manifest, _require_checkout, _write_json,
                                               _reported_machine)
from looking_glass.pretrained import CheckpointIntegrityError


class EventJournalTest(unittest.TestCase):
    def test_records_are_durable_and_existing_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            with EventJournal(path, max_bytes=4096) as journal:
                journal.emit({'status': 'started', 'forward_index': 1})
                journal.emit({'status': 'completed', 'forward_index': 1})
                with self.assertRaises(RuntimeError):
                    _ = journal.sha256
            self.assertEqual(journal.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records, [{'status': 'started', 'forward_index': 1},
                                       {'status': 'completed', 'forward_index': 1}])
            with self.assertRaises(FileExistsError):
                EventJournal(path, max_bytes=4096)
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 2)

    def test_cap_failure_preserves_prior_records_and_latches_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            with EventJournal(path, max_bytes=128) as journal:
                journal.emit({'status': 'started', 'forward_index': 1})
                with self.assertRaises(ValueError):
                    journal.emit({'oversized': 'x' * 300})
                with self.assertRaises(RuntimeError):
                    journal.emit({'status': 'completed', 'forward_index': 1})
            self.assertEqual([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()],
                             [{'status': 'started', 'forward_index': 1}])

    def test_cross_thread_emission_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            journal = EventJournal(path, max_bytes=12)
            failures = []
            def emit_from_other_thread():
                try:
                    journal.emit({'x': 0})
                except Exception as error:
                    failures.append(type(error))
            threads = [threading.Thread(target=emit_from_other_thread) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [RuntimeError, RuntimeError])
            self.assertEqual(path.read_bytes(), b'')
            journal.close()

    def test_close_failure_cannot_publish_a_success_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            journal = EventJournal(path)
            stream = journal._stream
            class BrokenClose:
                def close(self):
                    stream.close()
                    raise OSError('injected close failure')
            journal._stream = BrokenClose()
            with self.assertRaises(OSError):
                journal.close()
            with self.assertRaises(RuntimeError):
                _ = journal.sha256

    def test_closed_journal_reader_rejects_partial_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            with EventJournal(path) as journal:
                journal.emit({'status': 'started', 'forward_index': 1})
            records, digest = read_journal(path)
            self.assertEqual(records, [{'status': 'started', 'forward_index': 1}])
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b'{"status":"started"')
            with self.assertRaises(ValueError):
                read_journal(path)
            path.write_bytes(b'{"status":"started","status":"completed"}\n')
            with self.assertRaises(ValueError):
                read_journal(path)

    def test_source_verifier_detects_assume_unchanged_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
            git('init')
            (root / 'source.py').write_bytes(b'value = 1\n')
            git('add', 'source.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
            commit = git('rev-parse', 'HEAD')
            self.assertEqual(verify_git_sources(root, commit, ('source.py',))['source.py'],
                             hashlib.sha256(b'value = 1\n').hexdigest())
            git('update-index', '--assume-unchanged', 'source.py')
            (root / 'source.py').write_bytes(b'value = 2\n')
            with self.assertRaises(RuntimeError):
                verify_git_sources(root, commit, ('source.py',))

    def test_owner_success_requires_matching_process_exit_and_evidence(self):
        candidate = {'schema': 'bert-agreement-owner-v1', 'status': 'complete',
                     'run_id': 'f' * 32, 'pid': 42, 'source_commit': 'a' * 40,
                     'journal_sha256': 'b' * 64}
        self.assertTrue(validate_owner_result(candidate, run_id='f' * 32, pid=42,
                                              exit_code=0, source_commit='a' * 40,
                                              journal_sha256='b' * 64))
        for kwargs in ({'exit_code': 1}, {'pid': 43}, {'journal_sha256': 'c' * 64}):
            arguments = {'run_id': 'f' * 32, 'pid': 42, 'exit_code': 0,
                         'source_commit': 'a' * 40, 'journal_sha256': 'b' * 64}
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                validate_owner_result(candidate, **arguments)

    def test_status_publication_never_exposes_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            with patch('os.fsync', side_effect=OSError('injected fsync failure')):
                with self.assertRaises(OSError):
                    _write_json(target, {'status': 'complete'})
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_status_publication_preserves_link_but_propagates_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            original_link = __import__('os').link
            def link_then_raise(source, dest):
                original_link(source, dest)
                raise KeyboardInterrupt('injected post-link interruption')
            with patch('os.link', side_effect=link_then_raise) as linking:
                with self.assertRaises(KeyboardInterrupt):
                    _write_json(target, {'status': 'complete'})
            linking.assert_called_once()
            self.assertEqual(json.loads(target.read_text()), {'status': 'complete'})

    def test_status_publication_preserves_primary_error_if_cleanup_also_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            with patch('os.link', side_effect=OSError('link primary')), \
                 patch('os.unlink', side_effect=ValueError('cleanup secondary')):
                with self.assertRaisesRegex(OSError, 'link primary') as raised:
                    _write_json(target, {'status': 'complete'})
            self.assertIsInstance(raised.exception.__cause__, ValueError)
            self.assertFalse(target.exists())

    def test_status_publication_prioritizes_cancellation_during_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            with patch('os.link', side_effect=OSError('link primary')), \
                 patch('os.unlink', side_effect=KeyboardInterrupt('cleanup cancelled')):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    _write_json(target, {'status': 'complete'})
            self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_witness_retains_original_parameter_storage_after_rebinding(self):
        import gc
        import weakref
        model = torch.nn.Linear(3, 2).eval()
        original = weakref.ref(model.weight.untyped_storage())
        witness = capture_model_state(model)
        with torch.no_grad():
            model.weight.data = model.weight.detach().clone()
        gc.collect()
        self.assertIsNotNone(original())
        with self.assertRaises(RuntimeError):
            verify_model_state(model, witness)

    def test_evaluation_freeze_gate_rejects_protocol_changed_after_git_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            (root / 'experiments').mkdir()
            protocol = b'Status: **frozen before evaluation**\n'
            (root / 'docs/BERT_AGREEMENT_V1.md').write_bytes(protocol)
            real_manifest = (Path(__file__).resolve().parents[1] /
                             'experiments/bert_agreement_public_cases_v1.json').read_bytes()
            (root / 'experiments/bert_agreement_public_cases_v1.json').write_bytes(real_manifest)
            exposure = (Path(__file__).resolve().parents[1] /
                        'experiments/bert_agreement_exposures_v2.json').read_bytes()
            (root / 'experiments/bert_agreement_exposures_v2.json').write_bytes(exposure)
            verified = {'docs/BERT_AGREEMENT_V1.md': hashlib.sha256(b'draft protocol').hexdigest(),
                        'experiments/bert_agreement_public_cases_v1.json': hashlib.sha256(real_manifest).hexdigest(),
                        'experiments/bert_agreement_exposures_v2.json': hashlib.sha256(exposure).hexdigest()}
            with patch('looking_glass.bert_agreement_study.verify_git_sources', return_value=verified):
                with self.assertRaises(RuntimeError):
                    _checked_manifest(root, 'a' * 40, 'git', 'evaluation')

    def test_evaluation_freeze_gate_does_not_accept_quoted_status_in_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            (root / 'experiments').mkdir()
            protocol = b'# Protocol\n\nStatus: **draft**\n\nDo not claim Status: **frozen before evaluation.** here.\n'
            (root / 'docs/BERT_AGREEMENT_V1.md').write_bytes(protocol)
            manifest = (Path(__file__).resolve().parents[1] /
                        'experiments/bert_agreement_public_cases_v1.json').read_bytes()
            (root / 'experiments/bert_agreement_public_cases_v1.json').write_bytes(manifest)
            exposure = (Path(__file__).resolve().parents[1] /
                        'experiments/bert_agreement_exposures_v2.json').read_bytes()
            (root / 'experiments/bert_agreement_exposures_v2.json').write_bytes(exposure)
            verified = {'docs/BERT_AGREEMENT_V1.md': hashlib.sha256(protocol).hexdigest(),
                        'experiments/bert_agreement_public_cases_v1.json': hashlib.sha256(manifest).hexdigest(),
                        'experiments/bert_agreement_exposures_v2.json': hashlib.sha256(exposure).hexdigest()}
            with patch('looking_glass.bert_agreement_study.verify_git_sources', return_value=verified):
                with self.assertRaises(RuntimeError):
                    _checked_manifest(root, 'a' * 40, 'git', 'evaluation')

    def test_preflight_binds_the_exact_exposure_snapshot_and_rejects_changes(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs').mkdir()
            (root / 'experiments').mkdir()
            protocol = b'# Draft\n\nStatus: **draft**\n'
            manifest = (source / 'experiments/bert_agreement_public_cases_v1.json').read_bytes()
            exposure = (source / 'experiments/bert_agreement_exposures_v2.json').read_bytes()
            files = {'docs/BERT_AGREEMENT_V1.md': protocol,
                     'experiments/bert_agreement_public_cases_v1.json': manifest,
                     'experiments/bert_agreement_exposures_v2.json': exposure}
            for name, data in files.items():
                (root / name).write_bytes(data)
            verified = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
            with patch('looking_glass.bert_agreement_study.verify_git_sources', return_value=verified):
                hashes, _, _ = _checked_manifest(root, 'a' * 40, 'git', 'development')
                self.assertEqual(hashes['experiments/bert_agreement_exposures_v2.json'],
                                 '224d683291b5b4f3a617697c8f9811351cf63bef19958fff9870c943e88fcdbc')
                # A changed file must fail even if its altered digest is returned
                # by a source verifier: only the pinned canonical snapshot works.
                changed = exposure + b' '
                (root / 'experiments/bert_agreement_exposures_v2.json').write_bytes(changed)
                with self.assertRaises(RuntimeError):
                    _checked_manifest(root, 'a' * 40, 'git', 'development')
                verified['experiments/bert_agreement_exposures_v2.json'] = hashlib.sha256(changed).hexdigest()
                with self.assertRaises(ValueError):
                    _checked_manifest(root, 'a' * 40, 'git', 'development')

    def test_journal_body_error_remains_primary_if_close_also_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = EventJournal(Path(directory) / 'events.jsonl')
            stream = journal._stream
            class BrokenClose:
                def close(self):
                    stream.close()
                    raise OSError('injected close failure')
            journal._stream = BrokenClose()
            with self.assertRaises(ValueError) as raised:
                with journal:
                    raise ValueError('primary body failure')
            self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_journal_close_cancellation_takes_precedence_over_body_error(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = EventJournal(Path(directory) / 'events.jsonl')
            stream = journal._stream
            class CancelledClose:
                def close(self):
                    stream.close()
                    raise KeyboardInterrupt('cancelled during close')
            journal._stream = CancelledClose()
            with self.assertRaises(KeyboardInterrupt) as raised:
                with journal:
                    raise ValueError('body failure')
            self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_controller_rejects_checkout_other_than_its_executed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout_a, checkout_b = root / 'a', root / 'b'
            (checkout_a / 'looking_glass').mkdir(parents=True)
            checkout_b.mkdir()
            source = checkout_a / 'looking_glass/bert_agreement_study.py'
            source.write_bytes(b'# fixture\n')
            with patch('looking_glass.bert_agreement_study.__file__', str(source)), \
                    patch.dict('os.environ', {'LOOKING_GLASS_BERT_ROOT': str(checkout_a)}):
                self.assertEqual(_require_checkout(checkout_a), checkout_a.resolve())
                with self.assertRaises(RuntimeError):
                    _require_checkout(checkout_b)

    def test_failed_preflight_writes_an_incomplete_attempt_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'attempt'
            code = '''
import json, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from looking_glass.bert_agreement_study import run_cpu_controller
root = Path(sys.argv[1])
os.environ['LOOKING_GLASS_BERT_SITE'] = str(root)
with patch('looking_glass.bert_agreement_study._require_bootstrap'), \
     patch('looking_glass.bert_agreement_study._require_checkout'), \
     patch('looking_glass.bert_agreement_study._checked_manifest', side_effect=RuntimeError('bad source')), \
     patch('subprocess.Popen') as spawn:
    result = run_cpu_controller(repo=root, cache=root, output_dir=root/'attempt',
                                commit='a'*40, split='development', family=1)
    spawn.assert_not_called()
print(json.dumps(result))
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)['status'], 'incomplete')
            self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'incomplete')

    def test_controller_records_owner_failure_and_unknown_reap_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import json, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from looking_glass.bert_agreement_study import run_cpu_controller, _write_json
root, mode = Path(sys.argv[1]), sys.argv[2]
os.environ['LOOKING_GLASS_BERT_SITE'] = str(root)
class Child:
    pid = 12345
    def __init__(self, command):
        if mode in ('failed', 'owner_cancelled'):
            run_id = command[command.index('--run-id') + 1]
            if mode == 'failed':
                _write_json(root/'attempt'/'owner-failure.json',
                    {'schema':'bert-agreement-owner-failure-v1', 'status':'incomplete',
                     'run_id':run_id, 'pid':self.pid, 'stage':'forwards', 'error_type':'MemoryError'})
            else:
                _write_json(root/'attempt'/'owner-cancelled.json',
                    {'schema':'bert-agreement-owner-cancel-v1', 'status':'incomplete',
                     'run_id':run_id, 'pid':self.pid, 'stage':'verification',
                     'error_type':'KeyboardInterrupt', 'previous_error_type':'OSError'})
    def wait(self, timeout):
        if mode in ('failed', 'owner_cancelled'): return 9
        if mode == 'cancel_during_kill': raise subprocess.TimeoutExpired('owner', timeout)
        raise KeyboardInterrupt('cancelled while waiting')
    def poll(self): return None
    def kill(self):
        if mode == 'cancel_during_kill': raise KeyboardInterrupt('cancelled while killing')
        raise OSError('injected kill failure')
with patch('looking_glass.bert_agreement_study._require_bootstrap'), \
     patch('looking_glass.bert_agreement_study._require_checkout'), \
     patch('looking_glass.bert_agreement_study._checked_manifest', return_value=({}, {}, 'b'*64)), \
     patch('shutil.which', return_value=sys.executable), \
     patch('subprocess.Popen', side_effect=lambda command, **kw: Child(command)):
    result = run_cpu_controller(repo=root, cache=root, output_dir=root/'attempt',
                                commit='a'*40, split='development', family=1)
print(json.dumps(result))
'''
            for mode in ('failed', 'owner_cancelled', 'cancelled', 'cancel_during_kill'):
                case_dir = root / mode
                case_dir.mkdir()
                completed = subprocess.run([sys.executable, '-c', code, str(case_dir), mode],
                                           capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout)
                self.assertEqual(result['status'], 'incomplete')
                if mode == 'failed':
                    self.assertEqual(result['owner_state'], 'reaped')
                    self.assertEqual(result['owner_exit_code'], 9)
                    self.assertEqual(result['owner_failure']['stage'], 'forwards')
                    self.assertEqual(result['owner_failure']['error_type'], 'MemoryError')
                elif mode == 'owner_cancelled':
                    self.assertEqual(result['owner_state'], 'reaped')
                    self.assertTrue(result['cancelled'])
                    self.assertEqual(result['owner_cancellation']['previous_error_type'], 'OSError')
                else:
                    self.assertEqual(result['owner_state'], 'possibly_active')
                    self.assertEqual(result['error_type'], 'KeyboardInterrupt')
                    self.assertTrue(result['cancelled'])
                    self.assertEqual(result['cleanup_error_type'], 'KeyboardInterrupt' if mode == 'cancel_during_kill' else 'OSError')
                    if mode == 'cancel_during_kill':
                        self.assertEqual(result['wait_error_type'], 'TimeoutExpired')

    def test_controller_start_failure_still_writes_incomplete_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import json, os, sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
root = Path(sys.argv[1])
original = study._write_json
def fail_start(path, value):
    if Path(path).name == 'started.json': raise OSError('injected start failure')
    return original(path, value)
with patch.object(study, '_require_bootstrap'), patch.object(study, '_write_json', side_effect=fail_start), \
     patch('subprocess.Popen') as spawn:
    result = study.run_cpu_controller(repo=root, cache=root, output_dir=root/'attempt',
                                      commit='a'*40, split='development', family=1)
    spawn.assert_not_called()
print(json.dumps(result))
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['error_type'], 'OSError')
            self.assertEqual(json.loads((root/'attempt'/'result.json').read_text())['status'], 'incomplete')

    def test_owner_cancellation_during_failure_publication_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _write_json
            def interrupt_failure(path, value):
                if Path(path).name == 'owner-failure.json':
                    raise KeyboardInterrupt('cancelled during failure publication')
                return original(path, value)
            with patch('looking_glass.bert_agreement_study._require_bootstrap', side_effect=OSError('startup')), \
                 patch('looking_glass.bert_agreement_study._write_json', side_effect=interrupt_failure):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    _owner_main(repo=root, cache=root, run_dir=root, commit='a'*40, git_exe='git',
                                split='development', family=1, run_id='f'*32)
            self.assertIsInstance(raised.exception.__cause__, OSError)
            cancellation = json.loads((root/'owner-cancelled.json').read_text())
            self.assertEqual(cancellation['error_type'], 'KeyboardInterrupt')
            self.assertEqual(cancellation['previous_error_type'], 'OSError')

    def test_owner_rejects_evaluation_before_loading_or_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
root = Path(sys.argv[1])
with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout'), \
     patch.object(study, '_checked_manifest') as checked:
    status = study._owner_main(repo=root, cache=root, run_dir=root, commit='a'*40, git_exe='git',
                               split='evaluation', family=None, run_id='f'*32)
    assert status == 1
    checked.assert_not_called()
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads((root/'owner-failure.json').read_text())['error_type'], 'RuntimeError')

    def test_actual_supervised_bootstrap_claims_before_model_import(self):
        import shutil
        import sysconfig
        from looking_glass.cpu_study_supervision import run_cpu_study
        repo = Path(__file__).resolve().parent.parent
        workspace = repo.parent.parent
        with tempfile.TemporaryDirectory(prefix='bert-cpu-custody-test-', dir=workspace) as directory:
            root = Path(directory).resolve(strict=True)
            self.assertTrue(root.is_relative_to(workspace.resolve(strict=True)))
            custody, attempt, prefix, cwd, direct_attempt, direct_prefix = (
                root/name for name in ('custody', 'attempt', 'prefix', 'cwd', 'direct-attempt', 'direct-prefix'))
            for path in (custody, attempt, prefix, cwd, direct_attempt, direct_prefix): path.mkdir()
            site = Path(sysconfig.get_paths()['purelib']).resolve()
            git = shutil.which('git')
            self.assertIsNotNone(git)
            command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix='+str(prefix),
                       str(repo/'scripts/bert_agreement_bootstrap.py'), '--bootstrap-site', str(site),
                       '--worker', '--repo', str(repo), '--cache', str(root), '--output', str(attempt),
                       '--commit', '0'*40, '--git', git, '--split', 'development', '--family', '1',
                       '--run-id', 'f'*32, '--custody-root', str(custody)]
            environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT')
                           if key in os.environ}
            environment.update({'TEMP': str(cwd), 'TMP': str(cwd),
                                'PATH': os.environ.get('PATH', ''), 'CUDA_VISIBLE_DEVICES': '-1'})
            direct_command = command.copy()
            direct_command[5] = 'pycache_prefix='+str(direct_prefix)
            direct_command[direct_command.index('--output') + 1] = str(direct_attempt)
            direct_environment = dict(environment, CUDA_VISIBLE_DEVICES='')
            direct = subprocess.run(direct_command, cwd=cwd, env=direct_environment,
                                    capture_output=True, timeout=30)
            self.assertEqual(direct.returncode, 1, direct.stderr)
            direct_failure = json.loads((direct_attempt/'owner-failure.json').read_text())
            self.assertEqual(direct_failure['stage'], 'custody_claim')
            self.assertFalse((direct_attempt/'events.jsonl').exists())
            result = run_cpu_study(command, study_root=custody, timeout=30, cwd=cwd,
                                   env=environment, inherit_environment=False)
            self.assertEqual(result.exit_code, 1)
            self.assertFalse(result.clean)
            self.assertTrue((custody/'cpu-study.unsafe.json').exists())
            failure = json.loads((attempt/'owner-failure.json').read_text())
            self.assertEqual(failure['stage'], 'source_preflight')
            self.assertEqual(failure['error_type'], 'RuntimeError')
            self.assertFalse((attempt/'events.jsonl').exists())

    def test_controller_rejects_evaluation_even_when_protocol_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
root = Path(sys.argv[1])
with patch.object(study, '_require_bootstrap'), patch('subprocess.Popen') as spawn:
    try:
        study.run_cpu_controller(repo=root, cache=root, output_dir=root/'attempt',
                                 commit='a'*40, split='evaluation')
    except RuntimeError as error:
        assert 'custody' in str(error)
    else: raise RuntimeError('evaluation was not blocked')
    spawn.assert_not_called()
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((root/'attempt').exists())

    def test_controller_cancellation_during_failure_publication_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
root = Path(sys.argv[1])
original = study._write_json
def fail_then_cancel(path, value):
    if Path(path).name == 'started.json': raise OSError('startup')
    if Path(path).name == 'controller-failure.json': raise KeyboardInterrupt('cancelled')
    return original(path, value)
with patch.object(study, '_require_bootstrap'), patch.object(study, '_write_json', side_effect=fail_then_cancel):
    try:
        study.run_cpu_controller(repo=root, cache=root, output_dir=root/'attempt',
                                 commit='a'*40, split='development', family=1)
    except KeyboardInterrupt as error:
        print(json.dumps({'error_type':type(error).__name__, 'cause':type(error.__cause__).__name__}))
    else: raise RuntimeError('cancellation was swallowed')
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), {'error_type': 'KeyboardInterrupt', 'cause': 'OSError'})
            cancellation = json.loads((root/'attempt'/'controller-cancelled.json').read_text())
            self.assertEqual(cancellation['error_type'], 'KeyboardInterrupt')
            self.assertEqual(cancellation['previous_error_type'], 'OSError')

    def test_supervised_timeout_is_recorded_as_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import json, os, sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
from looking_glass.supervisor import SupervisionResult
root = Path(sys.argv[1])
os.environ['LOOKING_GLASS_BERT_SITE'] = str(root)
(root/'repo').mkdir(); (root/'custody').mkdir()
observed = SupervisionResult('c'*32, 12345, 9, False, True, 'timeout')
with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout'), \
     patch.object(study, '_checked_manifest', return_value=({}, {}, 'b'*64)), \
     patch('shutil.which', return_value=sys.executable), \
     patch('looking_glass.cpu_study_supervision.run_cpu_study', return_value=observed) as supervisor:
    result = study.run_cpu_controller(repo=root/'repo', cache=root, output_dir=root/'attempt',
                                      commit='a'*40, split='development', family=1, custody_root=root/'custody')
    assert supervisor.call_count == 1
    call = supervisor.call_args
    assert call.kwargs['cwd'] == root/'attempt'/'owner-working'
    assert call.kwargs['env']['TEMP'] == str(root/'attempt'/'owner-working'/'tmp')
    assert '--custody-root' in call.args[0]
print(json.dumps(result))
'''
            completed = subprocess.run([sys.executable, '-c', code, str(root)],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result['status'], 'incomplete')
            self.assertTrue(result['timed_out'])
            self.assertFalse(result['cpu_supervision']['clean'])
            self.assertEqual(result['owner_state'], 'reaped')

    def test_controller_inventories_success_and_marks_post_link_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = r'''
import hashlib, json, os, sys
from pathlib import Path
from unittest.mock import patch
import looking_glass.bert_agreement_study as study
from looking_glass.supervisor import SupervisionResult
root, mode = Path(sys.argv[1]), sys.argv[2]
os.environ['LOOKING_GLASS_BERT_SITE'] = str(root)
supervised = mode.startswith('supervised_')
if supervised: (root/'repo').mkdir(); (root/'custody').mkdir()
original = study._write_json
event_raw = b'{"example":true}\n'
event_sha = hashlib.sha256(event_raw).hexdigest()
audited = {'sha256':event_sha, 'families':[], 'summary':{}}
def maybe_cancel(path, value):
    result = original(path, value)
    if mode == 'cancel' and Path(path).name == 'result.json' and value['status'] == 'complete':
        raise KeyboardInterrupt('injected after final link')
    return result
class Child:
    pid = 12345
    def __init__(self, command):
        run_id = command[command.index('--run-id') + 1]
        out = root/'attempt'
        started = {'schema':'bert-agreement-owner-start-v1',
            'run_id':run_id, 'pid':self.pid, 'source_commit':'a'*40,
            'manifest_sha256':'b'*64, 'split':'development', 'family':1}
        binding = study._cpu_study_binding(root/'custody') if supervised else None
        if supervised: started['cpu_study'] = binding
        original(out/'owner-started.json', started)
        (out/'events.jsonl').write_bytes(event_raw)
        runtime = {'pytorch_cuda_initialized':False, 'denormals_preserved_calling_thread':True,
            'deterministic_algorithms':True, 'intraop_threads':1, 'interop_threads':1,
            'torch':'2.8.0+cu128', 'transformers':'4.57.3', 'numpy':'2.4.6', 'seed':2026102810,
            'python':'3.11.15', 'platform':'Windows', 'machine':'' if mode == 'bad_machine' else 'AMD64',
            'python_hash_seed':'isolated interpreter; OS-randomized'}
        model = {'model':study.MODEL_ID, 'revision':study.REVISION,
            'attention_implementation':'eager', 'device':'cpu', 'dtype':'float32',
            'model_class':'DistilBertForMaskedLM', 'tokenizer_class':'DistilBertTokenizerFast',
            'torch_version':'2.8.0+cu128', 'transformers_version':'4.57.3',
            'parameter_count':66985530, 'pytorch_cuda_initialized':False,
            'assets':{name:{'size':size, 'sha256':digest} for name,(size,digest) in study.ASSETS.items()}}
        candidate = {'schema':'bert-agreement-owner-v1', 'status':'complete',
            'run_id':run_id, 'pid':self.pid, 'source_commit':'a'*40, 'source_sha256':{},
            'manifest_sha256':'b'*64, 'split':'development', 'family':1, 'journal_sha256':event_sha,
            'audited':audited, 'runtime':runtime, 'model':model,
            'model_state_sha256_before_after':'c'*64}
        if supervised:
            candidate['cpu_study'] = dict(binding)
            if mode == 'supervised_bad_binding': candidate['cpu_study']['custody_root_sha256'] = 'd'*64
        original(out/'owner-candidate.json', candidate)
    def wait(self, timeout): return 0
def fake_supervisor(command, **kwargs):
    Child(command)
    return SupervisionResult('c'*32, 12345, 0, True, False, 'clean device teardown')
with patch.object(study, '_require_bootstrap'), patch.object(study, '_require_checkout'), \
     patch.object(study, '_checked_manifest', return_value=({}, {}, 'b'*64)), \
     patch.object(study, 'audit_event_journal', return_value=audited), \
     patch.object(study, 'verify_git_sources', return_value={}), \
     patch.object(study, '_write_json', side_effect=maybe_cancel), \
     patch('shutil.which', return_value=sys.executable), \
     patch('looking_glass.cpu_study_supervision.run_cpu_study', side_effect=fake_supervisor), \
     patch('subprocess.Popen', side_effect=lambda command, **kw: Child(command)):
    result = study.run_cpu_controller(repo=root/'repo' if supervised else root, cache=root, output_dir=root/'attempt',
                                      commit='a'*40, split='development', family=1,
                                      custody_root=root/'custody' if supervised else None)
print(json.dumps(result))
'''
            for mode in ('success', 'cancel', 'bad_machine', 'supervised_success', 'supervised_bad_binding'):
                case = root / mode
                case.mkdir()
                completed = subprocess.run([sys.executable, '-c', code, str(case), mode],
                                           capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout)
                attempt = case / 'attempt'
                if mode in ('success', 'supervised_success'):
                    self.assertEqual(result['status'], 'complete')
                    self.assertFalse((attempt/'controller-failure.json').exists())
                    if mode == 'supervised_success':
                        self.assertTrue(result['cpu_supervision']['clean'])
                        self.assertEqual(result['cpu_supervision']['reason'], 'clean device teardown')
                        self.assertEqual(result['cpu_supervision']['scope'], 'cooperative-cpu-study-custody-only')
                        self.assertEqual(result['cpu_study']['contract'], 'cooperative-cpu-study-v1')
                        self.assertEqual(json.loads((attempt/'preflight.json').read_text())['cpu_study'],
                                         result['cpu_study'])
                        self.assertEqual(json.loads((attempt/'owner-started.json').read_text())['cpu_study'],
                                         result['cpu_study'])
                    self.assertEqual(set(result['artifacts']),
                                     {'started.json', 'preflight.json', 'owner-started.json',
                                      'owner-candidate.json', 'events.jsonl'})
                    for name, entry in result['artifacts'].items():
                        raw = (attempt/name).read_bytes()
                        self.assertEqual(entry, {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
                elif mode == 'cancel':
                    self.assertEqual(result['status'], 'incomplete')
                    self.assertTrue(result['cancelled'])
                    self.assertEqual(json.loads((attempt/'controller-failure.json').read_text())['status'], 'incomplete')
                    self.assertEqual(json.loads((attempt/'result.json').read_text())['status'], 'complete')
                elif mode == 'bad_machine':
                    self.assertEqual(result['status'], 'incomplete')
                    self.assertEqual(result['stage'], 'runtime_validation')
                else:
                    self.assertEqual(result['status'], 'incomplete')
                    self.assertEqual(result['stage'], 'evidence_validation')

    def test_model_value_change_or_same_value_parameter_replacement_is_detected(self):
        model = torch.nn.Linear(3, 2).eval()
        model.register_buffer('marker', torch.tensor([1., 2.]))
        witness = capture_model_state(model)
        self.assertTrue(verify_model_state(model, witness))
        with torch.no_grad():
            model.marker[0] = 3.
        with self.assertRaises(RuntimeError):
            verify_model_state(model, witness)
        with torch.no_grad():
            model.marker[0] = 1.
        self.assertTrue(verify_model_state(model, witness))
        model.weight = torch.nn.Parameter(model.weight.detach().clone())
        with self.assertRaises(RuntimeError):
            verify_model_state(model, witness)

    def test_asset_stage_copies_only_exact_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, stage = root / 'snapshot', root / 'stage'
            source.mkdir()
            stage.mkdir()
            payload = b'fixed checkpoint bytes'
            (source / 'model.safetensors').write_bytes(payload)
            assets = {'model.safetensors': (len(payload), hashlib.sha256(payload).hexdigest())}
            with patch('looking_glass.bert_agreement_study.ASSETS', assets):
                stage_verified_assets(source, stage)
            self.assertEqual((stage / 'model.safetensors').read_bytes(), payload)
            other = root / 'other'
            other.mkdir()
            (source / 'model.safetensors').write_bytes(b'corrupt checkpoint data')
            with patch('looking_glass.bert_agreement_study.ASSETS', assets):
                with self.assertRaises(CheckpointIntegrityError):
                    stage_verified_assets(source, other)

    def test_tokenizer_mismatch_rejects_manifest_before_model_work(self):
        class Tokenizer:
            mask_token_id = 103
            def convert_tokens_to_ids(self, token):
                return {'is': 2003, 'are': 2024}[token]
            def encode(self, token, **_options):
                return [self.convert_tokens_to_ids(token)]
            def __call__(self, text, **_options):
                return {'input_ids': [101, 42, 103, 102], 'attention_mask': [1, 1, 1, 1]}
        manifest = {'rows': [{'text': 'public fixture', 'input_ids': [101, 42, 103, 102],
                              'attention_mask': [1, 1, 1, 1]}]}
        self.assertEqual(verify_tokenizer_rows(Tokenizer(), manifest), 1)
        manifest['rows'][0]['input_ids'][1] = 43
        with self.assertRaises(ValueError):
            verify_tokenizer_rows(Tokenizer(), manifest)

    def test_unavailable_machine_identity_is_explicitly_unknown(self):
        class Platform:
            def machine(self): return ''
        self.assertEqual(_reported_machine(Platform()), 'unknown')

    def test_held_out_decisions_keep_all_eight_family_units(self):
        bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
        rows, targets, results = [], [], []
        sites = ('candidate', 'block2_distractor', 'random_block3_distractor',
                 'random_block4_distractor', 'random_block0_subject')
        for family in range(1, 9):
            cases = []
            controls = []
            for index, (number, distractor) in enumerate((('singular', 'singular'), ('singular', 'plural'),
                                                          ('plural', 'singular'), ('plural', 'plural'))):
                row_id, case_id = f'row-{family}-{index}', f'case-{family}-{index}'
                rows.append({'row_id': row_id, 'split': 'evaluation', 'family': family,
                             'subject_number': number, 'distractor_number': distractor})
                targets.append({'case_id': case_id, 'target_row_id': row_id,
                                'split': 'evaluation', 'family': family})
                effects = dict.fromkeys(sites, 0.0)
                effects['candidate'] = 0.2 if family <= 6 else -0.05
                effects['block2_distractor'] = 0.08
                cases.append({'case_id': case_id, 'target_row_id': row_id,
                              'baseline_is_bits': bits(1. if number == 'singular' else 0.),
                              'baseline_are_bits': bits(0. if number == 'singular' else 1.),
                              'effects': effects})
                controls.extend({'name': 'self_patch', 'case_id': case_id, 'site': site, 'passed': True}
                                for site in sites)
                controls.extend({'name': 'clean_matches_candidate', 'case_id': row_id,
                                 'site': site, 'passed': True} for site in sites[1:])
                controls.extend({'name': name, 'case_id': case_id, 'site': site, 'passed': True}
                                for name, site in (('final_subject', 'final_subject'),
                                                   ('final_mask', 'final_mask'), ('final_clean', 'candidate')))
            results.append({'split': 'evaluation', 'family': family, 'forward_count': 140,
                            'cases': cases, 'controls': controls})
        summary = aggregate_families({'rows': rows, 'targets': targets}, results)
        self.assertAlmostEqual(summary['site_means']['candidate'], .1375)
        self.assertEqual(summary['positive_candidate_families'], 6)
        self.assertTrue(summary['directional_hypothesis_passed'])
        self.assertFalse(summary['site_specificity_passed'])
        self.assertEqual(summary['baseline']['correct'], 32)
        self.assertTrue(all(cell['correct'] == 8 for cell in summary['baseline']['cells'].values()))
        with self.assertRaises(ValueError):
            aggregate_families({'rows': rows, 'targets': targets}, results[:-1])


if __name__ == '__main__':
    unittest.main()
