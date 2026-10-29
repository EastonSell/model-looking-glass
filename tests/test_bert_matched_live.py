"""No-forward checks of the matched owner binding and terminal boundary."""

import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from looking_glass import bert_agreement_study, bert_matched_cases
from looking_glass.bert_evaluation_ledger import read_record, write_record
from looking_glass.supervisor import SupervisionResult


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/bert_matched_cpu.py'
spec = importlib.util.spec_from_file_location('matched_cpu_test', SCRIPT)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class MatchedLiveTests(unittest.TestCase):
    def test_repeated_console_interrupt_is_deferred_during_custody_recovery(self):
        owner._ACTIVE[:] = [object(), object(), True]
        owner._INTERRUPTED[0] = False
        try:
            with self.assertRaises(KeyboardInterrupt):
                owner._console_interrupt(None, None)
            owner._console_interrupt(None, None)
            self.assertTrue(owner._INTERRUPTED[0])
            self.assertTrue(owner._ACTIVE[2])
        finally:
            owner._ACTIVE[:] = [None, None, False]
            owner._INTERRUPTED[0] = False

    def test_recovery_condition_and_resource_unpack_cannot_drop_custody(self):
        lines, start = inspect.getsourcelines(owner._recover_active)
        for needle in ('if not _ACTIVE[2]:', 'ledger, source_lease = _ACTIVE[0], _ACTIVE[1]'):
            with self.subTest(boundary=needle):
                target = start + next(i for i, line in enumerate(lines) if needle in line)
                flags = {'released': False, 'closed': False, 'interrupted': False}
                class Ledger:
                    def release(self): flags['released'] = True
                class Lease:
                    _kernel = object()
                    def close(self): flags['closed'] = True
                owner._ACTIVE[:] = [Ledger(), Lease(), True]
                def trace(frame, event, _arg):
                    if frame.f_code is owner._recover_active.__code__ and event == 'line' and frame.f_lineno == target and not flags['interrupted']:
                        flags['interrupted'] = True
                        raise KeyboardInterrupt()
                    return trace
                try:
                    with patch('looking_glass.supervisor.pending_supervised_children', return_value=()), \
                         patch.object(owner.time, 'sleep', return_value=None), \
                         patch.object(owner.signal, 'signal', return_value=owner.signal.SIG_DFL):
                        sys.settrace(trace)
                        try:
                            owner._recover_active()
                        finally:
                            sys.settrace(None)
                    self.assertTrue(flags['interrupted'])
                    self.assertTrue(flags['closed'])
                    self.assertTrue(flags['released'])
                    self.assertEqual(owner._ACTIVE, [None, None, False])
                finally:
                    sys.settrace(None)
                    owner._ACTIVE[:] = [None, None, False]

    def test_candidate_requires_complete_runtime_and_forward_count(self):
        args = types.SimpleNamespace(revision='a' * 40, rehearsal=False, checkpoint_preflight=False)
        observed = SupervisionResult('c' * 32, 123, 0, True, False, 'fixture')
        binding = {'manifest_sha256': 'b' * 64}
        common = {'schema': 'bert-matched-owner-candidate-v1', 'pid': 123,
                  'reservation_sha256': 'd' * 64, 'source_commit': args.revision,
                  'manifest_sha256': binding['manifest_sha256'], 'forward_count': 480}
        with self.assertRaises(RuntimeError):
            owner._validate_candidate(common, observed, 'd' * 64, args, binding)
        candidate = {**common, 'journal_sha256': 'e' * 64, 'model_state_sha256_before_after': 'f' * 64,
                     'model': {}, 'runtime': {'python': '3.11.15', 'cuda_visible_devices': ''}, 'audit': {}}
        with patch('looking_glass.bert_agreement_study.validate_cpu_runtime_and_model') as check:
            owner._validate_candidate(candidate, observed, 'd' * 64, args, binding)
            check.assert_called_once()
            for changed in ({'forward_count': 479}, {'forward_count': True},
                            {'runtime': {'python': '3.11.15', 'cuda_visible_devices': '0'}},
                            {'journal_sha256': 'bad'}):
                with self.assertRaises(RuntimeError):
                    owner._validate_candidate({**candidate, **changed}, observed, 'd' * 64, args, binding)
            args.checkpoint_preflight = True
            with self.assertRaises(RuntimeError):
                owner._validate_candidate(candidate, observed, 'd' * 64, args, binding)
            preflight = {**candidate, 'forward_count': 0, 'audit': None, 'journal_sha256': None}
            owner._validate_candidate(preflight, observed, 'd' * 64, args, binding)

    def test_pending_query_allocation_failure_keeps_custody_rooted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('ledger', 'custody'):
                (root / name).mkdir()
            args = types.SimpleNamespace(revision='a' * 40, cache=str(root), ledger=str(root / 'ledger'),
                custody=str(root / 'custody'), site=str(root), rehearsal=True, checkpoint_preflight=False,
                timeout=10)
            flags = {'released': False, 'closed': False}
            class Ledger:
                def reserve(self, attempt_id):
                    path = root / 'ledger' / attempt_id
                    path.mkdir()
                    write_record(path / 'reservation.json', {'schema': 'fake'})
                    return path
                def release(self): flags['released'] = True
            class Lease:
                def __init__(self, *_args): pass
                def __enter__(self): return self
                def __exit__(self, *_args): flags['closed'] = True
            class Guards:
                SourceFileLease = Lease
                def WindowsJob(self): return object()
            launcher = {'_load_guards': lambda *_: Guards(), '_verify_sources': lambda *_: {'x': 'y'}}
            try:
                with patch.object(owner, '_binding', return_value=({'manifest_sha256': 'b' * 64}, {}, None)), \
                     patch('looking_glass.bert_matched_owner.MatchedLedger.acquire', return_value=Ledger()), \
                     patch('looking_glass.cpu_study_supervision.run_cpu_study', side_effect=KeyboardInterrupt()), \
                     patch('looking_glass.supervisor.pending_supervised_children', side_effect=[(), MemoryError()]):
                    with self.assertRaises(KeyboardInterrupt):
                        owner._controller(args, launcher, {'x': 'y'}, {'x': 'y'},
                                          types.SimpleNamespace(verify_inventory=lambda: None))
                self.assertTrue(owner._ACTIVE[2])
                self.assertIsNotNone(owner._ACTIVE[0])
                self.assertIsNotNone(owner._ACTIVE[1])
                self.assertEqual(flags, {'released': False, 'closed': False})
            finally:
                owner._ACTIVE[:] = [None, None, False]

    def test_binding_uses_exact_ordered_manifest_and_verified_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'experiments').mkdir()
            path = root / owner.EXTRA[2]
            path.write_bytes(bert_matched_cases.canonical_manifest_bytes())
            snapshot = (root / 'cache/models--distilbert--distilbert-base-uncased/snapshots'
                        / bert_agreement_study.REVISION)
            snapshot.mkdir(parents=True)
            assets = {}
            for index, name in enumerate(bert_agreement_study.ASSETS):
                raw = bytes([index + 1]) * (index + 2)
                (snapshot / name).write_bytes(raw)
                assets[name] = (len(raw), hashlib.sha256(raw).hexdigest())
            hashes = {owner.EXTRA[1]: bert_matched_cases.PROTOCOL_SHA256,
                      owner.EXTRA[2]: bert_matched_cases.MANIFEST_SHA256}
            with patch.object(owner, 'ROOT', root), patch.object(bert_agreement_study, 'ASSETS', assets):
                binding, manifest, _ = owner._binding('a' * 40, hashes, root / 'cache')
                self.assertEqual(binding['rows'], [{key: row[key] for key in
                    ('row_id', 'input_ids', 'attention_mask')} for row in manifest['rows']])
                self.assertEqual(len(binding['rows']), 64)
                (snapshot / 'vocab.txt').write_bytes(b'wrong')
                with self.assertRaisesRegex(RuntimeError, 'checkpoint asset'):
                    owner._binding('a' * 40, hashes, root / 'cache')
                path.write_bytes(b'{}')
                with self.assertRaises(ValueError):
                    owner._binding('a' * 40, hashes, root / 'cache')

    def test_post_link_error_keeps_candidate_and_failure_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('ledger', 'custody'):
                (root / name).mkdir()
            args = types.SimpleNamespace(revision='a' * 40, cache=str(root), ledger=str(root / 'ledger'),
                custody=str(root / 'custody'), site=str(root), rehearsal=True, checkpoint_preflight=False,
                timeout=10)
            binding = {'manifest_sha256': 'b' * 64}
            class Ledger:
                def reserve(self, attempt_id):
                    path = root / 'ledger' / attempt_id
                    path.mkdir()
                    write_record(path / 'reservation.json', {'schema': 'fake'})
                    return path
                def record_incomplete(self, _observed):
                    pass
                def release(self):
                    pass
            class Lease:
                def __init__(self, *_args): pass
                def __enter__(self): return self
                def __exit__(self, *_args): pass
            class Guards:
                SourceFileLease = Lease
                def WindowsJob(self): return object()
            launcher = {'_load_guards': lambda *_: Guards(), '_verify_sources': lambda *_: {'x': 'y'}}
            def supervise(command, **_kwargs):
                attempt_id = command[command.index('--attempt') + 1]
                digest = command[command.index('--reservation-sha') + 1]
                write_record(root / 'ledger' / attempt_id / 'owner-rehearsal.json', {
                    'schema': 'bert-matched-owner-rehearsal-v1', 'pid': 123,
                    'reservation_sha256': digest, 'source_commit': args.revision,
                    'manifest_sha256': binding['manifest_sha256'], 'forward_count': 0,
                    'cuda_visible_devices': ''})
                return SupervisionResult('b' * 32, 123, 0, True, False, 'fixture')
            def interrupted_write(path, value):
                result = write_record(path, value)
                if Path(path).name == 'controller-result.json':
                    raise OSError('injected error after publication')
                return result
            with patch.object(owner, '_binding', return_value=(binding, {}, None)), \
                 patch('looking_glass.bert_matched_owner.MatchedLedger.acquire', return_value=Ledger()), \
                 patch('looking_glass.cpu_study_supervision.run_cpu_study', side_effect=supervise), \
                 patch('looking_glass.bert_evaluation_ledger.write_record', side_effect=interrupted_write):
                with self.assertRaisesRegex(OSError, 'after publication'):
                    owner._controller(args, launcher, {'x': 'y'}, {'x': 'y'},
                                      types.SimpleNamespace(verify_inventory=lambda: None))
            attempt = next((root / 'ledger').iterdir())
            self.assertEqual(read_record(attempt / 'controller-result.json')[0]['status'], 'rehearsal_candidate')
            self.assertEqual(read_record(attempt / 'controller-failure.json')[0]['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
