"""Synthetic storage-limit and incomplete-attempt checks; no model required."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from looking_glass.agreement_journal import AgreementJournal, JournalError


class AgreementJournalTests(unittest.TestCase):
    def test_reused_thread_ident_or_dummy_object_cannot_inherit_owner(self):
        from looking_glass import agreement_journal as module
        for reuse_dummy_object in (False, True):
            with self.subTest(reuse_dummy_object=reuse_dummy_object), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'attempt'
                shared_dummy = object()
                current = (lambda: shared_dummy) if reuse_dummy_object else threading.current_thread
                # Simulate the same recycled numeric ID in two sequential thread
                # states. In the second case even current_thread() returns the
                # same stale dummy object, as can occur for alien threads.
                proxy = SimpleNamespace(get_ident=lambda: 47, current_thread=current, local=threading.local)
                created, attempts = [], []

                def create():
                    try:
                        journal = AgreementJournal(path, {'protocol': 'synthetic'})
                        reference = journal.record({'kind': 'owner'}, {'x': ((1,), b'\0' * 4)})['x']
                        assert journal.read_tensor(reference) == b'\0' * 4
                        created.append((journal, reference))
                    except BaseException as error:
                        created.append(error)

                def try_reused_identity():
                    journal, reference = created[0]
                    for operation in (lambda: journal.record({'kind': 'intruder'}, {}),
                                      lambda: journal.read_tensor(reference),
                                      lambda: journal.finish('complete', {})):
                        try:
                            operation()
                        except BaseException as error:
                            attempts.append(type(error))
                        else:
                            attempts.append(None)

                with mock.patch.object(module, 'threading', proxy):
                    first = threading.Thread(target=create)
                    first.start()
                    first.join(3)
                    self.assertFalse(first.is_alive())
                    self.assertEqual(len(created), 1)
                    self.assertIsInstance(created[0], tuple)
                    before = (path / 'events.jsonl').read_bytes()
                    second = threading.Thread(target=try_reused_identity)
                    second.start()
                    second.join(3)
                    self.assertFalse(second.is_alive())
                self.assertEqual(attempts, [JournalError, JournalError, JournalError])
                self.assertEqual((path / 'events.jsonl').read_bytes(), before)
                self.assertFalse((path / 'terminal.json').exists())

    def test_unique_attempt_deduplicates_only_identical_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            payload = b'\x00\x00\x80\x3f' * 3
            journal = AgreementJournal(path, {'protocol': 'synthetic'}, max_total_bytes=24,
                                       max_forward_bytes=24)
            refs = journal.record({'kind': 'capture'}, {'first': ((3,), payload),
                                                         'second': ((3,), payload)})
            self.assertEqual(journal.retained_bytes, len(payload))
            self.assertEqual(refs['first']['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(refs['first']['sha256'], refs['second']['sha256'])
            self.assertEqual((path / 'tensors' / (refs['first']['sha256'] + '.f32le')).read_bytes(), payload)
            self.assertEqual(journal.read_tensor(refs['first']), payload)
            journal.finish('complete', {'count': 1})
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'complete_candidate')
            with self.assertRaises(JournalError):
                journal.record({'kind': 'late'}, {})
            with self.assertRaises(FileExistsError):
                AgreementJournal(path, {'protocol': 'synthetic'})

    def test_per_forward_and_total_caps_preserve_failure_and_no_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'}, max_total_bytes=12,
                                       max_forward_bytes=8)
            with self.assertRaises(JournalError):
                journal.record({'kind': 'too-large'}, {'x': ((3,), b'\0' * 12)})
            self.assertEqual((path / 'events.jsonl').read_bytes(), b'')
            with self.assertRaises(JournalError):
                journal.record({'kind': 'after-failure'}, {})
            journal.finish('failed', {'reason': 'per-forward cap'})
            path = Path(temporary) / 'attempt-total'
            journal = AgreementJournal(path, {'protocol': 'synthetic'}, max_total_bytes=12,
                                       max_forward_bytes=8)
            journal.record({'kind': 'fits'}, {'x': ((2,), b'\0' * 8)})
            with self.assertRaises(JournalError):
                journal.record({'kind': 'total-overflow'}, {'y': ((2,), b'\1' * 8)})
            journal.finish('failed', {'reason': 'trace cap'})
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['retained_tensor_bytes'], 8)
            self.assertEqual(len((path / 'events.jsonl').read_text().splitlines()), 1)

    def test_invalid_shape_or_nonfinite_metadata_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            with self.assertRaises(JournalError):
                journal.record({'kind': 'bad-shape'}, {'x': ((2, 2), b'\0' * 4)})
            journal.finish('failed', {'reason': 'invalid shape'})
            path = Path(temporary) / 'attempt-metadata'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            with self.assertRaises(JournalError):
                journal.record({'kind': 'bad-metadata', 'value': float('nan')}, {})
            self.assertEqual((path / 'events.jsonl').read_bytes(), b'')
            journal.finish('failed', {'reason': 'invalid'})

    def test_event_log_byte_cap_does_not_publish_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'}, max_event_log_bytes=100)
            journal.record({'kind': 'first'}, {})
            with self.assertRaisesRegex(JournalError, 'event log'):
                journal.record({'kind': 'too-long', 'content': 'x' * 100}, {})
            journal.finish('failed', {'reason': 'event cap'})
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertEqual(terminal['event_log_bytes'], len((path / 'events.jsonl').read_bytes()))
            self.assertEqual(terminal['events'], 1)
            self.assertFalse(terminal['storage_accounting_complete'])

    def test_changed_tensor_file_is_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            ref = journal.record({'kind': 'capture'}, {'x': ((1,), b'\0' * 4)})['x']
            (path / 'tensors' / (ref['sha256'] + '.f32le')).write_bytes(b'\1' * 4)
            with self.assertRaises(JournalError):
                journal.read_tensor(ref)
            journal.finish('failed', {'reason': 'source changed'})

    def test_interrupted_tensor_write_cannot_be_reported_as_complete(self):
        from looking_glass import agreement_journal as module
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            original = module._exclusive_write
            def interrupted(target, data):
                original(target, data)
                raise OSError('after-write failure')
            with mock.patch.object(module, '_exclusive_write', side_effect=interrupted):
                with self.assertRaises(OSError):
                    journal.record({'kind': 'capture'}, {'x': ((1,), b'\0' * 4)})
            with self.assertRaises(JournalError):
                journal.finish('complete', {})
            journal.finish('failed', {'reason': 'write uncertainty'})
            terminal = json.loads((path / 'terminal.json').read_text())
            self.assertFalse(terminal['storage_accounting_complete'])

    def test_fsync_failure_during_terminal_does_not_publish_or_allow_more_events(self):
        from looking_glass import agreement_journal as module
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            with mock.patch.object(module.os, 'fsync', side_effect=OSError('flush failure')):
                with self.assertRaises(OSError):
                    journal.finish('complete', {})
            self.assertFalse((path / 'terminal.json').exists())
            with self.assertRaises(JournalError):
                journal.record({'kind': 'late'}, {})

    def test_interruption_after_publish_is_only_a_candidate(self):
        from looking_glass import agreement_journal as module
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'attempt'
            journal = AgreementJournal(path, {'protocol': 'synthetic'})
            real_link = module.os.link
            def linked_then_interrupted(source, target):
                real_link(source, target)
                raise KeyboardInterrupt('after publication')
            with mock.patch.object(module.os, 'link', side_effect=linked_then_interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    journal.finish('complete', {})
            self.assertEqual(json.loads((path / 'terminal.json').read_text())['status'], 'complete_candidate')
            with self.assertRaises(JournalError):
                journal.record({'kind': 'late'}, {})


if __name__ == '__main__':
    unittest.main()
