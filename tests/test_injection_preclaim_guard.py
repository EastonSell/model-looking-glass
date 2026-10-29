"""Synthetic SQLite claim tests; no production claim or private split."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager, closing
from dataclasses import replace
from unittest import mock

from looking_glass import injection_preclaim_guard as guard


def snapshot(meta, inventory):
    h = hashlib.sha256()
    for item in (meta.encode(), inventory.encode()):
        h.update(len(item).to_bytes(8, 'big'))
        h.update(item)
    return h.hexdigest()


def history():
    h = hashlib.sha256()
    h.update(b'reconciliations:present\n')
    h.update(b'retired_evidence:present\n')
    return h.hexdigest()


def claims_digest(rows):
    return hashlib.sha256(json.dumps(sorted(rows), ensure_ascii=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def row_digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode('ascii')).hexdigest()


@contextmanager
def writable(path):
    with closing(sqlite3.connect(path)) as db:
        with db:
            yield db


class PreclaimGuardTests(unittest.TestCase):
    def setUp(self):
        work = Path.cwd() / 'work'
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.db_path = self.root / 'ledger.sqlite3'
        self.phase = self.root / 'phase'
        self.orig = [('natural_discovery', str(self.root / 'natural'), 128 * 1024**2, 4096 * 1024**2, 0),
                     ('injection_discovery', str(self.root / 'injection'), 48 * 1024**2, 128 * 1024**2, 13 * 1024**3)]
        self.claim = ('injection_heldout', str(self.phase), 72 * 1024**2, 192 * 1024**2, 13 * 1024**3)
        self.scope = '1' * 64
        self.meta = json.dumps({'schema': 'mlg-agreement-phase-ledger-v6',
                                'current_history_sha256': history(),
                                'evidence_scope_amendment_sha256': self.scope},
                               sort_keys=True, separators=(',', ':'))
        self.inventory = '{"prior":{},"natural":{}}'
        with writable(self.db_path) as db:
            db.execute('CREATE TABLE meta (id INTEGER PRIMARY KEY, data TEXT NOT NULL, inventory TEXT NOT NULL)')
            db.execute('CREATE TABLE claims (phase TEXT PRIMARY KEY, output TEXT NOT NULL, logical INTEGER NOT NULL, disk INTEGER NOT NULL, free INTEGER NOT NULL)')
            db.execute('CREATE TABLE reconciliations (seq INTEGER, manifest_sha TEXT, gap_sha TEXT, claims_sha TEXT, previous_meta TEXT, previous_inventory TEXT)')
            db.execute('CREATE TABLE retired_evidence (side TEXT, path_sha TEXT, size INTEGER, content_sha TEXT)')
            db.execute('INSERT INTO meta VALUES(1, ?, ?)', (self.meta, self.inventory))
            db.executemany('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', self.orig + [self.claim])
        identity = self.db_path.stat()
        self.anchor = guard.ClaimAnchor(ledger_path=str(self.db_path), db_dev=identity.st_dev,
            db_ino=identity.st_ino, phase_path=str(self.phase), state_sha256=snapshot(self.meta, self.inventory),
            history_sha256=history(), scope_sha256=self.scope, original_claims_sha256=claims_digest(self.orig),
            heldout_claim_sha256=row_digest(self.claim))

    def verify(self, *, path=None, phase='injection_heldout', output=None, anchor=None):
        return guard.verify_existing_claim(path or self.db_path, phase, output_path=output or self.phase,
                                           anchor=anchor or self.anchor)

    def test_exact_claim_and_additional_natural_claim_are_accepted_read_only(self):
        self.assertTrue(self.verify())
        with writable(self.db_path) as db:
            db.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', ('natural_heldout',
                str(self.root / 'natural-heldout'), 160 * 1024**2, 224 * 1024**2, 13 * 1024**3))
        self.assertTrue(self.verify())

    def test_copied_or_hardlinked_database_cannot_reuse_anchor(self):
        copy = self.root / 'copy.sqlite3'
        shutil.copyfile(self.db_path, copy)
        with self.assertRaises(guard.PreclaimError):
            self.verify(path=copy)
        link = self.root / 'linked.sqlite3'
        os.link(self.db_path, link)
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_same_path_database_replacement_fails_original_file_identity(self):
        copy = self.root / 'replacement.sqlite3'
        shutil.copyfile(self.db_path, copy)
        os.replace(copy, self.db_path)
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_foreign_or_malformed_anchor_is_rejected(self):
        for anchor in (replace(self.anchor, db_ino=self.anchor.db_ino + 1),
                       replace(self.anchor, scope_sha256='z' * 64),
                       replace(self.anchor, phase_path=str(self.root / 'other'))):
            with self.subTest(anchor=anchor), self.assertRaises(guard.PreclaimError):
                self.verify(anchor=anchor)

    def test_wrong_output_phase_or_consumed_row_is_rejected(self):
        for args in ({'phase': 'natural_heldout'}, {'output': self.root / 'other'}):
            with self.subTest(args=args), self.assertRaises(guard.PreclaimError):
                self.verify(**args)
        with writable(self.db_path) as db:
            db.execute('UPDATE claims SET disk = disk - 1 WHERE phase = ?', ('injection_heldout',))
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_any_change_to_the_pinned_injection_row_is_rejected(self):
        with writable(self.db_path) as db:
            db.execute('UPDATE claims SET free=free+1 WHERE phase=?', ('injection_heldout',))
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_natural_heldout_row_must_be_disjoint_and_record_valid_free_space(self):
        with writable(self.db_path) as db:
            db.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', ('natural_heldout',
                str(self.phase), 160 * 1024**2, 224 * 1024**2, 13 * 1024**3))
        with self.assertRaises(guard.PreclaimError):
            self.verify()
        with writable(self.db_path) as db:
            db.execute('UPDATE claims SET output=?, free=0 WHERE phase=?',
                       (str(self.root / 'natural-heldout'), 'natural_heldout'))
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_overlong_claim_path_is_rejected_before_fetch(self):
        with writable(self.db_path) as db:
            db.execute('INSERT INTO claims VALUES(?, ?, ?, ?, ?)', ('natural_heldout',
                str(self.root) + ('x' * 40000), 160 * 1024**2, 224 * 1024**2, 13 * 1024**3))
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_changed_original_claim_or_reconciliation_state_is_rejected(self):
        with writable(self.db_path) as db:
            db.execute('UPDATE claims SET free = free + 1 WHERE phase = ?', ('injection_discovery',))
        with self.assertRaises(guard.PreclaimError):
            self.verify()
        with writable(self.db_path) as db:
            db.execute('UPDATE claims SET free = free - 1 WHERE phase = ?', ('injection_discovery',))
            db.execute("UPDATE meta SET inventory = inventory || ' '")
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_mutated_history_fails_even_if_stored_history_field_is_unchanged(self):
        with writable(self.db_path) as db:
            db.execute("INSERT INTO retired_evidence VALUES('prior', ?, 3, ?)", ('2' * 64, '3' * 64))
        with self.assertRaises(guard.PreclaimError):
            self.verify()

    def test_history_input_is_bounded_before_json_serialization(self):
        with writable(self.db_path) as db:
            db.execute('INSERT INTO reconciliations VALUES(?, ?, ?, ?, ?, ?)',
                       (1, '1' * 64, '2' * 64, '3' * 64, '{}', 'x' * 200))
            with mock.patch.object(guard, '_MAX_HISTORY', 100), mock.patch.object(
                    guard.json, 'dumps', side_effect=AssertionError('serialization should not start')):
                with self.assertRaises(guard.PreclaimError):
                    guard._history_hash(db)

    def test_connection_close_preserves_first_cancellation(self):
        for primary in (KeyboardInterrupt(), SystemExit(23)):
            with self.subTest(primary=type(primary).__name__):
                connection = mock.Mock()
                connection.close.side_effect = OSError('fabricated close failure')
                with mock.patch.object(guard.sqlite3, 'connect', return_value=connection):
                    with self.assertRaises(type(primary)) as outcome:
                        with guard._read_only_connection(self.db_path):
                            raise primary
                self.assertIs(outcome.exception, primary)
                self.assertIsInstance(outcome.exception.__cause__, OSError)

    def test_new_cancellation_during_close_overrides_ordinary_error(self):
        connection = mock.Mock()
        cancellation = SystemExit(19)
        connection.close.side_effect = cancellation
        with mock.patch.object(guard.sqlite3, 'connect', return_value=connection):
            with self.assertRaises(SystemExit) as outcome:
                with guard._read_only_connection(self.db_path):
                    raise OSError('fabricated query failure')
        self.assertIs(outcome.exception, cancellation)
        self.assertIsInstance(outcome.exception.__cause__, OSError)

    def test_verified_lock_serializes_competing_ledger_writers_until_release(self):
        with guard.hold_existing_claim(self.db_path, 'injection_heldout',
                                       output_path=self.phase, anchor=self.anchor) as valid:
            self.assertIs(valid, True)
            with closing(sqlite3.connect(self.db_path, timeout=0, isolation_level=None)) as other:
                with self.assertRaises(sqlite3.OperationalError):
                    other.execute('BEGIN IMMEDIATE')
        with closing(sqlite3.connect(self.db_path, timeout=0, isolation_level=None)) as other:
            other.execute('BEGIN IMMEDIATE')
            other.execute('ROLLBACK')

    def test_interrupt_in_verified_lock_releases_without_modifying_the_row(self):
        with self.assertRaises(KeyboardInterrupt):
            with guard.hold_existing_claim(self.db_path, 'injection_heldout',
                                           output_path=self.phase, anchor=self.anchor):
                raise KeyboardInterrupt()
        self.assertTrue(self.verify())


if __name__ == '__main__':
    unittest.main()
