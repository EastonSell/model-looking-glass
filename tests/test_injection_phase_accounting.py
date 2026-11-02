"""Metadata-only fabricated phase accounting; no private study data."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from looking_glass import injection_phase_accounting as accounting


class PhaseAccountingTests(unittest.TestCase):
    def setUp(self):
        work = Path.cwd() / 'work'
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.parent = self.root / 'protected'
        self.parent.mkdir()
        self.phase = self.parent / 'injection-heldout-fabricated'
        self.pilot = self.parent / 'runtime-compatibility-replay-v1'
        self.pilot.mkdir()
        self.rehearsal = self.root / 'rehearsal'
        self.rehearsal.mkdir()
        (self.pilot / 'evidence.bin').write_bytes(b'p' * 30)
        (self.rehearsal / 'diagnostic.bin').write_bytes(b'd' * 20)
        (self.parent / 'claim-intent.json').write_bytes(b'i' * 10)
        self.marker = self.parent / (self.phase.name + '.preparation-started.json')
        self.marker.write_bytes(b'm' * 5)
        def identity(path):
            info = path.stat()
            return info.st_dev, info.st_ino
        parent_dev, parent_ino = identity(self.parent)
        pilot_dev, pilot_ino = identity(self.pilot)
        rehearsal_dev, rehearsal_ino = identity(self.rehearsal)
        self.anchor = accounting.PhaseRoots(parent_path=str(self.parent), parent_dev=parent_dev,
            parent_ino=parent_ino, phase_path=str(self.phase), pilot_path=str(self.pilot),
            pilot_dev=pilot_dev, pilot_ino=pilot_ino, rehearsal_path=str(self.rehearsal),
            rehearsal_dev=rehearsal_dev, rehearsal_ino=rehearsal_ino)

    def check(self, stage='before_phase_creation', *, anchor=None, logical=128):
        return accounting.reconcile_phase_files(anchor or self.anchor, stage=stage,
                                                replay_tensor_bytes=logical)

    def test_counts_both_stages_and_every_regular_file(self):
        before = self.check()
        self.assertEqual((before['pilot_bytes'], before['rehearsal_bytes'], before['other_bytes']), (30, 20, 15))
        self.assertEqual(before['file_count'], 4)
        self.phase.mkdir()
        (self.phase / 'claim.json').write_bytes(b'c' * 7)
        after = self.check('before_nonce')
        self.assertEqual((after['pilot_bytes'], after['rehearsal_bytes'], after['other_bytes']), (30, 20, 22))
        self.assertEqual(after['file_count'], 5)

    def test_phase_stage_and_preparation_marker_are_required(self):
        with self.assertRaises(accounting.AccountingError):
            self.check('before_nonce')
        self.phase.mkdir()
        (self.phase / 'claim.json').write_bytes(b'claim')
        with self.assertRaises(accounting.AccountingError):
            self.check('before_phase_creation')
        (self.phase / 'extra').write_bytes(b'partial')
        with self.assertRaises(accounting.AccountingError):
            self.check('before_nonce')
        self.marker.unlink()
        with self.assertRaises(accounting.AccountingError):
            self.check('before_nonce')

    def test_each_fixed_reservation_and_tensor_limit_is_checked(self):
        for field, value in (('_PILOT_BYTES', 29), ('_REHEARSAL_BYTES', 19), ('_OTHER_BYTES', 14)):
            with self.subTest(field=field), mock.patch.object(accounting, field, value):
                with self.assertRaises(accounting.AccountingError):
                    self.check()
        for logical in (True, -1, 8 * 1024**2 + 1):
            with self.subTest(logical=logical), self.assertRaises(accounting.AccountingError):
                self.check(logical=logical)

    def test_replaced_root_hardlink_and_extra_path_overlap_are_rejected(self):
        wrong = replace(self.anchor, parent_ino=self.anchor.parent_ino + 1)
        with self.assertRaises(accounting.AccountingError):
            self.check(anchor=wrong)
        wrong = replace(self.anchor, rehearsal_path=str(self.pilot),
                        rehearsal_dev=self.anchor.pilot_dev, rehearsal_ino=self.anchor.pilot_ino)
        with self.assertRaises(accounting.AccountingError):
            self.check(anchor=wrong)
        os.link(self.pilot / 'evidence.bin', self.rehearsal / 'linked.bin')
        with self.assertRaises(accounting.AccountingError):
            self.check()

    def test_mutation_between_scans_fails_closed(self):
        original = accounting._snapshot
        count = []
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            count.append(1)
            if len(count) == 2:
                (self.pilot / 'evidence.bin').write_bytes(b'changed')
            return result
        with mock.patch.object(accounting, '_snapshot', side_effect=changed):
            with self.assertRaises(accounting.AccountingError):
                self.check()


if __name__ == '__main__':
    unittest.main()
