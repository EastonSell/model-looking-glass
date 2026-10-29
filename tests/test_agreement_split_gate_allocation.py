"""Fabricated-only source-first shared-gate tests; no real study allocation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('natural_gate_caller_test', HERE / 'tools' / 'agreement_split_gate_allocation.py')
CALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALLER)
REVIEWED = HERE / 'looking_glass'


class GateCallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'gate'
        self.root.mkdir()
        self.source_root = self.base / 'source'
        (self.source_root / 'looking_glass').mkdir(parents=True)
        for name in CALLER.SOURCE_HASHES:
            shutil.copyfile(REVIEWED / (name + '.py'), self.source_root / 'looking_glass' / (name + '.py'))
        st = self.root.lstat()
        path_hash = hashlib.sha256(os.path.normcase(os.path.abspath(str(self.root))).encode('utf-8')).hexdigest()
        self.patch = mock.patch.multiple(CALLER, GATE_ROOT_SHA256=path_hash,
                                        GATE_ROOT_IDENTITY=(st.st_dev, st.st_ino))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.clear_modules)

    @staticmethod
    def clear_modules():
        for name in tuple(sys.modules):
            if name == 'looking_glass' or name.startswith('looking_glass.'):
                del sys.modules[name]

    @staticmethod
    def reader(name):
        return (REVIEWED / (name + '.py')).read_bytes()

    def test_source_checked_allocation_is_single_use(self):
        result = CALLER.operate('allocate', 'natural', self.root, self.source_root, reader=self.reader)
        self.assertEqual(result, {'status': 'allocation_announced', 'track': 'natural'})
        marker = json.loads((self.root / 'natural-allocation.json').read_bytes())
        self.assertEqual(marker['freeze_sha256'], CALLER.FREEZES['natural'])
        self.assertEqual(set(p.name for p in self.root.iterdir()), {'gate.lock', 'natural-allocation.json'})
        self.clear_modules()
        with self.assertRaises(Exception):
            CALLER.operate('allocate', 'natural', self.root, self.source_root, reader=self.reader)
        self.assertFalse(any('nonce' in p.name for p in self.root.iterdir()))

    def test_changed_dependency_executes_no_project_code_or_marker(self):
        marker = self.base / 'untrusted-executed'
        target = self.source_root / 'looking_glass' / 'agreement_split_gate.py'
        target.write_bytes(target.read_bytes() + ("\nopen(%r, 'w').write('bad')\n" % str(marker)).encode())
        with self.assertRaises(CALLER.CallerError):
            CALLER.operate('allocate', 'natural', self.root, self.source_root, reader=self.reader)
        self.assertFalse(marker.exists())
        self.assertFalse(any(self.root.iterdir()))
        self.assertNotIn('looking_glass', sys.modules)

    def test_unreviewed_gate_root_is_rejected_before_lock(self):
        other = self.base / 'other'
        other.mkdir()
        with self.assertRaises(CALLER.CallerError):
            CALLER.operate('allocate', 'natural', other, self.source_root, reader=self.reader)
        self.assertFalse(any(other.iterdir()))

    def test_inspection_only_makes_the_lock(self):
        result = CALLER.operate('inspect', 'natural', self.root, self.source_root, reader=self.reader)
        self.assertEqual(result, {'status': 'inspected', 'track': 'natural'})
        self.assertEqual({p.name for p in self.root.iterdir()}, {'gate.lock'})


if __name__ == '__main__':
    unittest.main()
