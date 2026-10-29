"""Source-first entry tests; no model or protected data is used."""
import hashlib
import importlib.util
import os
from pathlib import Path
import py_compile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tools/audit_agreement_substitution_v1.py'
spec = importlib.util.spec_from_file_location('substitution_replay_entry_test', ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class SourceFirstEntryTests(unittest.TestCase):
    def test_actual_pinned_reader_loads(self):
        reader = entry.load_reader(ROOT / 'looking_glass/agreement_substitution_science_v1.py')
        self.assertTrue(callable(reader.audit_attempt))

    def test_captured_source_not_same_timestamp_bytecode(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'reader.py'
            source.write_bytes(b"VALUE = 'evil'\n")
            py_compile.compile(str(source), doraise=True)
            stamp = int(source.stat().st_mtime)
            source.write_bytes(b"VALUE = 'good'\n")
            os.utime(source, (stamp, stamp))
            with patch.object(entry, 'READER_SHA256', hashlib.sha256(source.read_bytes()).hexdigest()):
                self.assertEqual(entry.load_reader(source).VALUE, 'good')

    def test_changed_and_oversized_sources_reject_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'reader.py'
            source.write_bytes(b'raise AssertionError("must not run")\n')
            with self.assertRaisesRegex(ValueError, 'reviewed bytes'):
                entry.load_reader(source)
            source.write_bytes(b'#' * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, 'bounded'):
                entry.load_reader(source)

    def test_relative_or_aliased_attempt_rejects_output_inside_it(self):
        reader = entry.load_reader(ROOT / 'looking_glass/agreement_substitution_science_v1.py')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'attempt').mkdir()
            (folder / 'alias').mkdir()
            old = Path.cwd()
            try:
                os.chdir(folder)
                for attempt in ('attempt', str(folder / 'alias' / '..' / 'attempt')):
                    output = folder / 'attempt' / 'report.json'
                    args = ['--attempt', attempt, '--output', str(output),
                            '--cases', 'unused', '--directions', 'unused', '--numeric-source', 'unused',
                            '--source-revision', 'a' * 40, '--prelaunch-sha256', '1' * 64,
                            '--reservation-sha256', '2' * 64, '--latest-exposure-sha256', '3' * 64,
                            '--events-sha256', '4' * 64, '--start-scope',
                            'public prospective agreement noun-substitution']
                    with patch.object(reader, 'load_numeric', side_effect=AssertionError('must not load')):
                        with self.assertRaisesRegex(ValueError, 'outside the attempt'):
                            reader.main(args)
                    self.assertFalse(output.exists())
            finally:
                os.chdir(old)


if __name__ == '__main__':
    unittest.main()
