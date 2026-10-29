"""Model-free check that the builder executes captured bytes, not stale bytecode."""
import os
from pathlib import Path
import py_compile
import runpy
import tempfile
import unittest


class PrelaunchBuilderSourceTest(unittest.TestCase):
    def test_captured_source_wins_over_same_timestamp_cached_bytecode(self):
        builder = runpy.run_path(str(Path(__file__).with_name('build_prelaunch.py')))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'helper.py'
            source.write_text("VALUE = 'evil'\n", encoding='utf-8')
            py_compile.compile(str(source), doraise=True)
            stamp = int(source.stat().st_mtime)
            source.write_text("VALUE = 'good'\n", encoding='utf-8')
            os.utime(source, (stamp, stamp))
            raw = builder['bounded'](source, 1024)
            loaded = builder['load_captured']('_fixture', source, raw)
            self.assertEqual(loaded.VALUE, 'good')

    def test_bounded_read_rejects_oversized_input(self):
        builder = runpy.run_path(str(Path(__file__).with_name('build_prelaunch.py')))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'data'
            source.write_bytes(b'abcde')
            with self.assertRaisesRegex(ValueError, 'byte bound'):
                builder['bounded'](source, 4)


if __name__ == '__main__':
    unittest.main()
