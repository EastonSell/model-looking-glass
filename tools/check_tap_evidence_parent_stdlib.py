"""Exercise the exact storage helpers without importing the project or Torch.

This portable check does not exercise the controller or Windows volume rules.
"""
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / 'looking_glass/tap_repeated_gpu.py'
NAMES = {'_check_output_destination', '_evidence_parent', '_make_run_directory'}
tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
if {node.name for node in functions} != NAMES or len(functions) != len(NAMES):
    raise RuntimeError('storage helper definitions are missing or duplicated')
namespace = {'Path': Path, 'os': os, 'tempfile': tempfile}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class EvidenceParentTests(unittest.TestCase):
    def test_fresh_output_places_private_run_beside_result(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve()
            output = parent / 'result.json'
            self.assertEqual(namespace['_check_output_destination'](output), output)
            run = namespace['_make_run_directory'](output)
            self.assertEqual(run.parent, parent)
            self.assertTrue(run.is_dir())
            self.assertTrue(run.name.startswith('tap-repeated-v12-'))
            self.assertNotEqual(run, output)

    def test_noncanonical_missing_relative_and_symlink_parents_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve()
            nested = parent / 'nested'
            nested.mkdir()
            for output in (nested / '..' / 'result.json', parent / 'missing' / 'result.json',
                           Path('relative/result.json')):
                with self.subTest(output=output), self.assertRaises((ValueError, FileNotFoundError)):
                    namespace['_make_run_directory'](output)
            alias = parent / 'alias'
            try:
                alias.symlink_to(nested, target_is_directory=True)
            except (OSError, NotImplementedError):
                return  # Some Windows accounts cannot create symlinks.
            with self.assertRaises(ValueError):
                namespace['_make_run_directory'](alias / 'result.json')

    def test_existing_related_records_reject_output(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve()
            for suffix in ('', '.private.json', '.partial', '.incomplete.json'):
                output = parent / 'result.json'
                marker = Path(str(output) + suffix)
                marker.touch()
                with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                    namespace['_check_output_destination'](output)
                marker.unlink()

    def test_creation_failure_propagates(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw).resolve() / 'result.json'
            with mock.patch.object(tempfile, 'mkdtemp', side_effect=OSError('synthetic')):
                with self.assertRaisesRegex(OSError, 'synthetic'):
                    namespace['_make_run_directory'](output)

    def test_creation_result_outside_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve()
            evidence = parent / 'evidence'
            elsewhere = parent / 'elsewhere'
            evidence.mkdir()
            elsewhere.mkdir()
            with mock.patch.object(tempfile, 'mkdtemp', return_value=str(elsewhere)):
                with self.assertRaisesRegex(RuntimeError, 'does not match'):
                    namespace['_make_run_directory'](evidence / 'result.json')


if __name__ == '__main__':
    unittest.main()
