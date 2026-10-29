"""Source-first preparation child tests use fabricated source files only."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CHILD = Path(__file__).resolve().parents[1] / 'tools' / 'injection_preparation_child.py'


def load_child():
    assert CHILD.is_file(), 'the source-first preparation child is missing'
    spec = importlib.util.spec_from_file_location('_test_injection_preparation_child', CHILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SourceFirstTests(unittest.TestCase):
    def tearDown(self):
        for name in ('_injection_fixture_one', '_injection_fixture_two', '_injection_fixture_launcher'):
            sys.modules.pop(name, None)

    def test_changed_later_source_prevents_all_extra_source_execution(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / 'executed'
            first = root / 'first.py'
            second = root / 'second.py'
            first_raw = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n'.encode()
            first.write_bytes(first_raw)
            second.write_bytes(b'VALUE = 1\n')
            sources = [('_injection_fixture_one', '', first, sha(first_raw)),
                       ('_injection_fixture_two', '', second, sha(b'VALUE = 2\n'))]
            with self.assertRaises(ValueError):
                child.load_verified_modules(sources)
            self.assertFalse(marker.exists())
            self.assertNotIn('_injection_fixture_one', sys.modules)

    def test_no_model_configuration_rejects_relative_or_extra_paths(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = {'schema': 'mlg-injection-preparation-probe-v1',
                      'frozen_root': str(root), 'site_packages': str(root),
                      'cache_dir': str(root), 'source_root': str(root),
                      'heldout_runtime_file': str(root / 'runtime.json'),
                      'pycache_prefix': str(root)}
            raw = json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(child.parse_probe_config(raw), fields)
            fields['site_packages'] = 'relative-site'
            with self.assertRaises(ValueError):
                child.parse_probe_config(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode())
            fields['site_packages'] = str(root)
            fields['private_site'] = 'must-not-appear'
            with self.assertRaises(ValueError):
                child.parse_probe_config(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode())

    def test_configuration_file_read_is_bounded_before_parsing(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = {'schema': 'mlg-injection-preparation-probe-v1',
                      'frozen_root': str(root), 'site_packages': str(root),
                      'cache_dir': str(root), 'source_root': str(root),
                      'heldout_runtime_file': str(root / 'runtime.json'),
                      'pycache_prefix': str(root)}
            path = root / 'config.json'
            path.write_bytes(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode())
            real_open = Path.open

            class BoundedReader:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def read(self, size=-1):
                    if not 0 <= size <= 65537:
                        raise AssertionError('configuration read was not bounded')
                    return self.stream.read(size)

            def open_bounded(file_path, *args, **kwargs):
                return BoundedReader(real_open(file_path, *args, **kwargs))

            with mock.patch.object(Path, 'open', open_bounded):
                self.assertEqual(child.read_probe_config(path), fields)
                with real_open(path, 'wb') as stream:
                    stream.write(b'x' * 65537)
                with self.assertRaises(ValueError):
                    child.read_probe_config(path)
    def test_verified_modules_execute_in_order_with_their_real_source_paths(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / 'first.py'
            second = root / 'second.py'
            first_raw = b'VALUE = 37\n'
            second_raw = b'from _injection_fixture_one import VALUE\nRESULT = VALUE + 5\n'
            first.write_bytes(first_raw)
            second.write_bytes(second_raw)
            loaded = child.load_verified_modules([
                ('_injection_fixture_one', '', first, sha(first_raw)),
                ('_injection_fixture_two', '', second, sha(second_raw)),
            ])
            self.assertEqual(loaded['_injection_fixture_two'].RESULT, 42)
            self.assertEqual(Path(loaded['_injection_fixture_one'].__file__), first)

    def test_runtime_mismatch_stops_before_preparation_sources_execute(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher = root / 'launcher.py'
            extra = root / 'extra.py'
            marker = root / 'executed'
            launcher_raw = (
                b'_CONTEXT = None\n'
                b'def _boot(site, revision):\n'
                b'    assert site == "approved-site" and revision == "frozen-revision"\n'
                b'    return {"site": site}\n'
                b'def _initialize_runtime(*, source_commit, cache_dir, site_packages):\n'
                b'    assert _CONTEXT == {"site": site_packages}\n'
                b'    assert source_commit == "frozen-revision" and cache_dir == "offline-cache"\n'
                b'    return {"runtime_bytes": b"different"}\n'
            )
            extra_raw = f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n'.encode()
            launcher.write_bytes(launcher_raw)
            extra.write_bytes(extra_raw)
            with self.assertRaises(ValueError):
                child.bootstrap_runtime(
                    ('_injection_fixture_launcher', '', launcher, sha(launcher_raw)),
                    [('_injection_fixture_one', '', extra, sha(extra_raw))],
                    site_packages='approved-site', cache_dir='offline-cache',
                    source_commit='frozen-revision', expected_runtime=b'published',
                    publication_check=lambda path, data: True,
                )
            self.assertFalse(marker.exists())
            self.assertNotIn('_injection_fixture_one', sys.modules)

    def test_runtime_boot_installs_the_approved_site_before_extra_source(self):
        child = load_child()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher = root / 'launcher.py'
            extra = root / 'extra.py'
            site = str(root / 'approved-site')
            launcher_raw = (
                b'import sys\n_CONTEXT = None\n'
                b'def _boot(site, revision):\n'
                b'    sys.path.append(site)\n    return {"site": site}\n'
                b'def _initialize_runtime(*, source_commit, cache_dir, site_packages):\n'
                b'    assert _CONTEXT["site"] == site_packages\n'
                b'    assert sys.path.count(site_packages) == 1\n'
                b'    return {"runtime_bytes": b"published"}\n'
            )
            extra_raw = b'from pathlib import Path\nSOURCE_NAME = Path(__file__).name\n'
            launcher.write_bytes(launcher_raw)
            extra.write_bytes(extra_raw)
            try:
                _, state, modules = child.bootstrap_runtime(
                    ('_injection_fixture_launcher', '', launcher, sha(launcher_raw)),
                    [('_injection_fixture_one', '', extra, sha(extra_raw))],
                    site_packages=site, cache_dir='offline-cache', source_commit='frozen-revision',
                    expected_runtime=b'published', publication_check=lambda path, data: True,
                )
                self.assertEqual(state['runtime_bytes'], b'published')
                self.assertEqual(modules['_injection_fixture_one'].SOURCE_NAME, 'extra.py')
            finally:
                if site in sys.path:
                    sys.path.remove(site)

    def test_no_model_child_requires_the_supervisor_start_token(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + temp,
                 str(CHILD), '--validate-only', str(Path(temp) / 'config.json')],
                input=b'not-the-start-token\n', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=15, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(b'not-the-start-token', result.stderr)


if __name__ == '__main__':
    unittest.main()
