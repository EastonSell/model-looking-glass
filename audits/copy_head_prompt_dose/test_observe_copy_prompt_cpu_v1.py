"""Fabricated interruption tests for the direct-process observer; no model."""
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name("observe_copy_prompt_cpu_v1.py")
spec = importlib.util.spec_from_file_location("copy_observer_tested", SOURCE)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class FakeChild:
    pid = 12345

    def __init__(self):
        self.killed = False
        self.reaped = False

    def poll(self):
        return None if not self.reaped else -9

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.reaped = True
        return -9


class ObserverFaults(unittest.TestCase):
    def execute(self, popen, trace=None):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as tmp:
            root = Path(tmp)
            checkout, cache = root / "checkout", root / "cache"
            checkout.mkdir()
            cache.mkdir()
            manifest, reader = root / "manifest", root / "reader"
            manifest.write_bytes(b"fake manifest")
            reader.write_bytes(b"fake reader")
            replacements = {"CHECKOUT": checkout, "WORK": root / "output", "CACHE": cache,
                            "MANIFEST": manifest, "READER": reader, "BOOTSTRAP": root / "bootstrap",
                            "EXPECTED_MANIFEST": observer.sha(manifest),
                            "EXPECTED_READER": observer.sha(reader)}
            with mock.patch.multiple(observer, **replacements), \
                    mock.patch.object(observer, "git", side_effect=lambda *args: observer.EXPECTED_COMMIT if args[0] == "rev-parse" else ""), \
                    mock.patch.object(observer.subprocess, "Popen", side_effect=popen), \
                    mock.patch.object(sys, "argv", ["observer", "--public-prelaunch-commit", "a" * 40]), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    if trace:
                        sys.settrace(trace)
                    with self.assertRaises(KeyboardInterrupt):
                        observer.main()
                finally:
                    sys.settrace(None)
            files = list((root / "output").glob("*-exit.json"))
            self.assertEqual(len(files), 1)
            return json.loads(files[0].read_text(encoding="utf-8"))

    def test_interrupted_after_handle_binding_kills_and_reaps(self):
        child = FakeChild()
        lines, first = inspect.getsourcelines(observer.main)
        target = next(first + i for i, line in enumerate(lines) if "code = child.wait(timeout=TIMEOUT_SECONDS)" in line)
        fired = []
        def trace(frame, event, arg):
            if frame.f_code is observer.main.__code__ and event == "line" and frame.f_lineno == target and not fired:
                fired.append(True)
                raise KeyboardInterrupt()
            return trace
        record = self.execute(lambda *a, **kw: child, trace)
        self.assertTrue(fired)
        self.assertTrue(child.killed and child.reaped)
        self.assertEqual(record["child_pid"], 12345)
        self.assertTrue(record["reaped"])
        self.assertFalse(record["spawn_return_unbound_uncertain"])

    def test_interrupted_spawn_return_is_explicitly_uncertain(self):
        calls = []
        def popen(*args, **kwargs):
            calls.append(True)
            raise KeyboardInterrupt()
        record = self.execute(popen)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(record["child_pid"])
        self.assertFalse(record["reaped"])
        self.assertTrue(record["spawn_return_unbound_uncertain"])


if __name__ == "__main__":
    unittest.main()
