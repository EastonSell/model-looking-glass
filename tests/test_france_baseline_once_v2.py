"""Synthetic single-use owner boundary tests; no checkpoint or fixed inputs."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src" if (_root / "src" / "looking_glass").is_dir() else _root))
from looking_glass import targeted_edit_france_baseline_v2 as core
from looking_glass import targeted_edit_france_baseline_once_v2 as subject

spec = importlib.util.spec_from_file_location("_synthetic_baseline", pathlib.Path(__file__).with_name("test_france_baseline_v2.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def make(directory):
    fixture.torch.set_flush_denormal(False)
    guard = fixture.guard()
    rows, locality = fixture.inputs()
    screen = pathlib.Path(directory) / "screen"
    invoke = lambda: core.run_rows(guard, rows, locality, 1, 2, screen,
                                   vocabulary=40, identity={"fixture": True})
    return guard, screen, subject.OneShot(guard, screen, invoke)


class OnceTests(unittest.TestCase):
    def test_clean_return_is_single_use_and_guard_is_fenced(self):
        with tempfile.TemporaryDirectory() as directory:
            guard, screen, session = make(directory)
            result = session.run()
            self.assertEqual(result["baseline"]["status"], "eligible")
            self.assertTrue((screen.parent / "owner-return.json").is_file())
            self.assertTrue(guard.poisoned)
            self.assertEqual(guard.model.calls, 27)
            with self.assertRaises(subject.OwnerBoundaryError):
                session.run()
            self.assertEqual(guard.model.calls, 27)

    def test_exception_in_core_return_after_completion_is_incomplete_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            guard, screen, session = make(directory)
            def trace(frame, event, arg):
                if frame.f_code is core.run_rows.__code__ and event == "return":
                    raise RuntimeError("synthetic return-event interruption")
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaisesRegex(RuntimeError, "return-event"):
                    session.run()
            finally:
                sys.settrace(None)
            self.assertTrue((screen / "completed.json").is_file())
            self.assertTrue((screen.parent / "owner-incomplete.json").is_file())
            self.assertFalse((screen.parent / "owner-return.json").exists())
            self.assertEqual(json.loads((screen.parent / "owner-incomplete.json").read_bytes())["error_type"], "RuntimeError")
            self.assertTrue(guard.poisoned)
            with self.assertRaises(subject.OwnerBoundaryError):
                session.run()
            self.assertEqual(guard.model.calls, 27)

    def test_exception_in_owner_return_still_fences_and_requires_outer_exit_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            guard, screen, session = make(directory)
            def trace(frame, event, arg):
                if frame.f_code is subject.OneShot.run.__code__ and event == "return":
                    raise RuntimeError("synthetic outer return interruption")
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaisesRegex(RuntimeError, "outer return"):
                    session.run()
            finally:
                sys.settrace(None)
            self.assertTrue((screen.parent / "owner-return.json").is_file())
            self.assertTrue(guard.poisoned)
            with self.assertRaises(subject.OwnerBoundaryError):
                session.run()
            self.assertEqual(guard.model.calls, 27)


if __name__ == "__main__":
    unittest.main()
