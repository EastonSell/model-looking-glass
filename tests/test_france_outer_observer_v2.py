"""Synthetic direct-process observation; no checkpoint or study input."""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

root = pathlib.Path(__file__).resolve().parents[1]
path = root / "src/scripts/observe_france_baseline_v2.py"
if not path.exists():
    path = root / "scripts/observe_france_baseline_v2.py"
spec = importlib.util.spec_from_file_location("france_observer_test", path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class DirectObservationTests(unittest.TestCase):
    def test_observer_requires_matching_complete_operator_start(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "operator-start.json"
            with self.assertRaises(RuntimeError):
                subject.await_operator_start(path, "a" * 64, "b" * 40, "c" * 64, timeout=0.01)
            row = {"pid": os.getpid(), "source_sha256": "a" * 64, "source_revision": "b" * 40,
                   "operator_sha256": "c" * 64,
                   "propagation_report_sha256": subject.PROPAGATION_SHA256,
                   "started_utc": "2026-01-01T00:00:00Z", "scope": "cooperative observation"}
            path.write_text(json.dumps(row), encoding="utf-8")
            subject.await_operator_start(path, "a" * 64, "b" * 40, "c" * 64, timeout=0.1)
            row["pid"] += 1; path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                subject.await_operator_start(path, "a" * 64, "b" * 40, "c" * 64, timeout=0.1)
            path.write_text('{"pid": 1, "pid": 1}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                subject.await_operator_start(path, "a" * 64, "b" * 40, "c" * 64, timeout=0.1)

    def test_reaped_exit_and_logs_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subject.run_direct([sys.executable, "-I", "-B", "-c", "print('done')"],
                                        pathlib.Path(temp), 10, {})
            self.assertEqual(result["status"], "reaped")
            self.assertEqual(result["exit_code"], 0)
            self.assertFalse(result["timed_out"])
            self.assertGreater(result["pid"], 0)
            self.assertIn(b"done", (pathlib.Path(temp) / "stdout.log").read_bytes())

    def test_nonzero_and_timeout_never_report_complete(self):
        for code, timeout in (("raise SystemExit(7)", 10), ("import time; time.sleep(3)", 0.15)):
            with tempfile.TemporaryDirectory() as temp:
                result = subject.run_direct([sys.executable, "-I", "-B", "-c", code],
                                            pathlib.Path(temp), timeout, {})
                self.assertEqual(result["status"], "incomplete")
                if timeout < 1:
                    self.assertTrue(result["timed_out"])
                    self.assertEqual(result["descendant_cleanup"], "unverified")

    def test_interrupted_wait_kills_and_reaps_the_direct_process(self):
        process = mock.Mock(pid=123)
        process.wait.side_effect = [KeyboardInterrupt(), 1]
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(subject.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                subject.run_direct([sys.executable], pathlib.Path(temp), 10, {})
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_lost_popen_return_does_not_release_start_gate(self):
        real_popen = subject.subprocess.Popen
        created = []
        with tempfile.TemporaryDirectory() as temp:
            directory = pathlib.Path(temp)
            sentinel = directory / "forwarded.txt"
            code = ("import pathlib,sys; token=sys.stdin.buffer.read(6); "
                    "pathlib.Path(sys.argv[1]).write_text('forward') if token==b'start\\n' else None")
            def lost_return(*args, **kwargs):
                child = real_popen(*args, **kwargs); created.append(child)
                raise RuntimeError("synthetic exception at Popen return")
            with mock.patch.object(subject.subprocess, "Popen", side_effect=lost_return):
                with self.assertRaises(RuntimeError):
                    subject.run_direct([sys.executable, "-I", "-B", "-c", code, str(sentinel)],
                                       directory, 10, {}, gate=b"start\n")
            self.assertEqual(len(created), 1)
            created[0].stdin.close()
            self.assertEqual(created[0].wait(timeout=10), 0)
            self.assertFalse(sentinel.exists())

    def test_start_gate_is_sent_only_after_start_record(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = pathlib.Path(temp)
            code = "import sys; print(sys.stdin.buffer.read(6).decode().strip())"
            result = subject.run_direct([sys.executable, "-I", "-B", "-c", code],
                                        directory, 10, {}, gate=b"start\n")
            self.assertEqual(result["exit_code"], 0)
            self.assertIn(b"start", (directory / "stdout.log").read_bytes())
            self.assertTrue((directory / "owner-start.json").is_file())


if __name__ == "__main__":
    unittest.main()
