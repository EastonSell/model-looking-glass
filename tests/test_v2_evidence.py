"""Fabricated-only evidence tests; no study inputs or model cases."""
import json
import copy
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

from looking_glass import registry
from looking_glass.v2_evidence import EvidenceError, create_evidence, open_evidence


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def chunk(name, state, records):
    return encoded({"schema": "causal-v2-chunk-v1", "name": name,
                    "state": state, "records": records})


def failure(message="fabricated", kind="RuntimeError", outcome="failed"):
    return encoded({"schema": "causal-v2-failure-v1", "outcome": outcome, "stage": "discovery",
                    "error_type": kind, "message": message, "submissions": 1,
                    "root_forwards": 1, "cell_id": "opaque", "phase": "discovery"})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run"
        self.run = create_evidence(
            self.path, protocol={"id": "fabricated"}, inputs={}, model={}, code={},
            metadata={"label": "development-only"},
            chunk_limits={"one.json": 4096, "two.json": 4096},
            extra_limits={"precommit.json": 4096}, required_extra=["precommit.json"],
            failure_limit=4096, external_reserve_bytes=1024,
            disk_limit_bytes=8 * 1024 * 1024, free_margin_bytes=0,
        )

    def test_success_requires_every_complete_chunk_and_freeze(self):
        self.run.publish_artifact("precommit.json", encoded({"prediction": 1}))
        self.run.publish_chunk("one.json", chunk("one.json", "complete", [{"bits": "80000000"}]))
        self.run.publish_chunk("two.json", chunk("two.json", "complete", []))
        self.run.finalize("succeeded", {"count": 2})
        self.assertEqual(self.run.status()["state"], "succeeded")
        self.assertEqual(registry.verify_run(self.path)["state"], "succeeded")

    def test_incomplete_chunk_and_failure_are_saved_without_claiming_success(self):
        data = chunk("one.json", "incomplete", [{"completed": 1}])
        self.run.publish_chunk("one.json", data)
        self.run.publish_failure(failure())
        self.run.finalize("failed", {"stage": "discovery", "count": 1})
        self.assertEqual((self.path / "artifacts" / "one.json").read_bytes(), data)
        self.assertEqual(self.run.status()["state"], "failed")

    def test_bad_or_noncanonical_chunk_never_creates_artifact(self):
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("one.json", chunk("different.json", "complete", []))
        self.assertFalse((self.path / "artifacts" / "one.json").exists())
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("one.json", chunk("one.json", "complete", []))

    def test_prepublication_limit_error_allows_one_complete_failure_record(self):
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("one.json", b"x" * 4097)
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("two.json", chunk("two.json", "complete", []))
        data = failure("full evidence did not fit", kind="EvidenceLimit")
        self.run.publish_failure(data)
        self.run.finalize("failed", {"count": 1})
        self.assertEqual((self.path / "artifacts" / "failure.json").read_bytes(), data)
        self.assertEqual(self.run.status()["state"], "failed")

    def test_success_with_missing_chunk_poisoned_instead_of_resumed(self):
        self.run.publish_artifact("precommit.json", encoded({"prediction": 1}))
        self.run.publish_chunk("one.json", chunk("one.json", "complete", []))
        with self.assertRaises(EvidenceError):
            self.run.finalize("succeeded", {})
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("two.json", chunk("two.json", "complete", []))
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_incomplete_chunk_cannot_be_success(self):
        self.run.publish_chunk("one.json", chunk("one.json", "incomplete", []))
        self.run.publish_chunk("two.json", chunk("two.json", "complete", []))
        self.run.publish_artifact("precommit.json", encoded({"prediction": 1}))
        with self.assertRaises(EvidenceError):
            self.run.finalize("succeeded", {})
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_pending_bytes_are_retained_and_block_mutations(self):
        pending = self.path / ".pending"
        pending.write_bytes(b'{"partial":')
        self.assertTrue(self.run.status()["pending_write"])
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("one.json", chunk("one.json", "complete", []))
        with self.assertRaises(EvidenceError):
            self.run.finalize("failed", {})
        self.assertEqual(pending.read_bytes(), b'{"partial":')

    def test_failed_terminal_requires_bounded_failure_record(self):
        with self.assertRaises(EvidenceError):
            self.run.finalize("failed", {})
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_oversize_failure_is_not_truncated(self):
        with self.assertRaises(EvidenceError):
            self.run.publish_failure(b"x" * 4097)
        self.assertFalse((self.path / "artifacts" / "failure.json").exists())
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_empty_failure_is_not_complete_evidence(self):
        with self.assertRaises(EvidenceError):
            self.run.publish_failure(b"")
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_failure_record_must_match_terminal_state(self):
        self.run.publish_failure(failure(outcome="failed"))
        with self.assertRaises(EvidenceError):
            self.run.finalize("cancelled", {})
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_cancelled_failure_can_finalize_cancelled(self):
        self.run.publish_failure(failure(outcome="cancelled"))
        self.run.finalize("cancelled", {"reason": "cancelled"})
        self.assertEqual(self.run.status()["state"], "cancelled")

    def test_reopened_handle_is_inspection_only(self):
        reopened = open_evidence(self.path)
        with self.assertRaises(EvidenceError):
            reopened.publish_chunk("one.json", chunk("one.json", "complete", [1]))
        self.run.publish_chunk("one.json", chunk("one.json", "complete", [1]))
        self.assertEqual(self.run.status()["state"], "incomplete")

    def test_writer_handle_cannot_be_copied_or_serialized(self):
        for method in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.assertRaises(TypeError):
                method(self.run)

    def test_exception_after_publication_poisoned_even_without_pending(self):
        import looking_glass.v2_evidence as module
        publish = module._publish_preserving_pending
        def publish_then_fail(*args):
            publish(*args)
            raise KeyboardInterrupt("injected after unlink")
        with patch.object(module, "_publish_preserving_pending", side_effect=publish_then_fail):
            with self.assertRaises(KeyboardInterrupt):
                self.run.publish_chunk("one.json", chunk("one.json", "complete", []))
        state = self.run.status()
        self.assertFalse(state["pending_write"])
        self.assertIn("one.json", state["artifacts"])
        with self.assertRaises(EvidenceError):
            self.run.publish_chunk("two.json", chunk("two.json", "complete", []))

    def test_declared_name_and_registry_payload_ceilings(self):
        target = Path(self.tmp.name) / "over-limit"
        with self.assertRaises(EvidenceError):
            create_evidence(target, protocol={}, inputs={}, model={}, code={}, metadata={},
                            chunk_limits={f"c{i}.json": 1 for i in range(128)},
                            extra_limits={}, required_extra=[], failure_limit=1,
                            disk_limit_bytes=8 * 1024 * 1024, free_margin_bytes=0)
        self.assertFalse(target.exists())
        with self.assertRaises(EvidenceError):
            create_evidence(target, protocol={}, inputs={}, model={}, code={}, metadata={},
                            chunk_limits={"c.json": registry.MAX_TOTAL_BYTES},
                            extra_limits={}, required_extra=[], failure_limit=1,
                            disk_limit_bytes=1024 ** 3, free_margin_bytes=0)
        self.assertFalse(target.exists())

    def test_insufficient_disk_rejected_before_run_creation(self):
        path = Path(self.tmp.name) / "no-space"
        with patch("looking_glass.v2_evidence.shutil.disk_usage") as usage:
            usage.return_value.free = 1536 * 1024 * 1024
            with self.assertRaises(EvidenceError):
                create_evidence(path, protocol={}, inputs={}, model={}, code={}, metadata={},
                                chunk_limits={"one.json": 1}, extra_limits={},
                                required_extra=[], failure_limit=1,
                                disk_limit_bytes=4 * 1024 * 1024, free_margin_bytes=0)
        self.assertFalse(path.exists())

    def test_short_stream_writes_do_not_truncate_a_published_artifact(self):
        original = Path.open
        class ShortWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def write(self, data):
                return self.stream.write(data[:2])
            def flush(self):
                return self.stream.flush()
            def fileno(self):
                return self.stream.fileno()
            def close(self):
                return self.stream.close()
        def open_short(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            return ShortWriter(stream) if path.name == ".pending" and args[0] == "xb" else stream
        payload = encoded({"complete": "precommit"})
        with patch.object(Path, "open", new=open_short):
            self.run.publish_artifact("precommit.json", payload)
        self.assertEqual((self.path / "artifacts" / "precommit.json").read_bytes(), payload)

    def test_zero_progress_keeps_pending_and_never_publishes(self):
        original = Path.open
        class NoProgress:
            def __init__(self, stream):
                self.stream = stream
            def write(self, data):
                return 0
            def close(self):
                return self.stream.close()
        def open_no_progress(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            return NoProgress(stream) if path.name == ".pending" and args[0] == "xb" else stream
        with patch.object(Path, "open", new=open_no_progress):
            with self.assertRaises(OSError):
                self.run.publish_artifact("precommit.json", b"complete")
        self.assertFalse((self.path / "artifacts" / "precommit.json").exists())
        self.assertTrue(self.run.status()["pending_write"])
        with self.assertRaises(EvidenceError):
            self.run.publish_artifact("precommit.json", b"complete")

    def test_publication_error_remains_primary_if_lock_release_fails(self):
        primary = ValueError("primary publication")
        cleanup = OSError("secondary lock release")
        @contextmanager
        def broken_lock(*args):
            try:
                yield
            finally:
                raise cleanup
        with patch("looking_glass.v2_evidence.registry._lock", new=broken_lock), \
                patch("looking_glass.v2_evidence._publish_preserving_pending", side_effect=primary):
            with self.assertRaises(ValueError) as caught:
                self.run.publish_artifact("precommit.json", b"evidence")
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)

    def test_write_error_remains_primary_if_file_close_fails(self):
        original_open = Path.open
        primary = ValueError("primary write")
        cleanup = OSError("secondary close")
        class BrokenWriter:
            def __init__(self, stream):
                self.stream = stream
            def write(self, data):
                raise primary
            def close(self):
                self.stream.close()
                raise cleanup
        def open_broken(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return BrokenWriter(stream) if path.name == ".pending" and args[0] == "xb" else stream
        with patch.object(Path, "open", new=open_broken):
            with self.assertRaises(ValueError) as caught:
                self.run.publish_artifact("precommit.json", b"evidence")
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)
        self.assertTrue(self.run.status()["pending_write"])

    def test_interrupt_during_cleanup_remains_primary(self):
        primary = ValueError("primary publication")
        interrupt = KeyboardInterrupt("interrupted release")
        @contextmanager
        def broken_lock(*args):
            try:
                yield
            finally:
                raise interrupt
        with patch("looking_glass.v2_evidence.registry._lock", new=broken_lock), \
                patch("looking_glass.v2_evidence._publish_preserving_pending", side_effect=primary):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run.publish_artifact("precommit.json", b"evidence")
        self.assertIs(caught.exception, interrupt)
        self.assertIs(caught.exception.__cause__, primary)


class ProcessCrashTests(unittest.TestCase):
    def test_killed_owner_preserves_partial_pending_bytes(self):
        script = r"""
import os, sys
from pathlib import Path
import looking_glass.v2_evidence as e
path = Path(sys.argv[1])
run = e.create_evidence(path, protocol={}, inputs={}, model={}, code={}, metadata={},
    chunk_limits={'cell.json': 4096}, extra_limits={'precommit.json': 4096},
    required_extra=[], failure_limit=1024, disk_limit_bytes=8*1024*1024)
original = Path.open
class PartialWriter:
    def __init__(self, stream): self.stream = stream
    def write(self, data):
        self.stream.write(data[:5]); self.stream.flush(); os.fsync(self.stream.fileno())
        os._exit(19)
    def close(self): self.stream.close()
def partial(path, *args, **kwargs):
    stream = original(path, *args, **kwargs)
    return PartialWriter(stream) if path.name == '.pending' and args[0] == 'xb' else stream
Path.open = partial
run.publish_artifact('precommit.json', b'partial-evidence')
raise AssertionError('fault did not fire')
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            child = subprocess.run([sys.executable, "-c", script, str(path)],
                                   capture_output=True, timeout=15)
            self.assertEqual(child.returncode, 19, child.stderr.decode(errors="replace"))
            state = open_evidence(path).status()
            self.assertEqual(state["state"], "incomplete")
            self.assertTrue(state["pending_write"])
            self.assertEqual((path / ".pending").read_bytes(), b"parti")
            self.assertNotIn("precommit.json", state["artifacts"])

    def test_killed_owner_preserves_unacknowledged_publication(self):
        # Each child creates its own run: reopening is deliberately read-only.
        script = r"""
import json, os, sys
from pathlib import Path
import looking_glass.v2_evidence as e
path, mode = Path(sys.argv[1]), sys.argv[2]
run = e.create_evidence(path, protocol={}, inputs={}, model={}, code={}, metadata={},
    chunk_limits={'cell.json': 4096}, extra_limits={}, required_extra=[],
    failure_limit=1024, disk_limit_bytes=8*1024*1024, free_margin_bytes=0)
raw = json.dumps({'schema':'causal-v2-chunk-v1','name':'cell.json',
                  'state':'complete','records':[1]},sort_keys=True,separators=(',',':')).encode()
if mode in ('before_link','after_link'):
    original = e.os.link
    def stop(src, dst):
        if mode == 'after_link':
            original(src, dst)
        os._exit(19)
    e.os.link = stop
elif mode == 'after_unlink':
    calls = [0]
    original = e._sync_directory
    def stop(path):
        calls[0] += 1
        original(path)
        if calls[0] == 3:
            os._exit(19)
    e._sync_directory = stop
run.publish_chunk('cell.json', raw)
raise AssertionError('fault did not fire')
"""
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("before_link", "after_link", "after_unlink"):
                with self.subTest(mode=mode):
                    path = Path(temp) / mode
                    child = subprocess.run([sys.executable, "-c", script, str(path), mode],
                                           capture_output=True, timeout=15)
                    self.assertEqual(child.returncode, 19, child.stderr.decode(errors="replace"))
                    reopened = open_evidence(path)
                    state = reopened.status()
                    self.assertEqual(state["state"], "incomplete")
                    self.assertEqual(state["pending_write"], mode != "after_unlink")
                    self.assertEqual("cell.json" in state["artifacts"], mode != "before_link")
                    if mode == "after_link":
                        self.assertEqual(state["pending"]["linked"], ["cell.json"])
                    with self.assertRaises(EvidenceError):
                        reopened.publish_chunk("cell.json", chunk("cell.json", "complete", [1]))


if __name__ == "__main__":
    unittest.main()
