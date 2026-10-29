"""CPU-only fault tests for create-only calibration evidence publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from looking_glass.bert_gpu_evidence import write_json_new


class EvidenceWriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "result.json"
        self.partial = Path(self.directory.name) / "result.json.partial"

    def test_canonical_create_only_and_destination_conflict(self):
        write_json_new(self.destination, {"z": 2, "a": "hello"})
        expected = b'{"a":"hello","z":2}'
        self.assertEqual(self.destination.read_bytes(), expected)
        self.assertFalse(self.partial.exists())
        with self.assertRaises(FileExistsError):
            write_json_new(self.destination, {"replace": True})
        self.assertEqual(self.destination.read_bytes(), expected)

    def test_short_writes_are_retried_until_complete(self):
        original = os.write
        chunks = []

        def short_write(fd, data):
            written = original(fd, data[:3])
            chunks.append(written)
            return written

        with patch("looking_glass.bert_gpu_evidence.os.write", side_effect=short_write):
            write_json_new(self.destination, {"abc": "0123456789"})
        self.assertGreater(len(chunks), 1)
        self.assertEqual(json.loads(self.destination.read_bytes()), {"abc": "0123456789"})

    def test_interrupted_write_leaves_partial_without_final(self):
        original = os.write
        calls = 0

        def interrupted(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(fd, data[:2])
            raise OSError("injected write failure")

        with patch("looking_glass.bert_gpu_evidence.os.write", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                write_json_new(self.destination, {"data": "long enough"})
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.partial.read_bytes(), b'{"')
        with self.assertRaises(FileExistsError):
            write_json_new(self.destination, {"retry": True})
        self.assertEqual(self.partial.read_bytes(), b'{"')

    def test_nonprogress_and_invalid_write_counts_fail_closed(self):
        for value in (0, -1, True, 999999):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output.json"
                with patch("looking_glass.bert_gpu_evidence.os.write", return_value=value):
                    with self.assertRaises(OSError):
                        write_json_new(output, {"x": 1})
                self.assertFalse(output.exists())
                self.assertTrue(output.with_name(output.name + ".partial").exists())

    def test_sync_failure_keeps_partial_only(self):
        with patch("looking_glass.bert_gpu_evidence.os.fsync", side_effect=OSError("sync failed")):
            with self.assertRaisesRegex(OSError, "sync failed"):
                write_json_new(self.destination, {"x": 1})
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.partial.read_bytes(), b'{"x":1}')

    def test_link_failure_and_existing_partial_preserve_evidence(self):
        with patch("looking_glass.bert_gpu_evidence.os.link", side_effect=OSError("link failed")):
            with self.assertRaisesRegex(OSError, "link failed"):
                write_json_new(self.destination, {"x": 1})
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.partial.read_bytes(), b'{"x":1}')
        with self.assertRaises(FileExistsError):
            write_json_new(self.destination, {"x": 2})
        self.assertEqual(self.partial.read_bytes(), b'{"x":1}')

    def test_racing_final_creation_never_overwrites_existing_file(self):
        original = os.link

        def competing_link(source, target):
            Path(target).write_bytes(b"other owner")
            original(source, target)

        with patch("looking_glass.bert_gpu_evidence.os.link", side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                write_json_new(self.destination, {"x": 1})
        self.assertEqual(self.destination.read_bytes(), b"other owner")
        self.assertEqual(self.partial.read_bytes(), b'{"x":1}')

    def test_link_completion_followed_by_cancellation_is_not_retried(self):
        original = os.link

        def link_then_interrupt(source, target):
            original(source, target)
            raise KeyboardInterrupt("uncertain acknowledgement")

        with patch("looking_glass.bert_gpu_evidence.os.link", side_effect=link_then_interrupt):
            with self.assertRaisesRegex(KeyboardInterrupt, "uncertain acknowledgement"):
                write_json_new(self.destination, {"x": 1})
        self.assertEqual(self.destination.read_bytes(), b'{"x":1}')
        self.assertEqual(self.partial.read_bytes(), self.destination.read_bytes())
        with self.assertRaises(FileExistsError):
            write_json_new(self.destination, {"x": 2})

    def test_cleanup_error_after_link_keeps_final_authoritative(self):
        with patch("looking_glass.bert_gpu_evidence.os.unlink", side_effect=OSError("cleanup")):
            write_json_new(self.destination, {"x": 1})
        self.assertEqual(self.destination.read_bytes(), b'{"x":1}')
        self.assertEqual(self.partial.read_bytes(), self.destination.read_bytes())

    def test_cancellation_during_cleanup_propagates(self):
        with patch("looking_glass.bert_gpu_evidence.os.unlink", side_effect=KeyboardInterrupt("cancel")):
            with self.assertRaisesRegex(KeyboardInterrupt, "cancel"):
                write_json_new(self.destination, {"x": 1})
        self.assertEqual(self.destination.read_bytes(), b'{"x":1}')
        self.assertEqual(self.partial.read_bytes(), self.destination.read_bytes())

    def test_oversize_or_nonfinite_is_rejected_before_file_creation(self):
        with self.assertRaises(ValueError):
            write_json_new(self.destination, {"x": "long"}, max_bytes=3)
        with self.assertRaises(ValueError):
            write_json_new(self.destination, {"x": float("nan")})
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.partial.exists())

    def test_original_write_error_survives_close_error(self):
        close = os.close

        def close_then_fail(fd):
            close(fd)
            raise OSError("injected close failure")

        with patch("looking_glass.bert_gpu_evidence.os.write", side_effect=OSError("first failure")), \
             patch("looking_glass.bert_gpu_evidence.os.close", side_effect=close_then_fail):
            with self.assertRaisesRegex(OSError, "first failure"):
                write_json_new(self.destination, {"x": 1})
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.partial.exists())

    def test_real_process_death_before_and_after_link(self):
        program = r'''
import os, sys
from looking_glass import bert_gpu_evidence as evidence
target, phase = sys.argv[1:]
if phase == 'write':
    original = evidence.os.write
    def die(fd, data):
        original(fd, data[:3])
        os._exit(73)
    evidence.os.write = die
else:
    original = evidence.os.link
    def die(source, destination):
        original(source, destination)
        os._exit(74)
    evidence.os.link = die
evidence.write_json_new(target, {'some': 'payload'})
'''
        for phase, expected_exit in (("write", 73), ("link", 74)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "result.json"
                partial = destination.with_name(destination.name + ".partial")
                child = subprocess.run([sys.executable, "-B", "-c", program,
                                        str(destination), phase], timeout=15, capture_output=True,
                                       cwd=Path(__file__).resolve().parents[1])
                self.assertEqual(child.returncode, expected_exit, child.stderr)
                if phase == "write":
                    self.assertFalse(destination.exists())
                    self.assertEqual(partial.read_bytes(), b'{"s')
                else:
                    self.assertEqual(destination.read_bytes(), b'{"some":"payload"}')
                    self.assertEqual(partial.read_bytes(), destination.read_bytes())

    def test_competing_processes_publish_exactly_one_complete_result(self):
        program = r'''
import sys
from looking_glass.bert_gpu_evidence import write_json_new
try:
    write_json_new(sys.argv[1], {'owner': int(sys.argv[2])})
except FileExistsError:
    sys.exit(7)
'''
        children = [subprocess.Popen([sys.executable, "-B", "-c", program,
                                     str(self.destination), str(number)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=Path(__file__).resolve().parents[1])
                    for number in range(6)]
        completed = []
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=15)
                completed.append((child.returncode, stdout, stderr))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate()
        self.assertEqual(sorted(code for code, _, _ in completed), [0, 7, 7, 7, 7, 7], completed)
        published = json.loads(self.destination.read_bytes())
        self.assertIn(published, [{'owner': number} for number in range(6)])


if __name__ == "__main__":
    unittest.main()
