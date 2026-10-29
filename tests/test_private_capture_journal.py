"""CPU-only tests for private capture evidence after a failed forward."""

from pathlib import Path
import gc
import hashlib
import json
import os
import struct
import sys
import tempfile
from threading import Event, Thread
import tracemalloc
import unittest
from unittest import mock
import weakref

import torch
from torch import nn
from looking_glass.model_ownership import ModelLease

try:
    from looking_glass.private_capture_journal import (
        CaptureJournalFull, JournaledActivationPlan, PrivateCaptureJournal, read_capture_journal,
    )
except ModuleNotFoundError as error:
    if error.name != "looking_glass.private_capture_journal":
        raise
    CaptureJournalFull = JournaledActivationPlan = PrivateCaptureJournal = read_capture_journal = None


class ForwardFailsAfterCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, value):
        self.tap(value)
        raise ValueError("synthetic downstream failure")


class TwoCapturesThenFail(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Identity()
        self.second = nn.Identity()

    def forward(self, value):
        self.first(value)
        self.second(value + 1)
        raise ValueError("synthetic downstream failure")


class CaptureThenReturn(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, value):
        return self.tap(value)


class StreamProxy:
    """Fault injector around the real journal file; no fake persistence."""
    def __init__(self, stream, *, limit=None, stop_after=None, fail_close=False):
        self.stream = stream
        self.limit = limit
        self.stop_after = stop_after
        self.written = 0
        self.fail_close = fail_close

    def write(self, data):
        if self.stop_after is not None and self.written >= self.stop_after:
            return 0
        limit = len(data) if self.limit is None else min(len(data), self.limit)
        if self.stop_after is not None:
            limit = min(limit, self.stop_after - self.written)
        written = self.stream.write(data[:limit])
        self.written += written
        return written

    def flush(self):
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        if self.fail_close:
            self.fail_close = False
            raise OSError("injected close failure")
        return self.stream.close()


def journal_for(path, *, max_records=3, max_journal_bytes=65536):
    return PrivateCaptureJournal(path, case_id="synthetic-case", input_sha256="1" * 64,
                                 model_sha256="2" * 64, plan_sha256="3" * 64,
                                 max_journal_bytes=max_journal_bytes, max_records=max_records,
                                 max_metadata_bytes=4096, max_workspace_bytes=131072)


def inspect(path, *, max_records=3, quiescent=True):
    identity = {"case_id": "synthetic-case", "input_sha256": "1" * 64,
                "model_sha256": "2" * 64, "plan_sha256": "3" * 64, "byte_order": sys.byteorder}
    return read_capture_journal(path, expected_identity=identity, max_journal_bytes=65536,
                                max_records=max_records, max_metadata_bytes=4096,
                                max_read_payload_bytes=65536, externally_quiescent=quiescent)


class PrivateCaptureJournalTests(unittest.TestCase):
    def test_reader_requires_expected_case_and_provenance_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            wrong = {"case_id": "different", "input_sha256": "1" * 64,
                     "model_sha256": "2" * 64, "plan_sha256": "3" * 64,
                     "byte_order": sys.byteorder}
            try:
                with self.assertRaises(ValueError):
                    read_capture_journal(path, max_journal_bytes=65536, max_records=3,
                                         max_metadata_bytes=4096, max_read_payload_bytes=65536,
                                         externally_quiescent=True,
                                         expected_identity=wrong)
            except TypeError:
                self.fail("reader must bind a trusted expected identity")

    def test_reader_limits_separate_retained_payload_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = ForwardFailsAfterCapture().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with self.assertRaises(ValueError):
                with plan:
                    model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            expected = {"case_id": "synthetic-case", "input_sha256": "1" * 64,
                        "model_sha256": "2" * 64, "plan_sha256": "3" * 64,
                        "byte_order": sys.byteorder}
            try:
                evidence = read_capture_journal(path, max_journal_bytes=65536, max_records=3,
                                                max_metadata_bytes=4096, max_read_payload_bytes=7,
                                                externally_quiescent=True, expected_identity=expected)
            except TypeError:
                self.fail("reader needs its own retained-payload limit")
            self.assertEqual(evidence["tail"], "capacity")
            self.assertEqual(evidence["captures"], [])

    def test_completed_capture_survives_failed_forward_without_publication(self):
        self.assertIsNotNone(JournaledActivationPlan, "private capture journal is not implemented")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = PrivateCaptureJournal(path, case_id="synthetic-case",
                                            input_sha256="1" * 64, model_sha256="2" * 64,
                                            plan_sha256="3" * 64, max_journal_bytes=65536,
                                            max_records=3, max_metadata_bytes=4096, max_workspace_bytes=131072)
            model = ForwardFailsAfterCapture().eval()
            plan = JournaledActivationPlan(model, [{"name": "before_failure", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with self.assertRaisesRegex(ValueError, "synthetic downstream failure"):
                with plan:
                    model(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
            with self.assertRaises(RuntimeError):
                _ = plan.captures
            journal.close()
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "clean")
            self.assertEqual(evidence["identity"]["case_id"], "synthetic-case")
            self.assertEqual(len(evidence["captures"]), 1)
            self.assertEqual(evidence["captures"][0]["metadata"]["name"], "before_failure")
            self.assertEqual(evidence["captures"][0]["payload"], bytes.fromhex("0000803f00000040"))

    def test_quota_refusal_invalidates_plan_and_keeps_prior_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path, max_records=1)
            model = TwoCapturesThenFail().eval()
            plan = JournaledActivationPlan(model, [
                {"name": "first", "operation": "capture", "module_path": "first"},
                {"name": "second", "operation": "capture", "module_path": "second"},
            ], journal=journal)
            with self.assertRaises(CaptureJournalFull):
                with plan:
                    model(torch.tensor([[1.0, 2.0]]))
            with self.assertRaises(RuntimeError):
                _ = plan.captures
            journal.close()
            evidence = inspect(path, max_records=1)
            self.assertEqual([entry["metadata"]["name"] for entry in evidence["captures"]], ["first"])
            self.assertEqual(evidence["captures"][0]["payload"], bytes.fromhex("0000803f00000040"))

    def test_reader_requires_external_quiescence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            with self.assertRaises(RuntimeError):
                inspect(path, quiescent=False)

    def test_truncated_second_frame_preserves_first_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = TwoCapturesThenFail().eval()
            plan = JournaledActivationPlan(model, [
                {"name": "first", "operation": "capture", "module_path": "first"},
                {"name": "second", "operation": "capture", "module_path": "second"},
            ], journal=journal)
            with self.assertRaises(ValueError):
                with plan:
                    model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            with path.open("r+b") as stream:
                stream.truncate(path.stat().st_size - 2)
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "incomplete")
            self.assertEqual([entry["metadata"]["name"] for entry in evidence["captures"]], ["first"])

    def test_oversized_tail_does_not_allocate_advertised_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            with path.open("ab") as stream:
                stream.write(struct.pack("<IQ", 2, 2**64 - 1))
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual(evidence["captures"], [])

    def test_nondictionary_metadata_is_an_invalid_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            prefix = struct.pack("<IQ", 2, 0)
            metadata = b"[]"
            digest = hashlib.sha256(prefix + metadata).digest()
            with path.open("ab") as stream:
                stream.write(prefix + metadata + digest + b"END!")
            try:
                evidence = inspect(path)
            except Exception as error:
                self.fail(f"malformed tail must be reported, not crash: {type(error).__name__}")
            self.assertEqual(evidence["tail"], "invalid")

    def test_deeply_nested_metadata_is_an_invalid_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            metadata = b"[" * 1200 + b"]" * 1200
            prefix = struct.pack("<IQ", len(metadata), 0)
            with path.open("ab") as stream:
                stream.write(prefix + metadata + hashlib.sha256(prefix + metadata).digest() + b"END!")
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual(evidence["captures"], [])

    def test_failed_fsync_invalidates_plan_even_if_frame_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = ForwardFailsAfterCapture().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with mock.patch("looking_glass.private_capture_journal.os.fsync", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
            with self.assertRaises(RuntimeError):
                _ = plan.captures
            journal.close()
            evidence = inspect(path)
            self.assertEqual(evidence["durability"], "not-established-by-file")

    def test_capture_workspace_limit_is_checked_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            try:
                journal = PrivateCaptureJournal(path, case_id="synthetic-case", input_sha256="1" * 64,
                                                model_sha256="2" * 64, plan_sha256="3" * 64,
                                                max_journal_bytes=65536, max_records=3,
                                                max_metadata_bytes=4096, max_workspace_bytes=15)
            except TypeError:
                self.fail("capture workspace needs its own independent byte limit")
            model = ForwardFailsAfterCapture().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with mock.patch("looking_glass.private_capture_journal._cpu_bytes", side_effect=AssertionError("copied before cap")):
                with self.assertRaises(CaptureJournalFull):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            self.assertEqual(inspect(path)["captures"], [])

    def test_closing_journal_does_not_retain_the_bound_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = ForwardFailsAfterCapture().eval()
            model_ref = weakref.ref(model)
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            journal.close()
            del plan, model
            gc.collect()
            self.assertIsNone(model_ref())

    def test_short_writes_are_retried_and_preserve_snapshot_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal._stream = StreamProxy(journal._stream, limit=3)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            value = torch.tensor([[1.0, 2.0]])
            with plan:
                result = model(value)
            self.assertTrue(torch.equal(result, value))
            value.add_(10)
            journal.close()
            self.assertEqual(inspect(path)["captures"][0]["payload"], bytes.fromhex("0000803f00000040"))

    def test_capture_does_not_require_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            try:
                with mock.patch.object(torch.Tensor, "numpy", side_effect=RuntimeError("numpy unavailable")):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
            finally:
                journal.close()
            self.assertEqual(inspect(path)["captures"][0]["payload"], bytes.fromhex("0000803f00000040"))

    def test_cpu_byte_copy_preserves_noncontiguous_offsets_and_empty_tensors(self):
        from looking_glass import private_capture_journal as module
        for integer, floating, fmt in ((torch.int16, torch.float16, "=4H"),
                                       (torch.int16, torch.bfloat16, "=4H"),
                                       (torch.int32, torch.float32, "=4I"),
                                       (torch.int64, torch.float64, "=4Q")):
            with self.subTest(dtype=floating):
                base = torch.arange(20, dtype=integer).view(floating).reshape(4, 5)
                selected = base[1:4:2, 1:5:2]
                self.assertFalse(selected.is_contiguous())
                self.assertEqual(module._cpu_bytes(selected), struct.pack(fmt, 6, 8, 16, 18))
                self.assertEqual(module._cpu_bytes(base[0:0]), b"")

    def test_write_stall_invalidates_plan_and_leaves_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal._stream = StreamProxy(journal._stream, limit=2, stop_after=2)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with self.assertRaises(OSError):
                with plan:
                    model(torch.tensor([[1.0, 2.0]]))
            with self.assertRaises(RuntimeError):
                _ = plan.captures
            journal.close()
            self.assertEqual(inspect(path)["tail"], "incomplete")

    def test_impossible_write_count_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            class OverreportingStream(StreamProxy):
                def write(self, data):
                    return len(data) + 1
            journal._stream = OverreportingStream(journal._stream)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            try:
                with self.assertRaises(OSError):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
                with self.assertRaises(RuntimeError):
                    _ = plan.captures
            finally:
                journal.close()
            self.assertEqual(inspect(path)["captures"], [])

    def test_failed_close_can_be_retried_after_plan_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal._stream = StreamProxy(journal._stream, fail_close=True)
            with self.assertRaises(OSError):
                journal.close()
            journal.close()
            self.assertEqual(inspect(path)["tail"], "clean")

    def test_failed_lease_release_retains_journal_owner_until_plan_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            release = plan._lease.release
            calls = 0
            def fail_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("synthetic lease release failure")
                return release(*args, **kwargs)
            with mock.patch.object(plan._lease, "release", side_effect=fail_once):
                with self.assertRaises(OSError):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
                with self.assertRaises(RuntimeError):
                    journal.close()
                with self.assertRaises(RuntimeError):
                    plan.recover_interrupted_entry(externally_quiescent=True)
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
                plan.close()
            journal.close()
            lease = ModelLease(model).acquire()
            lease.release()

    def test_failed_borrow_release_retains_journal_and_parent_until_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            parent = ModelLease(model).acquire()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}],
                                           journal=journal, _parent_lease=parent)
            release = plan._lease.release
            calls = 0
            def fail_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("synthetic child release failure")
                return release(*args, **kwargs)
            with mock.patch.object(plan._lease, "release", side_effect=fail_once):
                with self.assertRaises(OSError):
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
                with self.assertRaises(RuntimeError):
                    journal.close()
                with self.assertRaises(RuntimeError):
                    parent.release()
                plan.close()
            journal.close()
            parent.release()
            lease = ModelLease(model).acquire()
            lease.release()

    def test_failed_header_fsync_preserves_primary_error_when_close_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            real_open = open
            streams = []
            def failing_close(*args, **kwargs):
                stream = StreamProxy(real_open(*args, **kwargs), fail_close=True)
                streams.append(stream)
                return stream
            try:
                with mock.patch("looking_glass.private_capture_journal.open", side_effect=failing_close, create=True):
                    with mock.patch("looking_glass.private_capture_journal.os.fsync", side_effect=OSError("fsync failure")):
                        with self.assertRaises(OSError) as caught:
                            journal_for(path)
                self.assertEqual(str(caught.exception), "fsync failure")
                self.assertIsInstance(caught.exception.__cause__, OSError)
                self.assertEqual(str(caught.exception.__cause__), "injected close failure")
            finally:
                for stream in streams:
                    stream.stream.close()

    def test_boolean_sequence_is_rejected_as_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            journal.close()
            metadata = json.dumps({"kind": "capture", "sequence": False, "element_size": 4,
                                   "event": {"selected_shape": [1], "operation": "capture", "device": "cpu"}},
                                  sort_keys=True, separators=(",", ":")).encode()
            payload = bytes.fromhex("0000803f")
            prefix = struct.pack("<IQ", len(metadata), len(payload))
            with path.open("ab") as stream:
                stream.write(prefix + metadata + payload + hashlib.sha256(prefix + metadata + payload).digest() + b"END!")
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual(evidence["captures"], [])

    def test_capture_only_preserves_outputs_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).eval()
            original = torch.tensor([[0.25, -0.5, 0.75]], requires_grad=True)
            baseline = model(original)
            baseline.sum().backward()
            expected_input_grad = original.grad.clone()
            expected_parameter_grads = [parameter.grad.clone() for parameter in model.parameters()]
            model.zero_grad(set_to_none=True)
            journal = journal_for(Path(tmp) / "private.bin")
            plan = JournaledActivationPlan(model, [{"name": "hidden", "operation": "capture",
                                                    "module_path": "0"}], journal=journal)
            candidate = original.detach().clone().requires_grad_()
            with plan:
                observed = model(candidate)
            observed.sum().backward()
            self.assertTrue(torch.equal(baseline, observed))
            self.assertTrue(torch.equal(expected_input_grad, candidate.grad))
            self.assertTrue(all(torch.equal(expected, parameter.grad)
                                for expected, parameter in zip(expected_parameter_grads, model.parameters())))
            journal.close()

    def test_close_cannot_race_a_plan_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            paused, resume = Event(), Event()
            errors = []
            original = journal._ensure_open
            called = False
            def pause_once(owner):
                nonlocal called
                original(owner)
                if not called:
                    called = True
                    paused.set()
                    if not resume.wait(5):
                        raise RuntimeError("test entry timed out")
            def run():
                try:
                    with plan:
                        model(torch.tensor([[1.0, 2.0]]))
                except BaseException as error:
                    errors.append(error)
            with mock.patch.object(journal, "_ensure_open", side_effect=pause_once):
                worker = Thread(target=run)
                worker.start()
                self.assertTrue(paused.wait(5))
                try:
                    with self.assertRaises(RuntimeError):
                        journal.close()
                finally:
                    resume.set()
                    worker.join(5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            journal.close()
            self.assertEqual(len(inspect(path)["captures"]), 1)

    def test_non_cpu_source_is_rejected_before_tensor_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            source = torch.empty((1, 2), device="meta")
            try:
                with mock.patch("torch.isfinite", side_effect=AssertionError("source was inspected")):
                    with self.assertRaises(ValueError):
                        JournaledActivationPlan(model, [{"name": "change", "operation": "add",
                                                         "module_path": "tap", "value": source}],
                                                journal=journal)
            finally:
                journal.close()

    def test_reader_rejects_unrecognized_float_dtype_even_with_valid_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with plan:
                model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            raw = bytearray(path.read_bytes())
            offset = 8
            header_size, header_payload = struct.unpack_from("<IQ", raw, offset)
            offset += 12 + header_size + header_payload + 32 + 4
            metadata_size, payload_size = struct.unpack_from("<IQ", raw, offset)
            start, end = offset + 12, offset + 12 + metadata_size
            metadata = bytes(raw[start:end]).replace(b"torch.float32", b"torch.float99")
            self.assertEqual(len(metadata), metadata_size)
            raw[start:end] = metadata
            payload_end = end + payload_size
            raw[payload_end:payload_end + 32] = hashlib.sha256(bytes(raw[offset:payload_end])).digest()
            path.write_bytes(raw)
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual(evidence["captures"], [])

    def test_interrupted_entry_finalizer_cleans_plan_or_keeps_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            fired = False
            def trace(frame, event, arg):
                nonlocal fired
                if event == "line" and frame.f_code.co_name == "_end_entry" and not fired:
                    fired = True
                    raise KeyboardInterrupt("synthetic entry-finalizer interruption")
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    with plan:
                        self.fail("interrupted entry must not reach body")
            finally:
                sys.settrace(None)
            self.assertTrue(fired)
            lease = ModelLease(model)
            lease.acquire()
            lease.release()
            journal.close()

    def test_rejected_reentry_does_not_abort_first_active_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with plan:
                with self.assertRaises(RuntimeError):
                    plan.__enter__()
                model(torch.tensor([[1.0, 2.0]]))
            self.assertIn("capture", plan.captures)
            journal.close()

    def test_reader_rejects_event_path_outside_reported_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            with plan:
                model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            raw = bytearray(path.read_bytes())
            offset = 8
            header_size, header_payload = struct.unpack_from("<IQ", raw, offset)
            offset += 12 + header_size + header_payload + 32 + 4
            metadata_size, payload_size = struct.unpack_from("<IQ", raw, offset)
            start, end = offset + 12, offset + 12 + metadata_size
            metadata = bytes(raw[start:end]).replace(b'"module_path":"tap"', b'"module_path":"xyz"')
            self.assertNotEqual(metadata, bytes(raw[start:end]))
            raw[start:end] = metadata
            payload_end = end + payload_size
            raw[payload_end:payload_end + 32] = hashlib.sha256(bytes(raw[offset:payload_end])).digest()
            path.write_bytes(raw)
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual(evidence["captures"], [])

    def test_metadata_encoding_stops_before_allocating_unbounded_output(self):
        from looking_glass import private_capture_journal as module
        values = ["a" * 100 for _ in range(10000)]
        tracemalloc.start()
        try:
            try:
                with self.assertRaises(CaptureJournalFull):
                    module._encode_bounded(values, 1024)
            except AttributeError:
                self.fail("metadata serialization needs a preallocation bound")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 200000)

    def test_owner_can_recover_interrupted_plan_return_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            fired = False
            def trace(frame, event, arg):
                nonlocal fired
                if (event == "return" and frame.f_code is JournaledActivationPlan.__enter__.__code__
                        and not fired):
                    fired = True
                    raise KeyboardInterrupt("synthetic return interruption")
                return trace
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt):
                    with plan:
                        self.fail("interrupted return must not reach body")
            finally:
                sys.settrace(None)
            self.assertTrue(fired)
            try:
                self.assertTrue(plan.recover_interrupted_entry(externally_quiescent=True))
            except AttributeError:
                self.fail("caller needs an explicit interrupted-entry recovery path")
            lease = ModelLease(model)
            lease.acquire()
            lease.release()
            journal.close()

    def test_recovery_rejects_an_active_forward_and_foreign_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = journal_for(Path(tmp) / "private.bin")
            model = CaptureThenReturn().eval()
            plan = JournaledActivationPlan(model, [{"name": "capture", "operation": "capture",
                                                    "module_path": "tap"}], journal=journal)
            failures = []
            def foreign_recovery():
                try:
                    plan.recover_interrupted_entry(externally_quiescent=True)
                except RuntimeError as error:
                    failures.append(error)
            def forward(value):
                with self.assertRaises(RuntimeError):
                    plan.recover_interrupted_entry(externally_quiescent=True)
                return model.tap(value)
            with mock.patch.object(model, "forward", side_effect=forward):
                with plan:
                    with self.assertRaises(RuntimeError):
                        plan.recover_interrupted_entry()
                    thread = Thread(target=foreign_recovery)
                    thread.start()
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                    model(torch.tensor([[1.0, 2.0]]))
            self.assertEqual(len(failures), 1)
            self.assertIn("capture", plan.captures)
            journal.close()

    def test_reader_rejects_duplicate_capture_names_with_valid_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.bin"
            journal = journal_for(path)
            model = TwoCapturesThenFail().eval()
            plan = JournaledActivationPlan(model, [
                {"name": "alpha", "operation": "capture", "module_path": "first"},
                {"name": "bravo", "operation": "capture", "module_path": "second"},
            ], journal=journal)
            with self.assertRaises(ValueError):
                with plan:
                    model(torch.tensor([[1.0, 2.0]]))
            journal.close()
            raw = bytearray(path.read_bytes())
            offset = 8
            for _ in range(2):
                metadata_size, payload_size = struct.unpack_from("<IQ", raw, offset)
                offset += 12 + metadata_size + payload_size + 32 + 4
            metadata_size, payload_size = struct.unpack_from("<IQ", raw, offset)
            start, end = offset + 12, offset + 12 + metadata_size
            metadata = bytes(raw[start:end]).replace(b'"name":"bravo"', b'"name":"alpha"')
            self.assertNotEqual(metadata, bytes(raw[start:end]))
            raw[start:end] = metadata
            payload_end = end + payload_size
            raw[payload_end:payload_end + 32] = hashlib.sha256(bytes(raw[offset:payload_end])).digest()
            path.write_bytes(raw)
            evidence = inspect(path)
            self.assertEqual(evidence["tail"], "invalid")
            self.assertEqual([entry["metadata"]["name"] for entry in evidence["captures"]], ["alpha"])
