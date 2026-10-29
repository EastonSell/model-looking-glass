import unittest
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass import gpu_tap_latency_smoke as smoke
from looking_glass import gpu_tap_latency_owner as owner

try:
    import torch
except ImportError:
    torch = None


class FakeCuda:
    def __init__(self):
        self.stream = object()
        self.peak = 450 * 1024 ** 2
    def synchronize(self, device): pass
    def reset_peak_memory_stats(self, device): pass
    def current_stream(self, device): return self.stream
    def default_stream(self, device): return self.stream
    def mem_get_info(self, device): return (6 * 1024 ** 3, 8 * 1024 ** 3)
    def memory_allocated(self, device): return 320 * 1024 ** 2
    def memory_reserved(self, device): return 400 * 1024 ** 2
    def max_memory_allocated(self, device): return 350 * 1024 ** 2
    def max_memory_reserved(self, device): return self.peak
    def Event(self, enable_timing):
        return SimpleNamespace(record=lambda stream: None, elapsed_time=lambda other: 0.1)


def fake_sample(wall_ns=100_000, peak_reserved_bytes=450 * 1024 ** 2):
    return {"wall_ns": wall_ns, "device_total_bytes": 8 * 1024 ** 3,
        "peak_reserved_bytes": peak_reserved_bytes,
        "device_free_before_bytes": 6 * 1024 ** 3, "device_free_after_bytes": 6 * 1024 ** 3,
        "allocated_before_bytes": 320 * 1024 ** 2, "allocated_after_bytes": 320 * 1024 ** 2,
        "reserved_before_bytes": 400 * 1024 ** 2, "reserved_after_bytes": 400 * 1024 ** 2}


class ScheduleAndAdmissionTests(unittest.TestCase):
    def test_deterministic_shuffled_rounds_contain_all_modes(self):
        rows = smoke.schedule(17, 6)
        self.assertEqual(rows, smoke.schedule(17, 6))
        for number in range(6):
            self.assertCountEqual([row["mode"] for row in rows if row["round"] == number], smoke.MODES)
        with self.assertRaises(ValueError):
            smoke.schedule(17, 0)

    def test_live_reserve_and_allocator_headroom_fail_closed(self):
        cuda = FakeCuda()
        cap = 2 * 1024 ** 3
        self.assertEqual(smoke.memory_guard(cuda, "cpu", cap)["allocator_cap_bytes"], cap)
        with patch.object(cuda, "memory_reserved", return_value=cap - smoke.HEADROOM + 1):
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                smoke.memory_guard(cuda, "cpu", cap)
        with patch.object(cuda, "mem_get_info", return_value=(1024 ** 3, 8 * 1024 ** 3)):
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                smoke.memory_guard(cuda, "cpu", cap)

    def test_warmups_are_retained_separately_and_pairs_use_measured_rounds(self):
        cuda = FakeCuda()
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        evidence, guards = [], []
        def sample(invoke, factory, mode, device, cuda, start, end, validate=None, memory_check=None):
            return fake_sample({"none": 100_000, "direct": 200_000, "gradient": 400_000}[mode])
        with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
             patch.object(smoke.bench, "measure_sample", side_effect=sample):
            result = smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda,
                device="cpu", cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                check_lease=lambda: guards.append(1), record=evidence.append)
        self.assertEqual(result["warmup_samples"], 6)
        self.assertEqual(result["measured_samples"], 18)
        self.assertEqual(result["median_paired_gradient_over_direct"], 2.0)
        self.assertEqual(result["median_paired_direct_over_none"], 2.0)
        self.assertEqual(len(evidence), 50)
        self.assertEqual(evidence[0]["stage"], "preflight")
        self.assertEqual(evidence[0]["postcheck_status"], "pending")
        self.assertEqual(evidence[0]["memory_before"]["device_total_bytes"], 8 * 1024 ** 3)
        for index in range(0, len(evidence), 2):
            observation, postcheck = evidence[index:index + 2]
            self.assertEqual(observation["postcheck_status"], "pending")
            self.assertEqual(postcheck["stage"], "postcheck")
            self.assertEqual(postcheck["status"], "passed")
            self.assertEqual(observation["observation_id"], postcheck["observation_id"])
        self.assertEqual([row["stage"] for row in evidence[2::2]], ["warmup"] * 6 + ["measured"] * 18)
        self.assertGreaterEqual(len(guards), 2 * 24 + 2)

    def test_calibration_peak_is_saved_then_aborts_before_trials(self):
        cuda = FakeCuda()
        cuda.peak = 2 * 1024 ** 3 - smoke.HEADROOM + 1
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        evidence = []
        with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
             patch.object(smoke.bench, "measure_sample") as measured:
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda, device="cpu",
                    cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                    check_lease=lambda: None, record=evidence.append)
        self.assertEqual([row["stage"] for row in evidence], ["preflight"])
        self.assertEqual(evidence[0]["postcheck_status"], "pending")
        measured.assert_not_called()

    def test_preflight_total_change_and_low_peak_abort_with_pending_evidence(self):
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        for failure in ("total_change", "low_initial_peak"):
            with self.subTest(failure=failure):
                cuda, evidence = FakeCuda(), []
                if failure == "total_change":
                    cuda.mem_get_info = unittest.mock.Mock(side_effect=[
                        (6 * 1024 ** 3, 8 * 1024 ** 3), (6 * 1024 ** 3, 9 * 1024 ** 3)])
                else:
                    cuda.max_memory_allocated = lambda device: 319 * 1024 ** 2
                with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
                     patch.object(smoke.bench, "measure_sample") as measured:
                    with self.assertRaisesRegex(RuntimeError, "preflight memory"):
                        smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda,
                            device="cpu", cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                            check_lease=lambda: None,
                            record=evidence.append)
                self.assertEqual(len(evidence), 1)
                self.assertEqual(evidence[0]["postcheck_status"], "pending")
                measured.assert_not_called()

    def test_failed_sample_postcheck_cannot_be_accepted_as_a_validated_row(self):
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        for failure in ("peak", "lease", "total_change"):
            with self.subTest(failure=failure):
                cuda, evidence, lease_calls = FakeCuda(), [], []
                def check_lease():
                    lease_calls.append(1)
                    if failure == "lease" and len(lease_calls) == 4:
                        raise RuntimeError("lease lost")
                sample = fake_sample(peak_reserved_bytes=(2 * 1024 ** 3 - smoke.HEADROOM + 1
                                     if failure == "peak" else 450 * 1024 ** 2))
                if failure == "total_change":
                    cuda.mem_get_info = unittest.mock.Mock(side_effect=[
                        (6 * 1024 ** 3, 8 * 1024 ** 3), (6 * 1024 ** 3, 8 * 1024 ** 3),
                        (6 * 1024 ** 3, 8 * 1024 ** 3), (6 * 1024 ** 3, 9 * 1024 ** 3)])
                with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
                     patch.object(smoke.bench, "measure_sample", return_value=sample):
                    with self.assertRaises(RuntimeError):
                        smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda,
                            device="cpu", cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                            check_lease=check_lease,
                            record=evidence.append)
                self.assertEqual([row["stage"] for row in evidence], ["preflight", "postcheck", "warmup"])
                self.assertEqual(evidence[-1]["postcheck_status"], "pending")

    def test_device_total_cannot_change_between_preflight_and_first_sample(self):
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        cuda, evidence = FakeCuda(), []
        cuda.mem_get_info = unittest.mock.Mock(side_effect=[
            (6 * 1024 ** 3, 8 * 1024 ** 3), (6 * 1024 ** 3, 8 * 1024 ** 3),
            (6 * 1024 ** 3, 9 * 1024 ** 3)])
        with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
             patch.object(smoke.bench, "measure_sample") as measured:
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda,
                    device="cpu", cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                    check_lease=lambda: None,
                    record=evidence.append)
        self.assertEqual([row["stage"] for row in evidence], ["preflight", "postcheck"])
        measured.assert_not_called()

    def test_sample_inner_total_change_is_rejected_before_forward(self):
        cuda, evidence, invoked = FakeCuda(), [], []
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True,
                  "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        def sample(*args, memory_check=None, **kwargs):
            memory_check(6 * 1024 ** 3, 9 * 1024 ** 3, 320 * 1024 ** 2, 400 * 1024 ** 2)
            invoked.append(True)
            return fake_sample()
        with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
             patch.object(smoke.bench, "measure_sample", side_effect=sample):
            with self.assertRaisesRegex(RuntimeError, "changed before the timed forward"):
                smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda, device="cpu",
                    cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                    check_lease=lambda: None, record=evidence.append)
        self.assertFalse(invoked)
        self.assertEqual([row["stage"] for row in evidence], ["preflight", "postcheck"])

    def test_low_internal_after_snapshot_leaves_a_pending_sample(self):
        cuda, evidence = FakeCuda(), []
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True,
                  "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        sample = fake_sample()
        sample["device_free_after_bytes"] = 0
        with patch.object(smoke.bench, "verify_equivalence", return_value=checks), \
             patch.object(smoke.bench, "measure_sample", return_value=sample):
            with self.assertRaises(smoke.MemoryPressureError) as raised:
                smoke.run_samples(invoke=None, factory=None, identity=None, cuda=cuda, device="cpu",
                    cap_bytes=2 * 1024 ** 3, expected_total_bytes=8 * 1024 ** 3,
                    check_lease=lambda: None, record=evidence.append)
        self.assertEqual(raised.exception.snapshot["device_free_bytes"], 0)
        self.assertEqual(evidence[-1]["postcheck_status"], "pending")
        self.assertEqual(evidence[-1]["stage"], "warmup")


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class ValidationTests(unittest.TestCase):
    def test_equivalence_completion_runs_before_live_outputs_and_traces_are_released(self):
        refs, sessions = [], []
        class Session:
            def __init__(self, mode):
                self.mode = mode
                value = None if mode == "id" else torch.ones((1, 1, 2))
                event = SimpleNamespace(shape=(1, 2, 2), dtype="torch.float32", device="cpu", value=value)
                self.events, self.tensor_bytes = [event], 0 if value is None else 8
                sessions.append(weakref.ref(self))
            def __enter__(self): return self
            def __exit__(self, *args): return False
        def invoke():
            value = torch.ones((1, 2, 3))
            refs.append(weakref.ref(value))
            return value
        boundaries = []
        def completion():
            boundaries.append((all(reference() is not None for reference in refs),
                               sessions[1]() is not None, sessions[2]() is not None))
        smoke.bench.verify_equivalence(invoke, Session, lambda event: (0,), completion=completion)
        gc.collect()
        boundaries.append((any(reference() is not None for reference in refs),
                           sessions[1]() is not None, sessions[2]() is not None))
        self.assertEqual(boundaries, [(True, True, True), (False, False, False)])

    def test_frozen_tap_and_nonfinite_output_checks(self):
        output = torch.zeros((1, 2, 50257), dtype=torch.float32)
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        identity = lambda event: expected
        smoke.validate_sample("none", output, None, identity=identity, expected=expected, device="cpu")
        for mode in ("direct", "gradient"):
            value = torch.zeros((1, 1, 768), dtype=torch.float32)
            event = SimpleNamespace(sequence=0, run_id=smoke.PROTOCOL, capture_mode=mode,
                shape=(1, 2, 768), dtype="torch.float32", device="cpu", value=value, tensor_bytes=3072)
            session = SimpleNamespace(successful=True, events=[event], tensor_bytes=3072)
            smoke.validate_sample(mode, output, session, identity=identity, expected=expected, device="cpu")
            event.value = torch.zeros((1, 2, 768))
            with self.assertRaisesRegex(AssertionError, "layout"):
                smoke.validate_sample(mode, output, session, identity=identity, expected=expected, device="cpu")
        output[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            smoke.validate_sample("none", output, None, identity=identity, expected=expected, device="cpu")

    def test_real_timing_helper_and_journal_reconstruction(self):
        cuda = FakeCuda()
        expected = ("transformer.h.0", "block", 0, 0, 0, (0,), (0,), (1,))
        checks = {"events_per_mode": 1, "value_tensor_bytes": 3072, "outputs_finite": True, "outputs_exact": True, "values_exact": True, "comparison": {"operator": "torch.equal"},
                  "event_identities": [expected], "value_layouts": [
                      {"shape": [1, 1, 768], "dtype": "torch.float32", "device": "cpu"}]}
        class Session:
            def __init__(self, mode):
                event = SimpleNamespace(sequence=0, run_id=smoke.PROTOCOL, capture_mode=mode,
                    shape=(1, 2, 768), dtype="torch.float32", device="cpu",
                    value=torch.zeros((1, 1, 768)), tensor_bytes=3072)
                self.events, self.tensor_bytes, self.successful = [event], 3072, True
            def __enter__(self): return self
            def __exit__(self, *args): return False
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "observations.jsonl"
            with journal.open("xb") as handle, patch.object(smoke.bench, "verify_equivalence", return_value=checks):
                result = smoke.run_samples(invoke=lambda: torch.zeros((1, 2, 50257)), factory=Session,
                    identity=lambda event: expected, cuda=cuda, device="cpu", cap_bytes=2 * 1024 ** 3,
                    expected_total_bytes=8 * 1024 ** 3, check_lease=lambda: None,
                    record=lambda row: owner._record_line(handle, row))
            self.assertEqual(owner.verify_observations(journal, result, expected_total_bytes=8 * 1024 ** 3,
                                                       expected_cap_bytes=2 * 1024 ** 3, device="cpu"),
                             {"preflight": 1, "warmups": 6, "measured": 18})
            rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
            original = json.dumps(rows)
            mutations = (
                lambda data: data[3].update(status="failed"),
                lambda data: data[0]["checks"].update(outputs_exact=False),
                lambda data: data[0]["checks"].update(events_per_mode=99),
                lambda data: data[1]["memory_after"].update(device_free_bytes=0),
                lambda data: data[2].update(output_shape=[9]),
                lambda data: data[2].update(events=99),
            )
            for mutate in mutations:
                data = json.loads(original)
                mutate(data)
                journal.write_text("\n".join(json.dumps(row) for row in data) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    owner.verify_observations(journal, result, expected_total_bytes=8 * 1024 ** 3,
                                              expected_cap_bytes=2 * 1024 ** 3, device="cpu")


class EvidenceBoundaryTests(unittest.TestCase):
    def test_exploratory_contamination_requires_fully_available_screen(self):
        ticks = [0.0, 0.3, 0.6, 1.1, 1.5, 2.1, 2.3, 3.1, 3.3, 3.5]
        rows = [{"gpu_uuid": "GPU-test", "total_bytes": 100, "used_bytes": 1, "free_bytes": 99,
                 "utilization_percent": 0, "query_started_monotonic_ns": int(t * 1e9),
                 "query_ended_monotonic_ns": int(t * 1e9), "observed_monotonic_ns": int(t * 1e9)} for t in ticks]
        queries = [{"ok": True, "pids": [42]}, {"ok": True, "pids": [42]}]
        def classify(observations, errors):
            return smoke.bench.classify_isolation(True, observations, 1_000_000_000, 2_000_000_000,
                                                  queries, 11, "GPU-test", errors)
        screen = classify(rows, [])
        self.assertEqual(screen["status"], "contaminated")
        self.assertEqual(screen["availability"], "complete")
        self.assertEqual(owner._activity_interpretation(screen, True), "exploratory_contaminated_activity")
        with self.assertRaises(RuntimeError):
            owner._activity_interpretation(screen, False)
        high_utilization = [dict(row) for row in rows]
        high_utilization[0]["utilization_percent"] = 6
        no_foreign = [{"ok": True, "pids": []}, {"ok": True, "pids": []}]
        high_screen = smoke.bench.classify_isolation(True, high_utilization, 1_000_000_000, 2_000_000_000,
                                                     no_foreign, 11, "GPU-test", [])
        self.assertEqual(high_screen["contamination_reasons"], ["idle utilization exceeded five percent"])
        self.assertEqual(owner._activity_interpretation(high_screen, True), "exploratory_contaminated_activity")
        wrong_gpu = [dict(row) for row in rows]
        wrong_gpu[3]["gpu_uuid"] = "GPU-other"
        gaps = rows[:3] + rows[-3:]
        for invalid in (classify(rows, [{"error_type": "query"}]), classify(wrong_gpu, []), classify(gaps, [])):
            self.assertEqual(invalid["status"], "contaminated")
            self.assertEqual(invalid["availability"], "unverified")
            with self.assertRaisesRegex(RuntimeError, "unverified"):
                owner._activity_interpretation(invalid, True)

    def test_replacement_environment_probe_saves_private_failure_and_aborts(self):
        result = SimpleNamespace(returncode=255, stdout=r"C:\Users\Private\NVML failed", stderr="")
        with tempfile.TemporaryDirectory() as directory, patch.object(owner.subprocess, "run", return_value=result):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "telemetry is unavailable"):
                owner._probe_telemetry_environment("unused", {}, output)
            self.assertIn("Private", (output / "telemetry_preflight.private.json").read_text())
            public = (output / "telemetry_preflight.json").read_text()
            self.assertNotIn("Private", public)
            self.assertIn("255", public)

    def test_initial_monitor_failure_prevents_all_samples(self):
        monitor = SimpleNamespace(observations=[], errors=[{"error_type": "RuntimeError"}])
        with tempfile.TemporaryDirectory() as directory, patch.object(smoke, "run_samples") as samples:
            summary, issue = owner._measure_if_monitored(Path(directory), monitor,
                {"ok": False, "pids": []}, "GPU-test")
            self.assertIsNone(summary)
            self.assertIn("telemetry", issue)
            self.assertFalse((Path(directory) / "observations.jsonl").exists())
            samples.assert_not_called()

    def test_initial_monitor_accepts_foreign_contexts_but_rejects_invalid_data(self):
        rows = [{"gpu_uuid": "GPU-test", "total_bytes": 100, "used_bytes": 1, "free_bytes": 99,
                 "utilization_percent": 0, "query_started_monotonic_ns": n * 100,
                 "query_ended_monotonic_ns": n * 100 + 1, "observed_monotonic_ns": n * 100 + 1}
                for n in range(3)]
        query = {"phase": "before", "ok": True, "pids": [123], "other_gpu_uuids": [],
                 "processes": [{"gpu_uuid": "GPU-test", "pid": 123, "name": "python.exe"}],
                 "observed_monotonic_ns": 100, "query_started_monotonic_ns": 100,
                 "query_ended_monotonic_ns": 150}
        self.assertIsNone(owner._initial_monitor_issue(rows, [], query, "GPU-test"))
        bad_rows = [dict(row) for row in rows]
        bad_rows[-1]["gpu_uuid"] = "GPU-other"
        self.assertIn("identity", owner._initial_monitor_issue(bad_rows, [], query, "GPU-test"))
        self.assertIn("unavailable", owner._initial_monitor_issue(rows, [], {"ok": False}, "GPU-test"))
        earlier = dict(rows[0], total_bytes=101, free_bytes=100,
                       query_started_monotonic_ns=0, query_ended_monotonic_ns=0, observed_monotonic_ns=0)
        self.assertIn("inconsistent", owner._initial_monitor_issue([earlier, *rows], [], query, "GPU-test"))
        self.assertIn("stale", owner._initial_monitor_issue(rows, [], query, "GPU-test", 2_000_000_201))
        self.assertIn("future", owner._initial_monitor_issue(rows, [], query, "GPU-test", 200))
        stale_query = dict(query, query_started_monotonic_ns=0, observed_monotonic_ns=0,
                           query_ended_monotonic_ns=100)
        fresh_rows = [dict(row, query_started_monotonic_ns=2_000_000_000 + n * 100,
                           query_ended_monotonic_ns=2_000_000_001 + n * 100,
                           observed_monotonic_ns=2_000_000_001 + n * 100) for n, row in enumerate(rows)]
        self.assertIn("stale", owner._initial_monitor_issue(fresh_rows, [], stale_query, "GPU-test", 2_000_000_201))
        future_query = dict(query, query_ended_monotonic_ns=301)
        self.assertIn("invalid timestamps", owner._initial_monitor_issue(rows, [], future_query, "GPU-test", 300))
        foreign_device = dict(query, other_gpu_uuids=["GPU-other"])
        self.assertIn("unavailable", owner._initial_monitor_issue(rows, [], foreign_device, "GPU-test", 300))
        mismatched_pid = dict(query, processes=[{"gpu_uuid": "GPU-test", "pid": 999, "name": "other.exe"}])
        self.assertIn("unavailable", owner._initial_monitor_issue(rows, [], mismatched_pid, "GPU-test", 300))
        mismatched_gpu = dict(query, processes=[{"gpu_uuid": "GPU-other", "pid": 123, "name": "python.exe"}])
        self.assertIn("unavailable", owner._initial_monitor_issue(rows, [], mismatched_gpu, "GPU-test", 300))
        invalid_name = dict(query, processes=[{"gpu_uuid": "GPU-test", "pid": 123, "name": " "}])
        self.assertIn("unavailable", owner._initial_monitor_issue(rows, [], invalid_name, "GPU-test", 300))

    def test_compute_process_query_validates_all_rows_and_records_bounds(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        result = SimpleNamespace(returncode=0, stderr="", stdout=f"{uuid}, 7, python.exe\n")
        with patch.object(smoke.bench, "_command", return_value=result), \
                patch.object(smoke.bench.time, "perf_counter_ns", side_effect=[10, 20]):
            query = smoke.bench._query_compute_processes("nvidia-smi", uuid, "before")
        self.assertTrue(query["ok"])
        self.assertEqual((query["query_started_monotonic_ns"], query["query_ended_monotonic_ns"]), (10, 20))
        self.assertEqual(query["pids"], [7])
        for row in ("N/A,not-a-pid,process", "N/A,7,process", f"{uuid},0,process", f"{uuid},7, "):
            result.stdout = row + "\n"
            with self.subTest(row=row), patch.object(smoke.bench, "_command", return_value=result):
                self.assertFalse(smoke.bench._query_compute_processes("nvidia-smi", uuid, "before")["ok"])

    def test_public_projections_do_not_expose_process_paths_or_query_output(self):
        report = {"classification": {"status": "contaminated", "foreign_compute_pids": [42]},
                  "process_queries": [{"phase": "before", "ok": True, "observed_monotonic_ns": 1,
                      "pids": [42], "processes": [{"pid": 42, "name": r"C:\Users\Private\app.exe"}]}],
                  "workload_started_monotonic_ns": 2, "workload_ended_monotonic_ns": 3,
                  "clock": "perf_counter_ns"}
        projected = json.dumps(owner._screen_projection(report))
        self.assertNotIn("Private", projected)
        self.assertNotIn("42", projected)
        telemetry = {"configured_interval_seconds": 0.2, "observations": [],
                     "errors": [{"returncode": 255, "stdout": r"C:\Users\Private\err", "error_type": "RuntimeError"}],
                     "limitations": []}
        projected = json.dumps(owner._telemetry_projection(telemetry))
        self.assertNotIn("Private", projected)
        self.assertIn("255", projected)

    def test_telemetry_failure_retains_bounded_stdout_and_returncode(self):
        monitor = smoke.bench._NvidiaMonitor("unused", interval_seconds=0)
        def command(*args, **kwargs):
            monitor._stop.set()
            return SimpleNamespace(returncode=255, stdout="X" * 5000, stderr="")
        with patch.object(smoke.bench, "_command", side_effect=command):
            monitor._poll()
            process = smoke.bench._query_compute_processes("unused", "GPU-test", "before")
        self.assertEqual(monitor.errors[0]["returncode"], 255)
        self.assertEqual(len(monitor.errors[0]["stdout"]), 4096)
        self.assertEqual(process["returncode"], 255)
        self.assertEqual(len(process["stdout"]), 4096)

    @unittest.skipUnless(torch is not None, "PyTorch is unavailable")
    def test_source_first_entry_requires_absolute_fresh_cache_prefix(self):
        project = Path(__file__).resolve().parents[1]
        script = project / "looking_glass" / "gpu_tap_latency_entry.py"
        site = Path(torch.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "never-created"
            common = [sys.executable, "-I", "-B", "-S", "-X"]
            args = [str(script), "--site-packages", str(site), "--help"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
            result = subprocess.run(common + ["pycache_prefix=" + str(prefix), *args],
                                    cwd=project, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(prefix.exists())
            result = subprocess.run(common + ["pycache_prefix=relative-cache", *args],
                                    cwd=project, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)

    def test_write_all_handles_short_writes_and_rejects_zero_progress(self):
        class ShortWriter:
            def __init__(self, limit): self.limit, self.contents = limit, bytearray()
            def write(self, data):
                count = min(len(data), self.limit)
                self.contents.extend(data[:count])
                return count
        output = ShortWriter(2)
        smoke.bench._write_all(output, b"abcdef")
        self.assertEqual(bytes(output.contents), b"abcdef")
        with self.assertRaisesRegex(OSError, "progress"):
            smoke.bench._write_all(ShortWriter(0), b"abc")

    def test_start_failure_is_recorded_and_missing_window_stops_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(owner, "_require_isolated_controller"):
            output = Path(directory) / "missing-window"
            with patch.object(owner, "_offline_assets") as assets:
                with self.assertRaisesRegex(RuntimeError, "coordinated timing window"):
                    owner.run_controller(output=output, workspace_root=directory, cache_dir=directory,
                                         smi="unused", exclusive=False)
            assets.assert_not_called()
            self.assertEqual(json.loads((output / "failure.json").read_text())["stage"], "parent_start")
            output = Path(directory) / "start-failure"
            original = smoke.bench._write_json
            def write(path, value):
                if path.name == "parent_started.json":
                    raise OSError("start failed")
                return original(path, value)
            with patch.object(smoke.bench, "_write_json", side_effect=write):
                with self.assertRaisesRegex(OSError, "start failed"):
                    owner.run_controller(output=output, workspace_root=directory, cache_dir=directory,
                                         smi="unused", exclusive=True)
            self.assertEqual(json.loads((output / "failure.json").read_text())["stage"], "parent_start")


class OwnerCustodyTests(unittest.TestCase):
    def tearDown(self):
        owner._CUSTODY[:] = [None] * len(owner._CUSTODY)
        owner._OWNER_USED = False
        gc.collect()

    def test_failed_owner_keeps_exception_frame_and_storage_rooted_after_return(self):
        from looking_glass.device_lease import DeviceLease
        from looking_glass.supervisor import OwnerSession
        class Payload: pass
        refs = []
        def failing_body(*args):
            payload = Payload()
            refs.append(weakref.ref(payload))
            raise RuntimeError("forward failed")
        session = SimpleNamespace(device_lease_path="unused", run_id="test", check=lambda lease: None)
        lease = object()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(OwnerSession, "from_environment", return_value=session), \
             patch.object(DeviceLease, "acquire", return_value=lease), \
             patch.object(owner, "_run_owner", side_effect=failing_body), \
             patch.object(owner, "_drain_after_failure", return_value={"synchronization": "failed"}):
            self.assertEqual(owner.owner_main(Path(directory), "unused", "unused"), 1)
            gc.collect()
            self.assertIsNotNone(refs[0]())
            self.assertIs(owner._CUSTODY[0], lease)
            self.assertIsInstance(owner._CUSTODY[5], RuntimeError)
            self.assertFalse((Path(directory) / "candidate.json").exists())
            with self.assertRaisesRegex(RuntimeError, "single-use"):
                owner.owner_main(Path(directory), "unused", "unused")
            gc.collect()
            self.assertIsNotNone(refs[0]())

    def test_lost_session_after_initialization_check_stops_before_device_query(self):
        state = {"lost": False, "device_queries": 0}
        def check(_):
            if state["lost"]:
                raise RuntimeError("marker lost")
        def initialized():
            state["lost"] = True
            return False
        def device_count():
            state["device_queries"] += 1
            return 1
        session = SimpleNamespace(check=check)
        torch_fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=initialized, device_count=device_count))
        with self.assertRaisesRegex(RuntimeError, "marker lost"):
            owner._verify_cuda_binding(session, object(), torch_fake)
        self.assertEqual(state["device_queries"], 0)


if __name__ == "__main__":
    unittest.main()
