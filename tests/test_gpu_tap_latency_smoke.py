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
