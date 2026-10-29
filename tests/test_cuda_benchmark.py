import json
import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import zipfile

from looking_glass import cuda_benchmark as bench

try:
    import torch
    from torch import nn
    from looking_glass.probes import TapPoint, TraceSession
except ImportError:
    torch = None


class ScheduleTests(unittest.TestCase):
    def test_preflight_rejects_missing_location_replaced_by_second_invocation(self):
        identities = []
        for layer in range(6):
            for kind in ("block", "attn", "mlp"):
                path = f"transformer.h.{layer}" + ("" if kind == "block" else f".{kind}")
                identities.append((path, kind, layer, 0, 0, (), (0,), (0, 3, 7)))
        bench.validate_transformer_coverage(identities)
        identities[-1] = (*identities[0][:4], 1, *identities[0][5:])
        with self.assertRaisesRegex(AssertionError, "coverage"):
            bench.validate_transformer_coverage(identities)

    def test_preflight_rejects_wrong_saved_shape_dtype_or_device(self):
        layouts = [{"shape": [1, 3, 768], "dtype": "torch.float32", "device": "cuda:0"}
                   for _ in range(18)]
        bench.validate_transformer_layouts(layouts)
        for field, value in (("shape", [3, 768]), ("dtype", "torch.int32"), ("device", "cpu")):
            modified = [dict(row) for row in layouts]
            modified[0][field] = value
            with self.assertRaisesRegex(AssertionError, "layout"):
                bench.validate_transformer_layouts(modified)

    def test_model_admission_keeps_reserve_and_accounts_for_load_headroom(self):
        gib = 1024 ** 3
        mib = 1024 ** 2
        accepted = bench.model_admission(free_bytes=7 * gib, total_bytes=8 * gib,
                                         model_bytes=350 * mib, trace_limit=4 * mib)
        self.assertGreaterEqual(accepted["reserve_bytes"], 1 * gib)
        self.assertGreater(accepted["estimated_increment_bytes"], 350 * mib)
        with self.assertRaisesRegex(RuntimeError, "memory"):
            bench.model_admission(free_bytes=2 * gib, total_bytes=8 * gib,
                                  model_bytes=350 * mib, trace_limit=4 * mib)

    def test_each_round_has_all_modes_in_reproducible_shuffled_order(self):
        schedule = bench.make_schedule(seed=23, rounds=8)
        self.assertEqual(schedule, bench.make_schedule(seed=23, rounds=8))
        self.assertNotEqual(schedule, bench.make_schedule(seed=24, rounds=8))
        for number in range(8):
            batch = [row for row in schedule if row["round"] == number]
            self.assertEqual([row["position"] for row in batch], [0, 1, 2, 3])
            self.assertCountEqual([row["mode"] for row in batch],
                                  ["none", "id", "direct", "gradient"])
        with self.assertRaises(ValueError):
            bench.make_schedule(seed=1, rounds=0)

    def test_summary_is_derived_from_raw_wall_and_event_samples(self):
        rows = [
            {"mode": "direct", "wall_ns": 1000, "cuda_ms": 1.0},
            {"mode": "none", "wall_ns": 500, "cuda_ms": 0.2},
            {"mode": "direct", "wall_ns": 3000, "cuda_ms": 3.0},
            {"mode": "none", "wall_ns": 1500, "cuda_ms": 0.4},
        ]
        summary = {row["mode"]: row for row in bench.summarize(rows)}
        self.assertEqual(summary["direct"]["median_wall_us"], 2.0)
        self.assertEqual(summary["direct"]["median_cuda_ms"], 2.0)
        self.assertEqual(summary["none"]["median_wall_us"], 1.0)
        self.assertEqual(summary["none"]["samples"], 2)
        with self.assertRaises(ValueError):
            bench.summarize([])

    def test_ratios_use_matched_seed_and_round_not_aggregate_medians(self):
        rows = []
        for rnd, samples in enumerate(((10, 20, 30, 60), (100, 200, 100, 100))):
            rows.extend({"seed": 17, "round": rnd, "mode": mode, "wall_ns": wall}
                        for mode, wall in zip(("none", "id", "direct", "gradient"), samples))
        ratios = bench.summarize_paired_ratios(rows)
        self.assertEqual(ratios["median_id_over_none"], 2.0)
        self.assertEqual(ratios["median_direct_over_none"], 2.0)
        self.assertEqual(ratios["median_gradient_over_direct"], 1.5)
        with self.assertRaises(ValueError):
            bench.summarize_paired_ratios(rows[:-1])

    def test_trial_runner_warms_each_mode_and_keeps_interleaved_raw_rows(self):
        calls = []
        def fake_measure(invoke, factory, mode, device, cuda, start, end):
            calls.append(mode)
            return {"wall_ns": len(calls), "cuda_ms": 0.1}
        progress = {"raw": []}
        with patch.object(bench, "measure_sample", side_effect=fake_measure):
            bench.collect_trials(None, None, 0, None, None, None,
                                 seed=17, warmup_rounds=2, measured_rounds=3, progress=progress)
        self.assertEqual(len(calls), 20)
        self.assertEqual(len(progress["raw"]), 12)
        self.assertCountEqual(calls[:4], ["none", "id", "direct", "gradient"])
        self.assertCountEqual(calls[4:8], ["none", "id", "direct", "gradient"])
        self.assertEqual({row["seed"] for row in progress["raw"]}, {17})
        self.assertEqual({row["round"] for row in progress["raw"]}, {0, 1, 2})
        self.assertEqual(progress["raw"][0]["wall_ns"], 9)

    def test_lease_loss_before_a_trial_aborts_and_preserves_finished_rows(self):
        class Lease:
            def __init__(self): self.calls = 0
            def check(self):
                self.calls += 1
                if self.calls == 6:
                    raise RuntimeError("lease replaced")
        class Cuda:
            def mem_get_info(self, device): return (7 * 1024 ** 3, 8 * 1024 ** 3)
        lease = Lease()
        guard = bench.runtime_guard(lease, Cuda(), 0, 1024 ** 3, 4 * 1024 ** 2)
        progress = {"raw": []}
        with patch.object(bench, "measure_sample", return_value={"wall_ns": 1, "cuda_ms": 0.1}):
            with self.assertRaisesRegex(RuntimeError, "lease replaced"):
                bench.collect_trials(None, None, 0, None, None, None, seed=17,
                    warmup_rounds=1, measured_rounds=2, progress=progress, guard=guard)
        self.assertEqual(len(progress["raw"]), 1)

    def test_runtime_memory_pressure_aborts_without_starting_a_trial(self):
        class Lease:
            def check(self): pass
        class Cuda:
            def mem_get_info(self, device): return (1024 ** 3, 8 * 1024 ** 3)
        guard = bench.runtime_guard(Lease(), Cuda(), 0, 1024 ** 3, 4 * 1024 ** 2)
        with self.assertRaisesRegex(RuntimeError, "memory"):
            guard()


class TelemetryTests(unittest.TestCase):
    def test_identity_uses_the_shared_uuid_derived_lock_before_cuda(self):
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        result = SimpleNamespace(returncode=0, stderr="", stdout=f"{uuid}, One GPU, 8192, 616.92\n")
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": uuid}), \
                 patch.object(bench, "_command", return_value=result):
                identity, inventory = bench.resolve_benchmark_identity(Path(root), "nvidia-smi")
            self.assertEqual(identity.gpu_uuid, uuid)
            self.assertEqual(identity.lock_path.name, uuid + ".lock")
            self.assertEqual(identity.marker_path.name, uuid + ".unsafe.json")
            self.assertEqual(inventory["uuid"], uuid)

    def test_parse_sample_retains_gpu_identity_and_units(self):
        row = bench.parse_telemetry_line(
            "2026/10/28 11:00:00.123, GPU-abc, 8192, 1200, 6992, 4", 1234)
        self.assertEqual(row["gpu_uuid"], "GPU-abc")
        self.assertEqual(row["observed_monotonic_ns"], 1234)
        self.assertEqual(row["used_bytes"], 1200 * 1024 * 1024)
        self.assertEqual(row["utilization_percent"], 4)
        with self.assertRaises(ValueError):
            bench.parse_telemetry_line("bad, data", 1234)

    def test_isolation_screen_rejects_missing_data_and_foreign_processes(self):
        observations = [{"observed_monotonic_ns": n * 1_000_000_000,
                         "utilization_percent": 0, "gpu_uuid": "GPU-abc"}
                        for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
        processes = [{"ok": True, "pids": [44]}, {"ok": True, "pids": [44]}]
        screened = bench.classify_isolation(True, observations, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])
        self.assertEqual(screened["status"], "no_contamination_observed")
        foreign = [{"ok": True, "pids": [44, 55]}, processes[1]]
        self.assertEqual(bench.classify_isolation(True, observations, 4_000_000_000,
            7_000_000_000, foreign, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])["status"], "contaminated")
        self.assertEqual(bench.classify_isolation(True, observations[:4], 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])["status"], "unverified")
        self.assertEqual(bench.classify_isolation(False, observations, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])["status"], "unverified")
        sparse = observations[:3] + observations[-3:]
        self.assertEqual(bench.classify_isolation(True, sparse, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])["status"], "unverified")
        self.assertEqual(bench.classify_isolation(True, observations, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=["query failed"])["status"], "unverified")
        self.assertEqual(bench.classify_isolation(True, observations, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-other", telemetry_errors=[])["status"], "unverified")
        observations[-1]["utilization_percent"] = 8
        self.assertEqual(bench.classify_isolation(True, observations, 4_000_000_000,
            7_000_000_000, processes, own_pid=44, gpu_uuid="GPU-abc", telemetry_errors=[])["status"], "contaminated")

    def test_inventory_refuses_ambiguous_mapping_and_processes_are_scoped_to_gpu(self):
        gpu = SimpleNamespace(returncode=0, stderr="", stdout="GPU-a, One GPU, 8192, 616.92\n")
        with patch.object(bench, "_command", return_value=gpu):
            self.assertEqual(bench._gpu_inventory("nvidia-smi", 1)["uuid"], "GPU-a")
            with self.assertRaisesRegex(RuntimeError, "one CUDA-visible"):
                bench._gpu_inventory("nvidia-smi", 2)
        gpu.stdout += "GPU-b, Second GPU, 8192, 616.92\n"
        with patch.object(bench, "_command", return_value=gpu):
            with self.assertRaisesRegex(RuntimeError, "unambiguously"):
                bench._gpu_inventory("nvidia-smi", 1)
        processes = SimpleNamespace(returncode=0, stderr="",
            stdout="GPU-a, 44, python.exe\nGPU-b, 55, other.exe\n")
        with patch.object(bench, "_command", return_value=processes):
            result = bench._query_compute_processes("nvidia-smi", "GPU-a", "before")
        self.assertTrue(result["ok"])
        self.assertEqual(result["pids"], [44])


class ArtifactTests(unittest.TestCase):
    def test_supervisor_only_publishes_success_after_clean_reaped_owner(self):
        def evaluate(*, progress, **kwargs):
            progress["record_manifest"]({"model": "fixture"})
            progress["record_sample"]({"mode": "id", "wall_ns": 123})
            return {"raw": progress["raw"], "protocol": bench.PROTOCOL}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            owner = SimpleNamespace(run_id="fixture-owner")
            with patch.object(bench, "_evaluate", side_effect=evaluate):
                bench.run_benchmark(output, quick=True, candidate_only=True, owner_session=owner)
            self.assertTrue((output / "candidate.json").is_file())
            self.assertFalse((output / "results.json").exists())
            unsafe = SimpleNamespace(clean=False, exit_code=0, run_id="fixture-owner", pid=os.getpid(),
                                     timed_out=False, reason="missing attestation")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                bench.publish_supervised_outcome(output, unsafe)
            self.assertFalse((output / "results.json").exists())
            wrong_owner = SimpleNamespace(clean=True, exit_code=0, run_id="different", pid=os.getpid(),
                                          timed_out=False, reason="clean exit")
            with self.assertRaisesRegex(ValueError, "owner"):
                bench.publish_supervised_outcome(output, wrong_owner)
            wrong_pid = SimpleNamespace(clean=True, exit_code=0, run_id="fixture-owner", pid=os.getpid() + 1,
                                        timed_out=False, reason="clean exit")
            with self.assertRaisesRegex(ValueError, "owner"):
                bench.publish_supervised_outcome(output, wrong_pid)
            clean = SimpleNamespace(clean=True, exit_code=0, run_id="fixture-owner", pid=os.getpid(),
                                    timed_out=False, reason="clean exit")
            result = bench.publish_supervised_outcome(output, clean)
            self.assertEqual(result["status"], "succeeded")
            self.assertTrue(bench.verify_artifact_inventory(output))

    def test_cleanly_drained_failed_experiment_remains_failed(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            owner = SimpleNamespace(run_id="fixture-owner")
            with patch.object(bench, "_evaluate", side_effect=RuntimeError("experiment failed")):
                with self.assertRaisesRegex(RuntimeError, "experiment failed"):
                    bench.run_benchmark(output, quick=True, candidate_only=True, owner_session=owner)
            self.assertTrue((output / "candidate_failure.json").is_file())
            self.assertFalse((output / "failure.json").exists())
            clean = SimpleNamespace(clean=True, exit_code=1, run_id="fixture-owner", pid=os.getpid(),
                                    timed_out=False, reason="clean exit")
            result = bench.publish_supervised_outcome(output, clean)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(bench.verify_artifact_inventory(output))
            self.assertFalse((output / "results.json").exists())
    def test_failed_evaluation_records_failure_without_success_result(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=RuntimeError("deliberate")):
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertEqual(failure["message"], "deliberate")
            self.assertTrue(failure["quick"])
            self.assertFalse((output / "results.json").exists())

    def test_existing_output_is_not_overwritten_or_evaluated(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "existing"
            output.mkdir()
            (output / "keep.txt").write_text("preserve")
            with patch.object(bench, "_evaluate") as evaluate:
                with self.assertRaises(FileExistsError):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            evaluate.assert_not_called()
            self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_failed_serialization_never_leaves_a_success_summary(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", return_value={"raw": [], "bad": object()}):
                with self.assertRaises(TypeError):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            self.assertTrue((output / "failure.json").is_file())
            self.assertFalse((output / "results.json").exists())

    def test_terminal_cleanup_failure_cannot_publish_both_outcomes(self):
        real_unlink = Path.unlink
        failed = False
        def fail_once(path, *args, **kwargs):
            nonlocal failed
            if path.name == "results.json.partial" and not failed:
                failed = True
                raise OSError("temporary cleanup failed")
            return real_unlink(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(Path, "unlink", fail_once), \
                 patch.object(bench, "_evaluate", return_value={"raw": [], "protocol": bench.PROTOCOL}):
                with self.assertRaisesRegex(OSError, "temporary cleanup failed"):
                    bench.run_benchmark(output, quick=True)
            self.assertFalse((output / "results.json").exists())
            self.assertTrue((output / "failure.json").is_file())

    def test_failed_terminal_rollback_is_explicitly_ambiguous(self):
        real_unlink = Path.unlink
        failed = False
        def fail_rollback(path, *args, **kwargs):
            nonlocal failed
            if path.name == "results.json.partial" and not failed:
                failed = True
                raise OSError("temporary cleanup failed")
            if path.name == "results.json":
                raise OSError("rollback failed")
            return real_unlink(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(Path, "unlink", fail_rollback), \
                 patch.object(bench, "_evaluate", return_value={"raw": [], "protocol": bench.PROTOCOL}):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    bench.run_benchmark(output, quick=True)
            self.assertTrue((output / "publication_ambiguous.json").is_file())
            self.assertFalse((output / "failure.json").exists())
            with self.assertRaisesRegex(ValueError, "artifact"):
                bench.verify_artifact_inventory(output)

    def test_final_lease_check_prevents_success_after_late_ownership_loss(self):
        def evaluate(*, progress, **kwargs):
            def lost(): raise RuntimeError("lease replaced late")
            progress["final_guard"] = lost
            return {"raw": [], "protocol": bench.PROTOCOL}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, "lease replaced late"):
                    bench.run_benchmark(output, quick=True)
            self.assertFalse((output / "results.json").exists())
            self.assertTrue((output / "failure.json").is_file())

    def test_journal_close_failure_never_publishes_success_or_masks_original(self):
        class CloseFailure:
            def __init__(self, wrapped): self.wrapped = wrapped
            def __getattr__(self, name): return getattr(self.wrapped, name)
            def close(self):
                self.wrapped.close()
                raise OSError("journal close failed")
        real_open = Path.open
        def intercept(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            return CloseFailure(handle) if path.name == "raw_progress.jsonl" and args and args[0] == "x" else handle
        with tempfile.TemporaryDirectory() as root:
            success_attempt = Path(root) / "success-attempt"
            with patch.object(Path, "open", intercept), \
                 patch.object(bench, "_evaluate", return_value={"raw": [], "protocol": bench.PROTOCOL}):
                with self.assertRaisesRegex(OSError, "journal close failed"):
                    bench.run_benchmark(success_attempt, quick=True)
            self.assertFalse((success_attempt / "results.json").exists())
            self.assertEqual(json.loads((success_attempt / "failure.json").read_text())["status"], "failed")
            failed_attempt = Path(root) / "failure-attempt"
            with patch.object(Path, "open", intercept), \
                 patch.object(bench, "_evaluate", side_effect=RuntimeError("original failure")):
                with self.assertRaisesRegex(RuntimeError, "original failure"):
                    bench.run_benchmark(failed_attempt, quick=True)
            failure = json.loads((failed_attempt / "failure.json").read_text())
            self.assertEqual(failure["message"], "original failure")
            self.assertEqual(failure["artifact_errors"][0]["artifact"], "raw_progress.jsonl.close")

    def test_atomic_writers_never_replace_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "saved.json"
            target.write_text("original")
            with self.assertRaises(FileExistsError):
                bench._write_json(target, {"new": True})
            self.assertEqual(target.read_text(), "original")
            with self.assertRaises(FileExistsError):
                bench._write_bytes(target, b"replacement")
            self.assertEqual(target.read_text(), "original")

    def test_partial_samples_survive_an_evaluation_failure(self):
        def fail(*, quick, exclusive, progress, cache_dir, lease_path, lease_timeout):
            progress["raw"].append({"mode": "id", "wall_ns": 123})
            raise RuntimeError("interrupted")
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=fail):
                with self.assertRaises(RuntimeError):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            self.assertEqual(json.loads((output / "failure.json").read_text())["partial_samples"], 1)
            self.assertIn("id,123", (output / "partial_raw_timing.csv").read_text())

    def test_bundle_write_failure_does_not_prevent_partial_sample_evidence(self):
        def fail(*, quick, exclusive, progress, cache_dir, lease_path, lease_timeout):
            progress["raw"].append({"mode": "id", "wall_ns": 123})
            progress["source_bundle"] = b"bundle"
            raise RuntimeError("evaluation failed")
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            actual_write = bench._write_bytes
            def fail_bundle(path, value):
                if path.name == "source_bundle.zip":
                    raise OSError("disk error")
                return actual_write(path, value)
            with patch.object(bench, "_evaluate", side_effect=fail), \
                 patch.object(bench, "_write_bytes", side_effect=fail_bundle):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            self.assertIn("id,123", (output / "partial_raw_timing.csv").read_text())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["artifact_errors"][0]["artifact"], "source_bundle.zip")

    def test_keyboard_interrupt_is_recorded_as_cancelled(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    bench.run_benchmark(output, quick=True, exclusive=False)
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "cancelled")

    def test_preload_identity_is_durable_when_model_transfer_fails(self):
        class Model:
            def to(self, **kwargs): raise RuntimeError("CUDA out of memory")
        stage = {"device": "GPU-a", "free_bytes": 7000,
                 "weights_sha256": "verified", "seed": 17}
        def evaluate(*, progress, **kwargs):
            bench.stage_model_to_cuda(Model(), "cuda:0", stage, progress)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    bench.run_benchmark(output, quick=True)
            self.assertEqual(json.loads((output / "preload.json").read_text()), stage)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["metadata"]["preload"], stage)
            self.assertFalse((output / "results.json").exists())

    def test_hard_process_exit_leaves_started_marker_and_durable_completed_sample(self):
        script = """\
import os, sys
from looking_glass import cuda_benchmark as bench
def stop(**kwargs):
    kwargs['progress']['record_manifest']({'seed': 17, 'model_sha256': 'verified'})
    kwargs['progress']['record_sample']({'mode': 'id', 'wall_ns': 123})
    os._exit(19)
bench._evaluate = stop
bench.run_benchmark(sys.argv[1], quick=True)
"""
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            child = subprocess.run([sys.executable, "-c", script, str(output)],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(child.returncode, 19, child.stderr)
            self.assertTrue((output / "started.json").is_file())
            self.assertEqual(json.loads((output / "manifest.json").read_text()),
                             {"seed": 17, "model_sha256": "verified"})
            row = json.loads((output / "raw_progress.jsonl").read_text().strip())
            self.assertEqual(row, {"mode": "id", "wall_ns": 123})
            self.assertFalse((output / "results.json").exists())
            self.assertFalse((output / "failure.json").exists())

    def test_terminal_inventory_detects_modified_or_missing_evidence(self):
        def evaluate(*, progress, **kwargs):
            progress["record_manifest"]({"model": "fixture"})
            progress["record_sample"]({"mode": "id", "wall_ns": 123})
            return {"raw": progress["raw"], "protocol": bench.PROTOCOL}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=evaluate):
                result = bench.run_benchmark(output, quick=True)
            self.assertEqual(result["status"], "succeeded")
            self.assertIn("manifest.json", result["artifact_inventory"])
            self.assertTrue(bench.verify_artifact_inventory(output))
            (output / "raw_timing.csv").write_text("altered")
            with self.assertRaisesRegex(ValueError, "artifact"):
                bench.verify_artifact_inventory(output)

    def test_inventory_verifier_rejects_incoherent_status_and_noninteger_sizes(self):
        def evaluate(*, progress, **kwargs):
            progress["record_manifest"]({"model": "fixture"})
            return {"raw": [], "protocol": bench.PROTOCOL}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_evaluate", side_effect=evaluate):
                bench.run_benchmark(output, quick=True)
            path = output / "results.json"
            original = json.loads(path.read_text())
            altered = dict(original, status="failed")
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "status"):
                bench.verify_artifact_inventory(output)
            original["artifact_inventory"]["manifest.json"]["bytes"] *= 1.0
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "artifact"):
                bench.verify_artifact_inventory(output)

    def test_source_archive_contains_exact_replayable_repository_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            project = Path(root)
            (project / "looking_glass").mkdir()
            (project / "docs").mkdir()
            (project / "looking_glass" / "a.py").write_bytes(b"x = 3\r\n")
            (project / "docs" / "CUDA_BENCHMARK_V1.md").write_bytes(b"protocol\n")
            (project / "pyproject.toml").write_bytes(b"[project]\n")
            archive, hashes = bench.create_source_bundle(project)
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                self.assertEqual(bundle.read("looking_glass/a.py"), b"x = 3\r\n")
                self.assertEqual(bundle.read("docs/CUDA_BENCHMARK_V1.md"), b"protocol\n")
            self.assertEqual(hashes["looking_glass/a.py"], hashlib.sha256(b"x = 3\r\n").hexdigest())
            self.assertEqual(archive, bench.create_source_bundle(project)[0])


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class EquivalenceTests(unittest.TestCase):
    def test_unsupervised_execution_stops_before_cuda_or_device_lease(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            with patch.object(bench, "_acquire_device_lease") as acquire, \
                 patch.object(torch.cuda, "is_available") as cuda_available:
                with self.assertRaisesRegex(RuntimeError, "supervised"):
                    bench.run_benchmark(output, quick=True, cache_dir=root,
                                        lease_path=Path(root) / "device.lock")
            acquire.assert_not_called()
            cuda_available.assert_not_called()

    def test_cuda_unavailable_is_recorded_as_failure_with_source_archive(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            owner = SimpleNamespace(run_id="fixture-owner", check=lambda lease: None)
            with patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(bench, "_acquire_device_lease") as acquire, \
                 patch.object(bench, "_validate_owner_session"):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    bench.run_benchmark(output, quick=True, exclusive=False,
                                        cache_dir=Path(root) / "cache",
                                        lease_path=Path(root) / "device.lock", candidate_only=True,
                                        owner_session=owner)
            acquire.assert_called_once()
            self.assertTrue((output / "candidate_failure.json").is_file())
            self.assertTrue((output / "source_bundle.zip").is_file())
            self.assertFalse((output / "results.json").exists())

    def test_real_taps_preserve_output_and_match_values_at_same_locations(self):
        model = nn.Sequential(TapPoint("before"), nn.Linear(2, 2), TapPoint("after"))
        model.eval()
        x = torch.tensor([[0.5, -0.25]])
        factory = lambda mode: TraceSession(model, mode=mode)
        report = bench.verify_equivalence(
            lambda: model(x), factory, lambda e: (e.node_id, e.occurrence))
        self.assertEqual(report["events_per_mode"], 2)
        self.assertEqual(report["value_tensor_bytes"], 16)
        self.assertTrue(report["outputs_exact"])
        self.assertTrue(report["values_exact"])

    def test_different_captured_values_fail_the_preflight(self):
        class WrongValueSession(TraceSession):
            def _capture(self, module, inputs, output):
                result = super()._capture(module, inputs, output)
                if self.mode == "gradient":
                    self.events[-1].value = self.events[-1].value + 1
                return result

        model = nn.Sequential(TapPoint("point"))
        with self.assertRaisesRegex(AssertionError, "values"):
            bench.verify_equivalence(lambda: model(torch.ones(1)),
                lambda mode: WrongValueSession(model, mode=mode),
                lambda e: (e.node_id, e.occurrence))

    def test_equal_numeric_values_with_wrong_dtype_fail_the_preflight(self):
        class WrongDtypeSession(TraceSession):
            def _capture(self, module, inputs, output):
                result = super()._capture(module, inputs, output)
                if self.mode == "gradient":
                    self.events[-1].value = self.events[-1].value.to(torch.int32)
                return result
        model = nn.Sequential(TapPoint("point"))
        self.assertTrue(torch.equal(torch.tensor([1.0]), torch.tensor([1], dtype=torch.int32)))
        with self.assertRaisesRegex(AssertionError, "dtype"):
            bench.verify_equivalence(lambda: model(torch.ones(1)),
                lambda mode: WrongDtypeSession(model, mode=mode),
                lambda e: (e.node_id, e.occurrence))

    def test_source_metadata_mismatch_fails_even_when_values_match(self):
        class WrongMetadataSession(TraceSession):
            def _capture(self, module, inputs, output):
                result = super()._capture(module, inputs, output)
                if self.mode == "id":
                    self.events[-1].dtype = "torch.int32"
                return result
        model = nn.Sequential(TapPoint("point"))
        with self.assertRaisesRegex(AssertionError, "source metadata"):
            bench.verify_equivalence(lambda: model(torch.ones(1)),
                lambda mode: WrongMetadataSession(model, mode=mode),
                lambda e: (e.node_id, e.occurrence))

    def test_timed_capture_synchronizes_before_and_after_retaining_values(self):
        trace_reference = []
        log = []

        class RecordingSession(TraceSession):
            def _capture(self, module, inputs, output):
                result = super()._capture(module, inputs, output)
                trace_reference.append(weakref.ref(self.events[-1].value))
                return result

        class FakeCuda:
            def synchronize(self, device):
                log.append("synchronize")
                if len(trace_reference) and len([x for x in log if x == "synchronize"]) == 2:
                    if trace_reference[0]() is None:
                        raise AssertionError("capture was released before completion synchronization")
            def reset_peak_memory_stats(self, device): pass
            def mem_get_info(self, device): return (700, 1000)
            def memory_allocated(self, device): return 100
            def memory_reserved(self, device): return 200
            def max_memory_allocated(self, device): return 150
            def max_memory_reserved(self, device): return 250

        class FakeEvent:
            def __init__(self, name): self.name = name
            def record(self): log.append(self.name)
            def elapsed_time(self, other): return 0.25

        model = nn.Sequential(TapPoint("point"))
        def invoke():
            log.append("forward")
            return model(torch.ones(1))
        result = bench.measure_sample(invoke, lambda mode: RecordingSession(model, mode=mode),
            "direct", 0, FakeCuda(), FakeEvent("start"), FakeEvent("end"))
        self.assertEqual(log, ["synchronize", "start", "forward", "end", "synchronize", "synchronize"])
        self.assertIsNone(trace_reference[0]())
        self.assertEqual(result["peak_allocated_bytes"], 150)
        self.assertEqual(result["peak_reserved_bytes"], 250)
        self.assertEqual(result["device_free_before_bytes"], 700)
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["recorded_tensor_bytes"], 4)


if __name__ == "__main__":
    unittest.main()
