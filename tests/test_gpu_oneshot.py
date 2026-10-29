"""CPU fakes for a narrowly scoped supervised GPU pilot; no CUDA calls."""
import contextlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

GIB = 1024 ** 3
UUID = "GPU-12345678-1234-5678-9abc-1234567890ab"


class FakeUuid:
    def __str__(self):
        return UUID[4:]


class FakeCuda:
    def __init__(self, *, free=6 * GIB, oom=False, workspace_bytes=0):
        self.total, self.free = 8 * GIB, free
        self.allocated = self.reserved = self.peak_allocated = self.peak_reserved = 0
        self.oom = oom
        self.syncs = 0
        self.fraction = None
        self.workspace_bytes, self.workspace_live = workspace_bytes, 0
    def is_initialized(self): return False
    def device_count(self): return 1
    def get_device_properties(self, index):
        return SimpleNamespace(uuid=FakeUuid(), name="Synthetic GPU", total_memory=self.total, major=7, minor=5)
    def mem_get_info(self, index): return self.free - self.reserved, self.total
    def memory_allocated(self, index): return self.allocated
    def memory_reserved(self, index): return self.reserved
    def max_memory_allocated(self, index): return self.peak_allocated
    def max_memory_reserved(self, index): return self.peak_reserved
    def synchronize(self, index): self.syncs += 1
    def set_per_process_memory_fraction(self, fraction, index): self.fraction = fraction
    def get_per_process_memory_fraction(self, index): return int(self.fraction * self.total) / self.total
    def reset_peak_memory_stats(self, index):
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved
    def manual_seed(self, seed): pass
    def empty_cache(self):
        if self.allocated == 0: self.reserved = 0
    def allocate(self, size):
        if self.oom: raise MemoryError("simulated out of memory")
        self.allocated += size
        self.reserved = max(self.reserved, self.allocated)
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
    def clear_workspaces(self):
        self.allocated -= self.workspace_live
        self.workspace_live = 0


class FakeTensor:
    def __init__(self, cuda, shape, *, gpu=False, dtype="float32"):
        self.cuda, self.shape, self.gpu, self.dtype = cuda, shape, gpu, dtype
        self.device = SimpleNamespace(type="cuda" if gpu else "cpu", index=0 if gpu else None)
        self.nbytes = (402056 if shape == (1, 2, 50257) else 16) if gpu else 0
        if gpu: cuda.allocate(self.nbytes)
    def to(self, device, **kwargs):
        return FakeTensor(self.cuda, self.shape, gpu=str(device).startswith("cuda"), dtype=self.dtype)
    def detach(self): return self
    def view(self, dtype): return self
    def contiguous(self): return self
    def numpy(self): return SimpleNamespace(tobytes=lambda: b"synthetic-output")
    def numel(self): return 100514 if self.shape == (1, 2, 50257) else 2
    def element_size(self): return 4 if self.dtype == "float32" else 8
    def __del__(self):
        if self.nbytes: self.cuda.allocated -= self.nbytes


class FakeModel:
    def __init__(self, cuda):
        self.cuda, self.nbytes = cuda, 0
        self.training = False
        self.config = SimpleNamespace(use_cache=False)
    def to(self, device):
        size = 400 * 1024 ** 2
        self.cuda.allocate(size)
        self.nbytes = size
        return self
    def __call__(self, **kwargs):
        if self.cuda.workspace_bytes:
            self.cuda.allocate(self.cuda.workspace_bytes)
            self.cuda.workspace_live = self.cuda.workspace_bytes
        return SimpleNamespace(logits=FakeTensor(self.cuda, (1, 2, 50257), gpu=True))
    def parameters(self):
        return [SimpleNamespace(device=SimpleNamespace(type="cuda", index=0), dtype="float32",
                                numel=lambda: 81912576)]
    def buffers(self): return []
    def __del__(self):
        if self.nbytes: self.cuda.allocated -= self.nbytes


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda
        self._C = SimpleNamespace(_CUuuid=FakeUuid, _cuda_clearCublasWorkspaces=cuda.clear_workspaces)
        self.float32, self.long, self.int32 = "float32", "int64", "int32"
        self.__version__ = "2.8.0+cu128"
        self.version = SimpleNamespace(cuda="fake")
        self.random = SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None))
    def tensor(self, value, *, dtype, device): return FakeTensor(self.cuda, (1, 2), dtype="int64")
    def inference_mode(self): return contextlib.nullcontext()
    def isfinite(self, value): return SimpleNamespace(all=lambda: True)


class OneShotTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("looking_glass.gpu_oneshot")
        self.session = object.__new__(self.module.OwnerSession)
        self.lease = object.__new__(self.module.DeviceLease)
        self.check = patch.object(self.module.OwnerSession, "check", autospec=True)
        self.binding = patch.object(self.module.OwnerSession, "verify_runtime_binding", autospec=True)
        self.check.start(); self.binding.start()
        self.addCleanup(self.check.stop); self.addCleanup(self.binding.stop)

    def run_fake(self, *, free=6 * GIB, oom=False):
        cuda = FakeCuda(free=free, oom=oom)
        torch = FakeTorch(cuda)
        def load():
            return SimpleNamespace(model=FakeModel(cuda), metadata={"parameter_count": 81912576})
        result = self.module._measure_once(self.session, self.lease, torch, load)
        return result, cuda

    def test_short_forward_reports_synchronized_memory_and_cleanup(self):
        result, cuda = self.run_fake()
        self.assertEqual(result["input_ids"], [15496, 995])
        self.assertEqual(result["output_shape"], [1, 2, 50257])
        self.assertEqual(len(result["output_sha256"]), 64)
        self.assertEqual(result["allocator_fraction"], 0.25)
        self.assertEqual(result["runtime_uuid_verified"], True)
        self.assertGreater(result["snapshots"]["after_model_copy"]["allocated_bytes"], 0)
        self.assertGreaterEqual(result["snapshots"]["after_forward"]["peak_allocated_bytes"],
                                result["snapshots"]["after_model_copy"]["allocated_bytes"])
        self.assertEqual(result["snapshots"]["after_cleanup"]["allocated_bytes"], 0)
        self.assertEqual(result["snapshots"]["after_cleanup"]["reserved_bytes"], 0)
        self.assertGreaterEqual(cuda.syncs, 4)

    def test_low_free_memory_refuses_model_copy(self):
        with self.assertRaises(RuntimeError): self.run_fake(free=2 * GIB)

    def test_out_of_memory_does_not_return_a_successful_measurement(self):
        with self.assertRaises(MemoryError): self.run_fake(oom=True)

    def test_uuid_parser_refuses_strings_and_incomplete_values(self):
        torch = FakeTorch(FakeCuda())
        self.assertEqual(self.module._runtime_uuid(torch, FakeUuid()), UUID)
        with self.assertRaises(ValueError): self.module._runtime_uuid(torch, UUID)

    def test_invalid_memory_counter_is_rejected(self):
        cuda = FakeCuda()
        cuda.peak_allocated, cuda.peak_reserved = 2, 1
        with self.assertRaises(RuntimeError):
            self.module._snapshot(self.session, self.lease, FakeTorch(cuda))

    def test_workspace_cleanup_and_stage_records_are_separate(self):
        cuda = FakeCuda(workspace_bytes=4 * 1024 ** 2)
        events = {}
        result = self.module._measure_once(
            self.session, self.lease, FakeTorch(cuda),
            lambda: SimpleNamespace(model=FakeModel(cuda), metadata={"parameter_count": 81912576}),
            record_stage=lambda stage, value: events.__setitem__(stage, value))
        self.assertEqual(events["cleanup_before_workspace_clear"]["allocated_bytes"], 4 * 1024 ** 2)
        self.assertEqual(events["after_cleanup"]["allocated_bytes"], 0)
        self.assertEqual(events["after_cleanup"]["reserved_bytes"], 0)
        self.assertEqual(events["cleanup_weakrefs"], {"model": False, "input_gpu": False, "logits": False})
        self.assertEqual(events["output"]["output_sha256"], result["output_sha256"])

    def test_stages_survive_workspace_cleanup_failure(self):
        cuda = FakeCuda(workspace_bytes=4 * 1024 ** 2)
        torch = FakeTorch(cuda)
        torch._C._cuda_clearCublasWorkspaces = lambda: (_ for _ in ()).throw(RuntimeError("fake failure"))
        events = {}
        with self.assertRaises(RuntimeError):
            self.module._measure_once(
                self.session, self.lease, torch,
                lambda: SimpleNamespace(model=FakeModel(cuda), metadata={"parameter_count": 81912576}),
                record_stage=lambda stage, value: events.__setitem__(stage, value))
        self.assertIn("after_forward", events)
        self.assertIn("cleanup_before_workspace_clear", events)
        self.assertNotIn("after_cleanup", events)


class SupervisedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("looking_glass.gpu_oneshot")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("work", "cache", "site"):
            (self.root / name).mkdir()
        self.out = self.root / "result.json"
        self.before = {"timestamp_utc": "2026-01-01T00:00:00+00:00", "free_bytes": 1}
        self.after = {"timestamp_utc": "2026-01-01T00:00:01+00:00", "free_bytes": 1}

    def run_parent(self, run_supervised, snapshots):
        def fake_git(command, **kwargs):
            return SimpleNamespace(stdout=("a" * 40 if command[1] == "rev-parse" else ""))
        with (patch.object(self.module.subprocess, "run", side_effect=fake_git),
              patch.object(self.module, "_verify_controller_import_mode"),
              patch.object(self.module, "_verify_committed_sources", return_value={"runner": "digest"}),
              patch.object(self.module, "_verify_offline_cache"),
              patch.object(self.module, "_smi_snapshot", side_effect=snapshots),
              patch.object(self.module, "resolve_device_identity", return_value=SimpleNamespace(lock_path="lock", marker_path="marker")),
              patch.object(self.module, "run_supervised", side_effect=run_supervised),
              patch.object(self.module, "_owner_presence_after_exit", return_value={"query": "ok", "owner_pid_listed": False})):
            return self.module.run_one_shot(workspace_root=self.root, cache_dir=self.root / "cache",
                                            output=self.out, site_packages=self.root / "site", smi="smi")

    def test_post_telemetry_failure_retains_clean_owner_candidate(self):
        def supervise(command, **kwargs):
            self.assertFalse(kwargs["inherit_environment"])
            run_id, run_dir = command[-1], Path(command[-3])
            candidate = {"schema": "gpu-oneshot-owner-v2", "run_id": run_id,
                         "supervisor_run_id": "b" * 32, "pid": 123, "measurement": {"finite": True}}
            (run_dir / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
            return SimpleNamespace(run_id="b" * 32, pid=123, clean=True, exit_code=0, timed_out=False, reason="clean")
        result = self.run_parent(supervise, [(UUID, self.before), OSError("telemetry unavailable")])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["post_telemetry_error_type"], "OSError")
        self.assertEqual(result["measurement"], {"finite": True})
        self.assertTrue(result["supervisor"]["clean"])
        self.assertEqual(json.loads(self.out.read_text())["status"], "incomplete")

    def test_supervisor_exception_reports_unknown_owner_state(self):
        def supervise(command, **kwargs):
            raise RuntimeError("unknown owner state")
        result = self.run_parent(supervise, [(UUID, self.before), (UUID, self.after)])
        self.assertEqual(result["status"], "quarantined")
        self.assertIsNone(result["supervisor"])
        self.assertEqual(result["supervisor_error_type"], "RuntimeError")
        self.assertEqual(json.loads(self.out.read_text())["status"], "quarantined")

    def test_mismatched_candidate_cannot_be_published_as_success(self):
        def supervise(command, **kwargs):
            run_dir = Path(command[-3])
            (run_dir / "candidate.json").write_text(json.dumps({"schema": "gpu-oneshot-owner-v2"}))
            return SimpleNamespace(run_id="b" * 32, pid=123, clean=True, exit_code=0, timed_out=False, reason="clean")
        result = self.run_parent(supervise, [(UUID, self.before), (UUID, self.after)])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["candidate_error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
