"""CPU-only regression probes for pair identity and failed-drain custody."""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from looking_glass import bert_gpu_exposed_pair as pair
from looking_glass.memory_budget import CudaMemorySnapshot


class PairGuards(unittest.TestCase):
    def setUp(self):
        for name in pair._OWNER_CUSTODY:
            pair._OWNER_CUSTODY[name] = False if name in ("entered", "drain_failed") else None

    def test_fixed_pair_matches_the_frozen_public_manifest(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(pair._verify_public_pair(repo), pair.MANIFEST_SHA256)
        self.assertEqual(pair.TARGET_IDS[:2], pair.DONOR_IDS[:2])
        self.assertEqual(pair.TARGET_IDS[3:], pair.DONOR_IDS[3:])
        self.assertNotEqual(pair.TARGET_IDS[2], pair.DONOR_IDS[2])

    def test_pair_margin_and_sign_use_the_saved_float32_bits(self):
        observations = [
            {"stage": "target", "is_bits": 0x40400000, "are_bits": 0x3f800000},
            {"stage": "donor", "is_bits": 0, "are_bits": 0x40000000},
            {"stage": "self", "is_bits": 0x40400000, "are_bits": 0x3f800000},
            {"stage": "cross", "is_bits": 0x3f000000, "are_bits": 0x3fc00000},
        ]
        result = pair.summarize_pair({"observations": observations})
        self.assertEqual(result["margins"], {"target": 2.0, "donor": -2.0, "self": 2.0, "cross": -1.0})
        self.assertEqual(result["target_minus_cross"], 3.0)
        self.assertIs(result["strict_sign_change"], True)

    def test_disabled_entries_do_not_start_or_import_device_work(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pair, "EXPOSED_PAIR_ENABLED", False), \
             patch.object(pair.OwnerSession, "from_environment") as session, \
             patch.object(pair, "_import_torch") as load:
            self.assertEqual(pair.owner_main(directory, directory, "id", "a" * 40, "unused"), 1)
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                pair.measure_owned(object(), object(), object(), directory, record_stage=lambda *_: None)
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                pair.run_calibration(workspace_root=directory, cache_dir=directory,
                    output=Path(directory) / "out.json", site_packages=directory, smi="unused")
            session.assert_not_called()
            load.assert_not_called()

    def test_workspace_is_checked_before_lease_and_again_before_torch(self):
        repo = Path(__file__).resolve().parents[1]
        workspace = next(parent.parent for parent in repo.parents if parent.name == "work")
        self.assertEqual(pair._workspace_root(repo, workspace), workspace)
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory).resolve()
            with self.assertRaises(ValueError):
                pair._workspace_root(repo, wrong)
            bad = SimpleNamespace(_identity_spec={"workspace_root": str(wrong)})
            with patch.object(pair, "EXPOSED_PAIR_ENABLED", True), \
                 patch.object(pair, "_verify_startup"), patch.object(pair, "_require_pristine_dependencies"), \
                 patch.object(pair.OwnerSession, "from_environment", return_value=bad), \
                 patch.object(pair.DeviceLease, "acquire") as acquire, \
                 patch.object(pair, "_import_torch") as load:
                self.assertEqual(pair.owner_main(directory, directory, "id", "a" * 40, "unused"), 1)
                acquire.assert_not_called()
                load.assert_not_called()
            for name in pair._OWNER_CUSTODY:
                pair._OWNER_CUSTODY[name] = False if name in ("entered", "drain_failed") else None
            session = SimpleNamespace(_identity_spec={"workspace_root": str(workspace)},
                                      identity=SimpleNamespace(workspace_root=wrong),
                                      device_lease_path="unused")
            with patch.object(pair, "EXPOSED_PAIR_ENABLED", True), \
                 patch.object(pair, "_verify_startup"), patch.object(pair, "_require_pristine_dependencies"), \
                 patch.object(pair.OwnerSession, "from_environment", return_value=session), \
                 patch.object(pair.OwnerSession, "check"), \
                 patch.object(pair.DeviceLease, "acquire", return_value=object()), \
                 patch.object(pair, "_import_torch") as load:
                self.assertEqual(pair.owner_main(directory, directory, "id", "a" * 40, "unused"), 1)
                load.assert_not_called()
            direct = object.__new__(pair.OwnerSession)
            direct._identity = SimpleNamespace(workspace_root=wrong)
            lease = object.__new__(pair.DeviceLease)
            cuda = SimpleNamespace(is_initialized=lambda: self.fail("CUDA was queried"))
            with patch.object(pair, "EXPOSED_PAIR_ENABLED", True), patch.object(pair.OwnerSession, "check"):
                with self.assertRaises(ValueError):
                    pair.measure_owned(direct, lease, SimpleNamespace(cuda=cuda), directory,
                                       record_stage=lambda *_: None)

    def test_workspace_rejects_nested_mixed_case_and_missing_work_ancestors(self):
        for relative in ("work/x/work/checkout", "WORK/checkout", "other/checkout"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                checkout = root / relative
                checkout.mkdir(parents=True, exist_ok=True)
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    pair._workspace_root(checkout, root)

    def test_claim_loss_is_detected_and_failed_release_retains_claims(self):
        session, device_lease = object(), object()
        gib = 1024 ** 3
        pair._OWNER_CUSTODY.update(session=session, device_lease=device_lease,
                                   cap=2 * gib, commitment=2 * gib + pair._HEADROOM)
        budget, lease = pair.reserve_owner_budget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                                                   cap=2 * gib, custody=pair._OWNER_CUSTODY)
        try:
            with patch.object(pair.OwnerSession, "check"):
                pair._check_pair_custody(session, device_lease)
                lease.release_trace("gpu", 64 * 1024)
                with self.assertRaisesRegex(RuntimeError, "custody"):
                    pair._check_pair_custody(session, device_lease)
                lease.reserve_trace("gpu", 64 * 1024)
            with patch.object(pair.OwnerSession, "check", side_effect=RuntimeError("lost owner")):
                with self.assertRaisesRegex(RuntimeError, "lost owner"):
                    pair._release_owner_budget_after_candidate(session, device_lease)
            self.assertEqual(budget.active_leases(), (lease,))
            self.assertEqual((lease.trace_bytes("gpu"), lease.trace_bytes("host")), (64 * 1024, pair._MIB))
        finally:
            lease.release_trace("gpu", 64 * 1024)
            lease.release_trace("host", pair._MIB)
            lease.release()

    def test_unverified_drain_latch_blocks_later_device_checks(self):
        pair._OWNER_CUSTODY["drain_failed"] = True
        with patch.object(pair.OwnerSession, "check") as check:
            with self.assertRaisesRegex(RuntimeError, "unverified drain"):
                pair._check_pair_custody(object(), object())
            check.assert_not_called()

    def test_failed_sync_then_interrupt_preserves_latch_and_blocks_reentry(self):
        session, device_lease = object(), object()
        gib = 1024 ** 3
        pair._OWNER_CUSTODY.update(session=session, device_lease=device_lease,
                                   cap=2 * gib, commitment=2 * gib + pair._HEADROOM)
        budget, lease = pair.reserve_owner_budget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                                                   cap=2 * gib, custody=pair._OWNER_CUSTODY)
        calls = []
        def failed_sync(index):
            self.assertIs(pair._OWNER_CUSTODY["drain_failed"], True)
            calls.append(index)
            raise RuntimeError("failed drain")
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=failed_sync))
        try:
            with patch.object(pair.OwnerSession, "check"):
                with self.assertRaises(KeyboardInterrupt):
                    try:
                        pair._synchronize_held(session, device_lease, torch)
                    except RuntimeError:
                        raise KeyboardInterrupt("interrupted while unwinding")
                self.assertIs(pair._OWNER_CUSTODY["drain_failed"], True)
                with self.assertRaisesRegex(RuntimeError, "unverified drain"):
                    pair._synchronize_held(session, device_lease, torch)
                self.assertEqual(calls, [0])
                self.assertEqual(budget.active_leases(), (lease,))
                self.assertEqual((lease.trace_bytes("gpu"), lease.trace_bytes("host")),
                                 (64 * 1024, pair._MIB))
        finally:
            lease.release_trace("gpu", 64 * 1024)
            lease.release_trace("host", pair._MIB)
            lease.release()

    def test_initial_failed_sync_latches_before_any_later_telemetry(self):
        calls = []
        def failed_sync(index):
            self.assertIs(pair._OWNER_CUSTODY["drain_failed"], True)
            calls.append("synchronize")
            raise KeyboardInterrupt("interrupted sync")
        cuda = SimpleNamespace(synchronize=failed_sync,
                               mem_get_info=lambda _: calls.append("memory"))
        with patch.object(pair.OwnerSession, "check"):
            with self.assertRaises(KeyboardInterrupt):
                pair._guarded_snapshot(object(), object(), SimpleNamespace(cuda=cuda))
            self.assertIs(pair._OWNER_CUSTODY["drain_failed"], True)
            with self.assertRaisesRegex(RuntimeError, "unverified drain"):
                pair._guarded_snapshot(object(), object(), SimpleNamespace(cuda=cuda))
        self.assertEqual(calls, ["synchronize"])

    def test_direct_owner_hard_exit_skips_python_shutdown_hooks(self):
        repo = Path(__file__).resolve().parents[1]
        script = r'''import atexit, pathlib, sys
from looking_glass import bert_gpu_exposed_pair as pair
atexit.register(pathlib.Path(sys.argv[1]).write_text, "atexit ran")
def fake_main(argv=None):
    if sys.argv[2] == "interrupt":
        raise KeyboardInterrupt("simulated failed-drain unwind")
    return int(sys.argv[2])
pair.main = fake_main
pair._owner_process_main()
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, expected in (("0", 0), ("1", 1), ("interrupt", 1)):
                marker = root / ("atexit-" + mode)
                result = subprocess.run([sys.executable, "-B", "-X", "pycache_prefix=" + str(root / "cache"),
                                         "-c", script, str(marker), mode], cwd=repo,
                                        capture_output=True, timeout=20)
                with self.subTest(mode=mode):
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertFalse(marker.exists())
            direct_marker = root / "atexit-direct-entry"
            direct = r'''import atexit, pathlib, runpy, sys
atexit.register(pathlib.Path(sys.argv[1]).write_text, "atexit ran")
owner_dir = sys.argv[2]
sys.argv = ["pair", "--owner", owner_dir, owner_dir, "run", "a" * 40, "unused"]
runpy.run_module("looking_glass.bert_gpu_exposed_pair", run_name="__main__")
'''
            result = subprocess.run([sys.executable, "-B", "-X", "pycache_prefix=" + str(root / "direct-cache"),
                                     "-c", direct, str(direct_marker), str(root)], cwd=repo,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(direct_marker.exists())

    def test_successful_sync_with_interrupted_postcheck_keeps_latch(self):
        session, device_lease = object(), object()
        gib = 1024 ** 3
        pair._OWNER_CUSTODY.update(session=session, device_lease=device_lease,
                                   cap=2 * gib, commitment=2 * gib + pair._HEADROOM)
        _, lease = pair.reserve_owner_budget(lambda: CudaMemorySnapshot(6 * gib, 8 * gib, 0, 0),
                                              cap=2 * gib, custody=pair._OWNER_CUSTODY)
        checks, calls = [], []
        def owner_check(*args):
            checks.append(1)
            if len(checks) == 5:
                raise KeyboardInterrupt("interrupted postcheck")
        try:
            with patch.object(pair.OwnerSession, "check", side_effect=owner_check):
                with self.assertRaises(KeyboardInterrupt):
                    pair._synchronize_held(session, device_lease,
                        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: calls.append(0))))
            self.assertEqual(calls, [0])
            self.assertIs(pair._OWNER_CUSTODY["drain_failed"], True)
        finally:
            lease.release_trace("gpu", 64 * 1024)
            lease.release_trace("host", pair._MIB)
            lease.release()

    def test_source_identity_ignores_git_overrides_and_replacement_refs(self):
        git = shutil.which("git")
        if git is None:
            self.skipTest("Git is unavailable")
        git = str(Path(git).resolve())
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            def run(*args):
                return subprocess.run([git, *args], cwd=repo, check=True,
                                      capture_output=True, text=True).stdout.strip()
            run("init", "-q")
            run("config", "user.name", "Pair Test")
            run("config", "user.email", "pair-test@example.invalid")
            source = repo / "source.py"
            source.write_text("honest = True\n", encoding="utf-8")
            run("add", "source.py")
            run("commit", "-qm", "honest")
            honest = run("rev-parse", "HEAD")
            with patch.object(pair, "_SOURCE_PATHS", ("source.py",)), \
                 patch.dict(os.environ, {"GIT_DIR": str(repo / "missing-git-dir"),
                                         "GIT_WORK_TREE": str(repo / "elsewhere")}):
                self.assertEqual(pair._source_identity(repo, git)[0], honest)
            source.write_text("honest = False\n", encoding="utf-8")
            run("add", "source.py")
            run("commit", "-qm", "replacement")
            replacement = run("rev-parse", "HEAD")
            run("checkout", "-q", "--detach", honest)
            run("replace", honest, replacement)
            source.write_text("honest = False\n", encoding="utf-8")
            with patch.object(pair, "_SOURCE_PATHS", ("source.py",)):
                with self.assertRaisesRegex(RuntimeError, "clean|differ"):
                    pair._source_identity(repo, git)

    def test_failed_live_sync_retains_plan_sources_and_captures(self):
        import torch
        from transformers import DistilBertConfig, DistilBertForMaskedLM

        torch.set_num_threads(1)
        config = DistilBertConfig(vocab_size=64, n_layers=6, n_heads=4, dim=16,
                                  hidden_dim=32, max_position_embeddings=16)
        model = DistilBertForMaskedLM(config).eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int64)
        source = torch.arange(16, dtype=torch.float32)
        for broken_clear in (False, True):
            custody = pair._ObservationCustody()
            calls = []
            class InterruptedCaptures(pair._RetainedPlanCaptures):
                def clear(self):
                    raise MemoryError("interrupted capture cleanup")
            def synchronize(stage):
                calls.append(stage)
                if stage == "output_live":
                    raise RuntimeError("failed live drain")
            wrapper = patch.object(pair, "_RetainedPlanCaptures", InterruptedCaptures) if broken_clear else \
                      patch.object(pair, "_RetainedPlanCaptures", pair._RetainedPlanCaptures)
            with wrapper, self.assertRaises((MemoryError, RuntimeError)):
                pair.observe(model, inputs, is_id=11, are_id=12, source=source,
                             synchronize=synchronize, custody=custody)
            self.assertEqual(calls, ["output_live"])
            self.assertEqual(len(custody.plan_sources), 1)
            self.assertEqual(len(custody.capture_roots), 2)
            refs = [weakref.ref(tensor) for tensor in (*custody.plan_sources, *custody.capture_roots.values())]
            gc.collect()
            self.assertTrue(all(ref() is not None for ref in refs))
            self.assertIsNotNone(custody.logits)


if __name__ == "__main__":
    unittest.main()
