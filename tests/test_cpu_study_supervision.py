"""Real CPU-only subprocess tests for the cooperative study contract."""

import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from looking_glass import cpu_study_supervision as api
from looking_glass import supervisor
from looking_glass.device_lease import LeaseTimeoutError


CLEAN = textwrap.dedent("""
    import os, sys
    from pathlib import Path
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(Path(sys.argv[1]), lease_timeout=3)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert session.identity is None
    import math
    assert math.isqrt(49) == 7
    Path(sys.argv[2]).touch()
    session.attest_clean(lease)
""")

HOLD = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(Path(sys.argv[1]), lease_timeout=3)
    root = Path(sys.argv[1])
    (root / 'ready').touch()
    deadline = time.monotonic() + 12
    while not (root / 'stop').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert (root / 'stop').exists()
    session.attest_clean(lease)
""")


class CpuStudySupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.result_file = self.root / "worked"

    def launch(self, code=CLEAN, *, root=None, args=(), executable=None, **kwargs):
        root = self.study if root is None else root
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
               if key in os.environ}
        # Keep interpreter/OS cache side effects inside this test's root.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        return api.run_cpu_study(
            [executable or sys._base_executable, "-c", textwrap.dedent(code), str(root),
             str(self.result_file), *map(str, args)],
            study_root=root, timeout=8, lease_timeout=3, terminate_grace=2,
            cwd=self.root, env=env, **kwargs,
        )

    def test_clean_study_reaps_and_allows_another(self):
        result = self.launch()
        self.assertTrue(result.clean)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(self.result_file.exists())
        self.assertFalse((self.study / "cpu-study.unsafe.json").exists())
        second = self.launch()
        self.assertTrue(second.clean)
        self.assertNotEqual(result.run_id, second.run_id)

    def test_torch_cpu_operation_after_claim(self):
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None or not torch_spec.origin:
            self.skipTest("Torch is not installed in this interpreter")
        dependency_root = Path(torch_spec.origin).resolve().parent.parent
        code = textwrap.dedent("""
            import sys
            from pathlib import Path
            from looking_glass.cpu_study_supervision import claim_cpu_study
            session, lease = claim_cpu_study(Path(sys.argv[1]))
            sys.path.insert(0, sys.argv[3])  # verified dependency path in this CPU test
            import torch
            value = torch.tensor([1.0, 2.0], device='cpu')
            assert value.device.type == 'cpu' and value.sum().item() == 3.0
            del value
            Path(sys.argv[2]).touch()
            session.attest_clean(lease)
        """)
        result = self.launch(code, args=(dependency_root,))
        self.assertTrue(result.clean)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(self.result_file.exists())

    def test_cpu_study_does_not_touch_separate_gpu_marker(self):
        gpu_root = self.root / "gpu"
        gpu_root.mkdir()
        gpu_marker = gpu_root / "GPU-test.unsafe.json"
        gpu_marker.write_text("existing GPU quarantine")
        self.assertTrue(self.launch().clean)
        self.assertEqual(gpu_marker.read_text(), "existing GPU quarantine")

    def test_invalid_roots_fail_before_owner_start(self):
        for root in (Path("relative-study"), self.root / "missing", self.result_file):
            if root == self.result_file:
                self.result_file.touch()
            with self.subTest(root=root), self.assertRaises((ValueError, OSError)):
                self.launch(root=root)
        self.assertFalse((self.study / "cpu-study.unsafe.json").exists())

    def test_mismatched_root_is_rejected_before_work(self):
        other = self.root / "other"
        other.mkdir()
        code = textwrap.dedent("""
            import sys
            from pathlib import Path
            from looking_glass.cpu_study_supervision import claim_cpu_study, CpuStudyError
            try:
                claim_cpu_study(Path(sys.argv[3]))
            except CpuStudyError:
                sys.exit(17)
            Path(sys.argv[2]).touch()
        """)
        result = self.launch(code, args=(other,))
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 17)
        self.assertFalse(self.result_file.exists())
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())
        self.assertFalse((other / "cpu-study.unsafe.json").exists())

    def test_preimported_cuda_capable_module_is_rejected(self):
        code = textwrap.dedent("""
            import sys, types
            from pathlib import Path
            sys.modules['torch.submodule'] = types.ModuleType('torch.submodule')
            from looking_glass.cpu_study_supervision import claim_cpu_study, CpuStudyError
            try:
                claim_cpu_study(Path(sys.argv[1]))
            except CpuStudyError:
                sys.exit(18)
            Path(sys.argv[2]).touch()
        """)
        result = self.launch(code)
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 18)
        self.assertFalse(self.result_file.exists())
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())

    def test_missing_or_nonempty_cuda_mask_is_rejected(self):
        code = textwrap.dedent("""
            import os, sys
            from pathlib import Path
            from looking_glass.cpu_study_supervision import claim_cpu_study, CpuStudyError
            if sys.argv[3] == 'missing':
                del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            try:
                claim_cpu_study(Path(sys.argv[1]))
            except CpuStudyError:
                sys.exit(19)
            Path(sys.argv[2]).touch()
        """)
        for mode in ("missing", "nonempty"):
            root = self.root / mode
            root.mkdir()
            result = self.launch(code, root=root, args=(mode,))
            self.assertFalse(result.clean)
            self.assertEqual(result.exit_code, 19)
            self.assertFalse(self.result_file.exists())
            self.assertTrue((root / "cpu-study.unsafe.json").exists())

    def test_identity_aware_session_is_rejected_before_lease(self):
        code = textwrap.dedent("""
            import os, sys
            from pathlib import Path
            os.environ['MLG_DEVICE_IDENTITY'] = '{}'
            from looking_glass.cpu_study_supervision import claim_cpu_study, CpuStudyError
            try:
                claim_cpu_study(Path(sys.argv[1]))
            except CpuStudyError:
                sys.exit(20)
            Path(sys.argv[2]).touch()
        """)
        result = self.launch(code)
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 20)
        self.assertFalse(self.result_file.exists())
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())

    def test_unclean_exit_preserves_marker_and_blocks_restart(self):
        result = self.launch("pass")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.clean)
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())
        with self.assertRaises(supervisor.UnsafeOwnerError):
            self.launch()

    def test_same_root_serializes_while_distinct_root_can_run(self):
        outcomes = []
        def run_holding():
            try:
                outcomes.append(self.launch(HOLD))
            except BaseException as exc:
                outcomes.append(exc)
        thread = threading.Thread(target=run_holding, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup((self.study / "stop").touch)
        deadline = time.monotonic() + 8
        while not (self.study / "ready").exists() and thread.is_alive():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.02)
        self.assertTrue((self.study / "ready").exists())
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
               if key in os.environ}
        with self.assertRaises(LeaseTimeoutError):
            api.run_cpu_study(
                [sys._base_executable, "-c", "raise AssertionError('must not spawn')"],
                study_root=self.study, timeout=3, lease_timeout=.2, env=env)
        second_root = self.root / "separate"
        second_root.mkdir()
        self.assertTrue(self.launch(root=second_root).clean)
        (self.study / "stop").touch()
        thread.join(timeout=8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0].clean)

    def test_interrupted_grant_reaps_child_and_keeps_study_marker(self):
        previous = sys.gettrace()
        children = []
        def trace(frame, event, arg):
            if frame.f_code is supervisor._SpawnCustody._grant.__code__ and event == "return":
                children.append(frame.f_locals["self"]._child)
                raise KeyboardInterrupt("grant return interrupted")
            return trace
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                self.launch()
        finally:
            sys.settrace(previous)
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())

    def test_interrupted_claim_return_exits_without_attestation(self):
        code = textwrap.dedent("""
            import sys
            from pathlib import Path
            from looking_glass import cpu_study_supervision as api
            def trace(frame, event, arg):
                if frame.f_code is api.claim_cpu_study.__code__ and event == 'return':
                    Path(sys.argv[1], 'claim-return').touch()
                    raise KeyboardInterrupt('claim return lost')
                return trace
            try:
                sys.settrace(trace)
                api.claim_cpu_study(Path(sys.argv[1]))
            except KeyboardInterrupt:
                sys.exit(17)
            Path(sys.argv[2]).touch()
        """)
        result = self.launch(code)
        self.assertFalse(result.clean)
        self.assertEqual(result.exit_code, 17)
        self.assertTrue((self.study / "claim-return").exists())
        self.assertFalse(self.result_file.exists())
        self.assertTrue((self.study / "cpu-study.unsafe.json").exists())
        with self.assertRaises(supervisor.UnsafeOwnerError):
            self.launch()

    def test_missing_launch_custody_contract_fails_before_spawn(self):
        with mock.patch.object(supervisor, "SUPERVISOR_LAUNCH_CUSTODY_CONTRACT", "old"):
            with self.assertRaises(api.CpuStudyError):
                self.launch()
        self.assertFalse((self.study / "cpu-study.unsafe.json").exists())


if __name__ == "__main__":
    unittest.main()
