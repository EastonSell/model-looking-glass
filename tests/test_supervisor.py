"""Real-process tests for owner supervision; no CUDA device is accessed."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


OWNER_CLEAN = textwrap.dedent("""
    from looking_glass.supervisor import OwnerSession
    from looking_glass.device_lease import DeviceLease
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
    session.check(lease)
    # Production owners must first stop, drain, synchronize and check quarantine.
    session.attest_clean(lease)
    # Deliberately keep lease until process exit.
""")

GPU_A = 'GPU-12345678-1234-5678-9abc-1234567890ab'
OWNER_IDENTITY_CLEAN = textwrap.dedent("""
    from looking_glass.supervisor import OwnerSession
    from looking_glass.device_lease import DeviceLease
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
    session.check(lease)
    assert session.identity is not None
    session.verify_runtime_binding(lease, runtime_uuid=session.identity.gpu_uuid, device_count=1)
    session.attest_clean(lease)
""")

OWNER_HOLD = textwrap.dedent("""
    from pathlib import Path
    import sys, time
    from looking_glass.supervisor import OwnerSession
    from looking_glass.device_lease import DeviceLease
    session = OwnerSession.from_environment()
    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
    session.check(lease)
    ready, stop, done = map(Path, sys.argv[1:4])
    ready.write_text('ready')
    deadline = time.monotonic() + 15
    try:
        while not stop.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        session.attest_clean(lease)
    finally:
        done.write_text('done')
""")

SUPERVISOR_DRIVER = textwrap.dedent("""
    import sys
    from looking_glass.supervisor import run_supervised
    result = run_supervised(
        [sys._base_executable, '-c', sys.argv[3], *sys.argv[4:]],
        device_lease_path=sys.argv[1], marker_path=sys.argv[2],
        timeout=20, lease_timeout=3,
    )
    sys.exit(0 if result.clean else 4)
""")


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("looking_glass.supervisor")
        except ModuleNotFoundError as exc:
            if exc.name == "looking_glass.supervisor":
                self.fail("the owner supervisor is not implemented")
            raise
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.device = self.directory / "gpu.lock"
        self.marker = self.directory / "gpu.unsafe.json"

    def run_owner(self, code=OWNER_CLEAN, timeout=5):
        return self.api.run_supervised(
            [sys._base_executable, "-c", code],
            device_lease_path=self.device,
            marker_path=self.marker,
            timeout=timeout,
            lease_timeout=3,
        )

    def start_holding_supervisor(self):
        ready = self.directory / "ready"
        stop = self.directory / "stop"
        done = self.directory / "done"
        process = subprocess.Popen(
            [sys._base_executable, "-c", SUPERVISOR_DRIVER, str(self.device),
             str(self.marker), OWNER_HOLD, str(ready), str(stop), str(done)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        def cleanup():
            stop.touch()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            deadline = time.monotonic() + 1
            while ready.exists() and not done.exists() and time.monotonic() < deadline:
                time.sleep(.02)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 8
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(ready.exists(), 'supervised child did not start')
        return process, stop, done

    def test_clean_reaped_owner_allows_subsequent_owner(self):
        first = self.run_owner()
        self.assertTrue(first.clean)
        self.assertEqual(first.exit_code, 0)
        self.assertFalse(self.marker.exists())
        second = self.run_owner()
        self.assertTrue(second.clean)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertFalse(self.marker.exists())

    def test_missing_attestation_keeps_marker_and_blocks_successor(self):
        code = textwrap.dedent("""
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
        """)
        failed = self.run_owner(code)
        self.assertFalse(failed.clean)
        self.assertTrue(self.marker.exists())
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_timeout_terminates_only_owner_and_retains_marker(self):
        code = textwrap.dedent("""
            import time
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            time.sleep(20)
        """)
        failed = self.run_owner(code, timeout=0.8)
        self.assertFalse(failed.clean)
        self.assertTrue(failed.timed_out)
        self.assertIsNotNone(failed.exit_code)
        self.assertTrue(self.marker.exists())
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_deleted_marker_must_not_allow_a_successor(self):
        code = textwrap.dedent("""
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.marker_path.unlink()
        """)
        failed = self.run_owner(code)
        self.assertFalse(failed.clean)
        self.assertTrue(self.marker.exists(), 'lost marker must be restored as unsafe')
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_same_lease_cannot_bypass_quarantine_with_another_marker(self):
        failed = self.run_owner("pass")
        self.assertFalse(failed.clean)
        other = self.directory / "another.unsafe.json"
        with self.assertRaises(ValueError):
            self.api.run_supervised(
                [sys._base_executable, "-c", OWNER_CLEAN],
                device_lease_path=self.device, marker_path=other,
                timeout=5, lease_timeout=3,
            )
        self.assertTrue(self.marker.exists())

    def test_wrapper_descendant_is_rejected_before_device_initialization(self):
        initialized = self.directory / "initialized"
        descendant = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass.supervisor import OwnerSession, UnsafeOwnerError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            try:
                session.check(lease)
            except UnsafeOwnerError:
                sys.exit(17)
            Path(sys.argv[1]).write_text('simulated CUDA initialization')
            session.attest_clean(lease)
        """)
        wrapper = textwrap.dedent("""
            import subprocess, sys
            sys.exit(subprocess.call([sys._base_executable, '-c', sys.argv[1], sys.argv[2]]))
        """)
        result = self.api.run_supervised(
            [sys._base_executable, "-c", wrapper, descendant, str(initialized)],
            device_lease_path=self.device, marker_path=self.marker,
            timeout=5, lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertFalse(initialized.exists(), 'descendant reached simulated CUDA')
        self.assertTrue(self.marker.exists())

    def test_venv_launcher_is_rejected_before_device_use_when_it_spawns(self):
        probe = subprocess.Popen(
            [sys.executable, '-c', 'import os; print(os.getpid())'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        stdout, _ = probe.communicate(timeout=5)
        if probe.returncode != 0 or int(stdout.strip()) == probe.pid:
            self.skipTest('this interpreter is a direct executable, not a launcher')
        initialized = self.directory / 'launcher-initialized'
        code = textwrap.dedent("""
            from pathlib import Path
            import sys
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            try:
                session.check(lease)
            except SupervisorError:
                sys.exit(17)
            Path(sys.argv[1]).touch()
        """)
        result = self.api.run_supervised(
            [sys.executable, '-c', code, str(initialized)],
            device_lease_path=self.device, marker_path=self.marker,
            timeout=5, lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertFalse(initialized.exists())
        self.assertTrue(self.marker.exists())

    def test_competing_supervisor_cannot_start_a_second_owner(self):
        from looking_glass.device_lease import LeaseTimeoutError
        first, stop, _ = self.start_holding_supervisor()
        second_started = self.directory / "second-started"
        code = f"from pathlib import Path; Path({str(second_started)!r}).touch()"
        with self.assertRaises(LeaseTimeoutError):
            self.api.run_supervised(
                [sys._base_executable, "-c", code], device_lease_path=self.device,
                marker_path=self.marker, timeout=5, lease_timeout=.2,
            )
        self.assertFalse(second_started.exists())
        stop.touch()
        self.assertEqual(first.wait(timeout=8), 0)
        self.assertFalse(self.marker.exists())

    def test_killed_supervisor_leaves_owner_quarantined_after_child_exits(self):
        from looking_glass.device_lease import DeviceLease
        supervisor, _, _ = self.start_holding_supervisor()
        supervisor.terminate()
        supervisor.wait(timeout=5)
        self.assertTrue(self.marker.exists())
        # Wait for the exact device lock to become available before attempting
        # recovery; the missing parent still must not clear the sticky marker.
        lease = DeviceLease.acquire(self.device, timeout=3)
        lease.release()
        self.assertTrue(self.marker.exists())
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_nonzero_workload_exit_can_have_verified_clean_device_teardown(self):
        result = self.run_owner(OWNER_CLEAN + "\nimport sys; sys.exit(17)\n")
        self.assertEqual(result.exit_code, 17)
        self.assertTrue(result.clean, 'safe device teardown should be recorded independently')
        self.assertFalse(self.marker.exists())
        self.assertTrue(self.run_owner().clean)

    def test_killed_supervisor_forces_owner_to_exit_without_attestation(self):
        from looking_glass.device_lease import DeviceLease
        supervisor, _, done = self.start_holding_supervisor()
        supervisor.terminate()
        supervisor.wait(timeout=5)
        # The owner still holds its lease until it exits; no stop signal is sent.
        # This fails if a dead supervisor leaves its device-owning child alive.
        lease = DeviceLease.acquire(self.device, timeout=3)
        lease.release()
        self.assertFalse(done.exists(), 'watchdog exit must not run normal attestation')
        self.assertTrue(self.marker.exists())
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_mismatched_attestation_is_never_accepted(self):
        code = textwrap.dedent("""
            import json
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.attest_clean(lease)
            state = json.loads(session.attestation_path.read_text())
            state['owner_pid'] += 1
            session.attestation_path.write_text(json.dumps(state))
        """)
        failed = self.run_owner(code)
        self.assertFalse(failed.clean)
        self.assertTrue(self.marker.exists())
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_dangling_attestation_is_uncertain_state_not_an_empty_slot(self):
        attestation = self.marker.with_name(self.marker.name + ".attestation")
        try:
            attestation.symlink_to(self.directory / "missing")
        except OSError as exc:
            self.skipTest(f"creating symlinks unavailable: {exc}")
        with self.assertRaises(self.api.UnsafeOwnerError):
            self.run_owner()

    def test_parent_identity_mismatch_blocks_owner_before_device_use(self):
        initialized = self.directory / "initialized"
        code = textwrap.dedent("""
            import os, sys
            from pathlib import Path
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            os.environ['MLG_SUPERVISOR_BIRTH'] = 'wrong-parent'
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            try:
                session.check(lease)
            except SupervisorError:
                sys.exit(17)
            Path(sys.argv[1]).touch()
        """)
        result = self.api.run_supervised(
            [sys._base_executable, "-c", code, str(initialized)],
            device_lease_path=self.device, marker_path=self.marker,
            timeout=5, lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertFalse(initialized.exists())
        self.assertTrue(self.marker.exists())

    def test_hardlink_alias_cannot_create_an_independent_marker_namespace(self):
        from looking_glass.device_lease import DeviceLease
        lease = DeviceLease.acquire(self.device, timeout=2)
        lease.release()
        alias = self.directory / 'alias.lock'
        try:
            os.link(self.device, alias)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        with self.assertRaises(self.api.SupervisorError):
            self.api.run_supervised(
                [sys._base_executable, "-c", OWNER_CLEAN],
                device_lease_path=alias, marker_path=alias.with_suffix('.unsafe.json'),
                timeout=5, lease_timeout=3,
            )

    def test_forged_lease_subclass_cannot_authorize_device_use(self):
        initialized = self.directory / "forged"
        code = textwrap.dedent("""
            import sys
            from pathlib import Path
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            class ForgedLease(DeviceLease):
                def __init__(self): self.path = session.device_lease_path
                def check(self): pass
            try:
                session.check(ForgedLease())
            except SupervisorError:
                sys.exit(17)
            Path(sys.argv[1]).touch()
        """)
        outcome = self.api.run_supervised(
            [sys._base_executable, "-c", code, str(initialized)],
            device_lease_path=self.device, marker_path=self.marker,
            timeout=5, lease_timeout=3,
        )
        self.assertFalse(outcome.clean)
        self.assertFalse(initialized.exists())
        self.assertTrue(self.marker.exists())

    def test_owner_spawn_failure_preserves_unsafe_marker(self):
        # An unknown native-spawn outcome intentionally retains custody until
        # this process exits. Run the check in a disposable process rather than
        # clearing its recovery record and affecting later tests.
        driver = textwrap.dedent("""
            from pathlib import Path
            import os, sys
            from looking_glass import supervisor as api
            device, marker, missing = map(Path, sys.argv[1:4])
            try:
                api.run_supervised([str(missing)], device_lease_path=device,
                                   marker_path=marker, timeout=2, lease_timeout=2)
            except OSError:
                pass
            else:
                raise AssertionError('nonexistent executable unexpectedly started')
            if not marker.exists():
                raise AssertionError('spawn failure lost its unsafe marker')
            records = api.pending_supervised_children()
            if os.name == 'nt':
                if len(records) != 1 or not records[0]._attempted:
                    raise AssertionError('unknown spawn did not retain its custody record')
                if records[0]._known_child():
                    raise AssertionError('failed spawn unexpectedly has a known child')
                try:
                    records[0].recover()
                except api.UnsafeOwnerError:
                    pass
                else:
                    raise AssertionError('unknown spawn must remain quarantined')
            try:
                api.run_supervised([sys._base_executable, '-c', 'pass'],
                                   device_lease_path=device, marker_path=marker,
                                   timeout=2, lease_timeout=2)
            except api.UnsafeOwnerError:
                pass
            else:
                raise AssertionError('unsafe marker allowed a successor')
        """)
        result = subprocess.run(
            [sys._base_executable, '-c', driver, str(self.device), str(self.marker),
             str(self.directory / 'nonexistent-executable')],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.marker.exists())

    def test_gpu_mode_revalidates_identity_under_both_owner_leases(self):
        from looking_glass.device_identity import resolve_device_identity
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = GPU_A
        identity = resolve_device_identity(self.directory, GPU_A, physical_uuids=(GPU_A,), environ=env)
        result = self.api.run_supervised(
            [sys._base_executable, '-c', OWNER_IDENTITY_CLEAN],
            device_lease_path=identity.lock_path, marker_path=identity.marker_path,
            identity=identity, env={'CUDA_VISIBLE_DEVICES': GPU_A}, timeout=5,
            lease_timeout=3,
        )
        self.assertTrue(result.clean)
        self.assertFalse(identity.marker_path.exists())

    def test_gpu_mode_refuses_clean_attestation_without_runtime_verification(self):
        from looking_glass.device_identity import resolve_device_identity
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = GPU_A
        identity = resolve_device_identity(self.directory, GPU_A, physical_uuids=(GPU_A,), environ=env)
        code = textwrap.dedent("""
            import sys
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            try:
                session.attest_clean(lease)
            except SupervisorError:
                sys.exit(17)
        """)
        result = self.api.run_supervised(
            [sys._base_executable, '-c', code],
            device_lease_path=identity.lock_path, marker_path=identity.marker_path,
            identity=identity, env={'CUDA_VISIBLE_DEVICES': GPU_A}, timeout=5,
            lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())

    def test_gpu_mode_rejects_identity_with_substituted_marker(self):
        from dataclasses import replace
        from looking_glass.device_identity import DeviceIdentityError, resolve_device_identity
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = GPU_A
        identity = resolve_device_identity(self.directory, GPU_A, physical_uuids=(GPU_A,), environ=env)
        forged = replace(identity, marker_path=self.directory / 'other.unsafe.json')
        with self.assertRaises(DeviceIdentityError):
            self.api.run_supervised(
                [sys._base_executable, '-c', OWNER_IDENTITY_CLEAN],
                device_lease_path=identity.lock_path, marker_path=identity.marker_path,
                identity=forged, env={'CUDA_VISIBLE_DEVICES': GPU_A}, timeout=5,
                lease_timeout=3,
            )
        self.assertFalse(identity.marker_path.exists())

    def test_gpu_owner_cannot_drop_identity_environment_and_bypass_binding(self):
        from looking_glass.device_identity import resolve_device_identity
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = GPU_A
        identity = resolve_device_identity(self.directory, GPU_A, physical_uuids=(GPU_A,), environ=env)
        initialized = self.directory / 'identity-bypass'
        code = textwrap.dedent("""
            import os, sys
            from pathlib import Path
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            os.environ.pop('MLG_DEVICE_IDENTITY')
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            try:
                session.check(lease)
            except SupervisorError:
                sys.exit(17)
            Path(sys.argv[1]).touch()
        """)
        result = self.api.run_supervised(
            [sys._base_executable, '-c', code, str(initialized)],
            device_lease_path=identity.lock_path, marker_path=identity.marker_path,
            identity=identity, env={'CUDA_VISIBLE_DEVICES': GPU_A}, timeout=5,
            lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertFalse(initialized.exists())
        self.assertTrue(identity.marker_path.exists())

    def test_gpu_attestation_rejects_a_new_lease_after_runtime_verification(self):
        from looking_glass.device_identity import resolve_device_identity
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = GPU_A
        identity = resolve_device_identity(self.directory, GPU_A, physical_uuids=(GPU_A,), environ=env)
        code = textwrap.dedent("""
            import sys
            from looking_glass.supervisor import OwnerSession, SupervisorError
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            first = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(first)
            session.verify_runtime_binding(first, runtime_uuid=session.identity.gpu_uuid, device_count=1)
            first.release()
            second = DeviceLease.acquire(session.device_lease_path, timeout=3)
            try:
                session.attest_clean(second)
            except SupervisorError:
                sys.exit(17)
        """)
        result = self.api.run_supervised(
            [sys._base_executable, '-c', code], identity=identity,
            device_lease_path=identity.lock_path, marker_path=identity.marker_path,
            env={'CUDA_VISIBLE_DEVICES': GPU_A}, timeout=5, lease_timeout=3,
        )
        self.assertFalse(result.clean)
        self.assertTrue(identity.marker_path.exists())

    def test_monkeypatched_session_check_cannot_forge_clean_attestation(self):
        code = textwrap.dedent("""
            import sys
            from looking_glass.supervisor import OwnerSession, SupervisorError
            session = OwnerSession.from_environment()
            session.check = lambda lease: None
            try:
                session.attest_clean(None)
            except SupervisorError:
                sys.exit(17)
        """)
        result = self.run_owner(code)
        self.assertFalse(result.clean)
        self.assertTrue(self.marker.exists())

    def test_identityless_cpu_mode_masks_cuda_visibility(self):
        code = textwrap.dedent("""
            import os
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.attest_clean(lease)
        """)
        self.assertTrue(self.run_owner(code).clean)


if __name__ == "__main__":
    unittest.main()
