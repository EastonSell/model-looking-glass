"""CPU-only supervised-owner check for interrupted lease release."""
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from looking_glass.supervisor import run_supervised


_INTERRUPTED_OWNER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from unittest.mock import patch

    from looking_glass import device_lease as api
    from looking_glass.supervisor import OwnerSession, SupervisorError

    report = Path(sys.argv[1])
    owner = OwnerSession.from_environment()
    lease = api.DeviceLease.acquire(owner.device_lease_path, timeout=3)
    owner.check(lease)
    unlock = api._unlock

    def interrupt_after_unlock(fd):
        unlock(fd)
        raise KeyboardInterrupt('synthetic interruption after OS unlock')

    try:
        with patch.object(api, '_unlock', interrupt_after_unlock):
            lease.release()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('expected release interruption')

    # The old descriptor may still exist, but the OS lock can have passed to
    # another owner. It must never regain authority or invalidate the successor.
    successor = api.DeviceLease.acquire(owner.device_lease_path, timeout=3)
    try:
        for name, callback in (('check', lambda: owner.check(lease)),
                               ('attest', lambda: owner.attest_clean(lease)),
                               ('retry', lease.release)):
            try:
                callback()
            except api.LeaseOwnershipError:
                pass
            else:
                raise AssertionError(name + ' accepted the stale lease')
        try:
            owner.check(successor)
        except SupervisorError:
            pass
        else:
            raise AssertionError('owner session switched to a successor lease')
        successor.check()
        assert not owner.attestation_path.exists()
        report.write_text('stale owner rejected; successor remains held', encoding='utf-8')
    finally:
        successor.release()
""")


class DeviceLeaseOwnerSeamTests(unittest.TestCase):
    def test_interrupted_unlock_cannot_attest_or_reuse_stale_owner(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / 'gpu.lock'
            marker = root / 'gpu.unsafe.json'
            report = root / 'child.txt'
            result = run_supervised([sys._base_executable, '-B', '-c', _INTERRUPTED_OWNER, str(report)],
                                    device_lease_path=device, marker_path=marker,
                                    timeout=20, lease_timeout=3, cwd=repo)
            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.clean)
            self.assertTrue(marker.is_file())
            self.assertEqual(report.read_text(encoding='utf-8'),
                             'stale owner rejected; successor remains held')


if __name__ == '__main__':
    unittest.main()
