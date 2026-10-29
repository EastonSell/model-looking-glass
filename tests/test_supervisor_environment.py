"""Real CPU children exercise supervisor environment replacement."""

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

from looking_glass.supervisor import StartupChannel, run_supervised


class SupervisorEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.device = self.root / 'device.lock'
        self.marker = self.root / 'device.unsafe.json'
        self.project = Path(__file__).resolve().parents[1]

    def run_child(self, code, *, env, **options):
        return run_supervised(
            [sys._base_executable, '-S', '-P', '-c', textwrap.dedent(code)],
            device_lease_path=self.device, marker_path=self.marker,
            timeout=8, lease_timeout=3, env=env, **options,
        )

    def test_replacement_drops_parent_poison_before_python_imports(self):
        poison = self.root / 'poison'
        poison.mkdir()
        sentinel = self.root / 'shadow-executed'
        (poison / 'json.py').write_text(
            'from pathlib import Path\nPath(' + repr(str(sentinel)) + ').touch()\n'
            'raise RuntimeError("poisoned stdlib import")\n', encoding='utf-8')
        child_env = {'TEST_PROJECT': str(self.project), 'TEST_KEEP': 'explicit',
                     'mlg_owner_run_id': 'forged', 'mLg_DeViCe_IdEnTiTy': 'forged',
                     'cUdA_Visible_Devices': 'forged'}
        if 'SystemRoot' in os.environ:
            child_env['SystemRoot'] = os.environ['SystemRoot']
        code = """
            import json, os, sys
            assert 'poison' not in (json.__file__ or '')
            assert 'PYTHONPATH' not in os.environ and 'PYTHONHOME' not in os.environ
            assert 'HOST_SECRET_SENTINEL' not in os.environ
            assert 'CUDA_PARENT_SENTINEL' not in os.environ
            assert os.environ['TEST_KEEP'] == 'explicit'
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            assert os.environ['MLG_OWNER_RUN_ID'] != 'forged'
            assert 'MLG_DEVICE_IDENTITY' not in os.environ
            sys.path.append(os.environ['TEST_PROJECT'])
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.attest_clean(lease)
        """
        parent = {'PYTHONPATH': str(poison), 'PYTHONHOME': str(poison),
                  'HOST_SECRET_SENTINEL': 'not-for-child', 'CUDA_PARENT_SENTINEL': 'not-for-child'}
        with mock.patch.dict(os.environ, parent):
            result = self.run_child(code, env=child_env, inherit_environment=False)
            self.assertEqual(os.environ['HOST_SECRET_SENTINEL'], 'not-for-child')
        self.assertTrue(result.clean)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(sentinel.exists())
        self.assertFalse(self.marker.exists())

    def test_default_preserves_existing_environment_behavior(self):
        env = {'TEST_PROJECT': str(self.project)}
        code = """
            import os, sys
            assert os.environ['HOST_INHERITED_SENTINEL'] == 'parent'
            sys.path.append(os.environ['TEST_PROJECT'])
            from looking_glass.supervisor import OwnerSession
            from looking_glass.device_lease import DeviceLease
            session = OwnerSession.from_environment()
            lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
            session.check(lease)
            session.attest_clean(lease)
        """
        with mock.patch.dict(os.environ, {'HOST_INHERITED_SENTINEL': 'parent'}):
            result = self.run_child(code, env=env)
        self.assertTrue(result.clean)

    def test_replacement_flag_rejects_nonboolean(self):
        with self.assertRaises(ValueError):
            self.run_child('', env={}, inherit_environment='false')
        self.assertFalse(self.marker.exists())

    def test_replacement_preserves_the_authoritative_startup_channel(self):
        channel = StartupChannel(max_bytes=32)
        self.addCleanup(channel.close)
        env = {'TEST_PROJECT': str(self.project),
               'mlg_startup_channel_handle': '0', 'mlg_startup_channel_limit': '1'}
        if 'SystemRoot' in os.environ:
            env['SystemRoot'] = os.environ['SystemRoot']
        result = {}

        def launch():
            try:
                result['value'] = self.run_child("""
                    import os, sys
                    sys.path.append(os.environ['TEST_PROJECT'])
                    from looking_glass.supervisor import OwnerSession
                    from looking_glass.device_lease import DeviceLease
                    session = OwnerSession.from_environment()
                    lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
                    session.send_startup(lease, b'valid')
                    session.attest_clean(lease)
                """, env=env, inherit_environment=False, startup_channel=channel)
            except BaseException as exc:
                result['error'] = exc

        thread = threading.Thread(target=launch, daemon=True)
        thread.start()
        self.assertEqual(channel.receive(timeout=6), b'valid')
        self.assertTrue(channel.wait_closed(timeout=3))
        thread.join(timeout=8)
        self.assertFalse(thread.is_alive())
        self.assertNotIn('error', result)
        self.assertTrue(result['value'].clean)


if __name__ == '__main__':
    unittest.main()
