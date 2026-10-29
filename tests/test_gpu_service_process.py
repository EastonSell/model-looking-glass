"""CPU-only config and startup-gate checks for the fixed service child."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from looking_glass import gpu_service_process as child


class ServiceProcessConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stop = self.root / 'stop'
        self.value = {
            'schema': 'gpu-service-owner-v1', 'repo_root': str(self.root),
            'cache_root': str(self.root), 'workspace_root': str(self.root),
            'stop_path': str(self.stop), 'teams': ['alpha', 'beta'],
            'source_base_sha256': 'a' * 64, 'asset_manifest_sha256': 'b' * 64,
            'profiles': [
                {'document': str(self.root / f'{i}.json'),
                 'evidence': str(self.root / f'{i}.evidence'),
                 'source_manifest': str(self.root / f'{i}.source'),
                 'approved_profile_id': str(i) * 64}
                for i in (1, 2, 3)
            ],
        }

    def data(self, value=None):
        return json.dumps(self.value if value is None else value).encode()

    def test_valid_configuration_has_fixed_teams_and_three_distinct_pins(self):
        config = child._parse_config(self.data())
        self.assertEqual(config.teams, ('alpha', 'beta'))
        self.assertEqual(tuple(record[3] for record in config.profiles),
                         ('1' * 64, '2' * 64, '3' * 64))
        self.assertEqual(config.stop_path, self.stop)

    def test_invalid_configuration_is_rejected_before_owner(self):
        for mutate in (
            lambda x: x.update(schema='other'),
            lambda x: x.update(repo_root='relative'),
            lambda x: x.update(teams=['alpha', 'alpha']),
            lambda x: x.update(source_base_sha256='bad'),
            lambda x: x['profiles'][2].update(approved_profile_id='1' * 64),
            lambda x: x.update(extra='unexpected'),
        ):
            value = json.loads(self.data())
            mutate(value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'invalid private service'):
                child._parse_config(self.data(value))
        with self.assertRaisesRegex(ValueError, 'invalid private service'):
            child._parse_config(b'{"schema":1,"schema":2}')

    def test_main_parses_only_pinned_captured_bytes_before_owner(self):
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        info = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['--config', str(path), '--config-id', f'{info.st_dev}:{info.st_ino}',
                '--config-sha256', digest]
        with patch.object(child, '_owner', return_value=0) as owner:
            self.assertEqual(child.main(argv), 0)
            self.assertEqual(owner.call_count, 1)
        path.write_bytes(self.data({**self.value, 'teams': ['other']}))
        with patch.object(child, '_owner') as owner, patch('sys.stderr') as error:
            self.assertEqual(child.main(argv), 20)
            owner.assert_not_called()
            error.write.assert_not_called()

    def test_preloaded_gpu_dependency_fails_before_owner_acquisition(self):
        config = child._parse_config(self.data())
        with patch.dict('sys.modules', {'torch': object()}):
            with self.assertRaisesRegex(RuntimeError, 'imported before owner acquisition'):
                child._owner(config)

    def test_preloaded_hub_with_stale_home_fails_before_owner_acquisition(self):
        # The hub caches HF_HOME at import; changing os.environ afterward
        # does not change its effective cache configuration.
        program = r'''
import os, sys
from pathlib import Path
from unittest.mock import patch
sys.path.extend((sys.argv[1], sys.argv[2]))
os.environ['HF_HOME'] = sys.argv[4]
import huggingface_hub.constants as constants
assert Path(constants.HF_HOME).resolve() == Path(sys.argv[4]).resolve()
os.environ['HF_HOME'] = sys.argv[5]
assert Path(constants.HF_HOME).resolve() != Path(os.environ['HF_HOME']).resolve()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
from looking_glass import gpu_service_process as child, two_token_service as service
config = child._parse_config(Path(sys.argv[3]).read_bytes())
with patch.object(service, '_check_owner_contract') as gate:
    try:
        child._owner(config)
    except RuntimeError as exc:
        assert 'imported before owner acquisition' in str(exc)
    else:
        raise AssertionError('preloaded Hub was accepted')
    gate.assert_not_called()
'''
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        old_home, new_home = self.root / 'old-hf-home', self.root / 'new-hf-home'
        old_home.mkdir()
        new_home.mkdir()
        repo = Path(__file__).resolve().parents[1]
        outcome = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-c', program,
                                  str(repo), sysconfig.get_paths()['purelib'], str(path),
                                  str(old_home), str(new_home)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)

    def test_owner_contract_gate_precedes_lease_and_torch_import(self):
        program = r'''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.extend((sys.argv[1], sys.argv[2]))
from looking_glass import gpu_service_process as child, two_token_service as service
from looking_glass.supervisor import OwnerSession
config = child._parse_config(Path(sys.argv[3]).read_bytes())
with patch.object(service, '_check_owner_contract', side_effect=RuntimeError('missing contract')), \
     patch.object(OwnerSession, 'from_environment') as owner:
    try:
        child._owner(config)
    except RuntimeError as exc:
        assert str(exc) == 'missing contract'
    else:
        raise AssertionError('owner gate was bypassed')
    owner.assert_not_called()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        repo = Path(__file__).resolve().parents[1]
        outcome = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-c', program,
                                  str(repo), sysconfig.get_paths()['purelib'], str(path)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)

    def test_missing_completion_contract_blocks_child_before_owner(self):
        program = r'''
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
sys.path.extend((sys.argv[1], sys.argv[2]))
from looking_glass import gpu_service_process as child, two_token_service as service
from looking_glass.device_lease import DeviceLease
from looking_glass.supervisor import OwnerSession
config = replace(child._parse_config(Path(sys.argv[3]).read_bytes()), repo_root=Path(sys.argv[1]))
with patch.object(service._owner_module, 'OWNER_COMPLETION_CUSTODY_CONTRACT', None, create=True), \
     patch.object(OwnerSession, 'from_environment') as owner, \
     patch.object(DeviceLease, 'acquire') as lease:
    try:
        child._owner(config)
    except RuntimeError as exc:
        assert 'completion custody' in str(exc)
    else:
        raise AssertionError('missing completion contract was accepted')
    owner.assert_not_called()
    lease.assert_not_called()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        repo = Path(__file__).resolve().parents[1]
        outcome = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-c', program,
                                  str(repo), sysconfig.get_paths()['purelib'], str(path)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)

    def test_owner_failure_does_not_touch_a_failing_stderr(self):
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        info = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['--config', str(path), '--config-id', f'{info.st_dev}:{info.st_ino}',
                '--config-sha256', digest]
        with patch.object(child, '_owner', side_effect=RuntimeError('post-owner failure')), \
             patch('sys.stderr') as error:
            error.write.side_effect = RuntimeError('blocked or broken pipe')
            self.assertEqual(child.main(argv), 32)
            error.write.assert_not_called()
        with patch.object(child, '_owner', side_effect=MemoryError('allocation failure')):
            self.assertEqual(child.main(argv), 32)

    def test_workload_status_is_distinct_from_verified_teardown(self):
        def code(status, verified, error=None, *, failed=False, alive=False):
            result = SimpleNamespace(status=status, teardown_verified=verified, error_type=error)
            return child._result_code(result, monitor_failed=failed, monitor_alive=alive)
        self.assertEqual(code('clean', True), 0)
        self.assertEqual(code('clean', False), 32)
        self.assertEqual(code('worker_failed', True), 31)
        self.assertEqual(code('worker_failed', False, 'OutOfMemoryError'), 32)
        self.assertEqual(code('worker_failed', True, 'OutOfMemoryError'), 36)
        self.assertEqual(code('quarantined', False), 32)
        self.assertEqual(code('shutdown_incomplete', False), 33)
        self.assertEqual(code('clean', True, failed=True), 34)
        self.assertEqual(code('clean', True, alive=True), 35)

    def test_direct_owner_exit_skips_atexit_on_success_and_interruption(self):
        program = ("import atexit,sys;from pathlib import Path;"
                   "atexit.register(lambda:Path(sys.argv[1]).write_text('ran'));"
                   "from looking_glass import gpu_service_process as child;"
                   "child.main=(lambda:0) if sys.argv[2]=='clean' else "
                   "(lambda:(_ for _ in ()).throw(KeyboardInterrupt()));child._exit_owner()")
        repo = Path(__file__).resolve().parents[1]
        for mode, expected in (('clean', 0), ('interrupt', 70)):
            sentinel = self.root / (mode + '.txt')
            outcome = subprocess.run([sys.executable, '-c', program, str(sentinel), mode],
                                     cwd=repo, capture_output=True, text=True, timeout=10)
            with self.subTest(mode=mode):
                self.assertEqual(outcome.returncode, expected)
                self.assertFalse(sentinel.exists())

    def test_real_backend_constructor_and_identity_callbacks_compose_on_cpu(self):
        # This fresh subprocess uses the real merged backend constructor, but
        # fakes the owner, measured profiles and device work. It never imports
        # Torch or provides evidence that a live service is safe to launch.
        program = r'''
import hashlib, json, sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.extend((sys.argv[1], sys.argv[2]))
from looking_glass import gpu_service_process as child
from looking_glass import gpu_service, trusted_service_profile as profile, two_token_service as service
from looking_glass.device_identity import DeviceIdentity
from looking_glass.device_lease import DeviceLease
from looking_glass.supervisor import OwnerSession
config = child._parse_config(Path(sys.argv[3]).read_bytes())
source = SimpleNamespace(source_base=b'source', asset_manifest=b'assets')
config = replace(config, source_base_sha256=hashlib.sha256(source.source_base).hexdigest(),
                 asset_manifest_sha256=hashlib.sha256(source.asset_manifest).hexdigest())
identity = object.__new__(DeviceIdentity)
object.__setattr__(identity, 'workspace_root', config.workspace_root)
lease = object()
session = Mock()
session.identity = identity
session.device_lease_path = config.workspace_root / 'device.lock'
live = SimpleNamespace(keys=('load', 'cold', 'warm'))
registry = object()
@dataclass(frozen=True)
class FakeProfile:
    estimated_incremental_bytes: int
profiles = (FakeProfile(1), FakeProfile(profile.JOB_COMMITMENT), FakeProfile(profile.JOB_COMMITMENT))
model = object()
server = SimpleNamespace(address=('127.0.0.1', 32100), broker_id='c' * 32,
                         team_tokens={'alpha': 'a' * 64, 'beta': 'b' * 64})
events = []
session.check.side_effect = lambda actual: events.append(('lease-checked', actual))
def prepare(*args):
    assert events == [('lease-checked', lease)]
    events.append(('source', None))
    return source
def execute(actual_session, actual_lease, *, backend, registry, stop_event, announce):
    assert actual_session is session and actual_lease is lease
    assert isinstance(backend, gpu_service._TorchServiceBackend)
    assert backend.live_keys(identity) == live.keys
    assert backend.resolve_profiles(registry, live.keys) == profiles
    assert backend.verify_loaded_model(model) is True
    assert backend.verify_device_model(model, identity, session, lease) is True
    announce(server)
    return SimpleNamespace(status='clean', teardown_verified=True, error_type=None)
with patch.object(service, '_check_owner_contract') as gate, \
     patch.object(OwnerSession, 'from_environment', return_value=session), \
     patch.object(DeviceLease, 'acquire', return_value=lease), \
     patch.object(profile, 'prepare_trusted_source', side_effect=prepare), \
     patch.object(profile, 'build_trusted_service_identity', return_value=live), \
     patch.object(profile, 'resolve_trusted_service_profiles', return_value=profiles) as resolve, \
     patch.object(profile, 'verify_trusted_cpu_model', return_value=True), \
     patch.object(profile, 'verify_trusted_device_tie', return_value=True), \
     patch.object(gpu_service, '_load_profile_registry', return_value=(registry, live.keys)), \
     patch.object(gpu_service._TorchServiceBackend, 'resolve_profiles', return_value=profiles), \
     patch.object(gpu_service, '_run_bound_owner', side_effect=execute):
    assert child._owner(config) == 0
    gate.assert_called_once_with(config.repo_root)
    assert resolve.call_count == 1
    assert session.send_startup.call_count == 1
assert 'torch' not in sys.modules
'''
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        repo = Path(__file__).resolve().parents[1]
        outcome = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-c', program,
                                  str(repo), sysconfig.get_paths()['purelib'], str(path)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)

    def test_measured_profiles_compose_with_fixed_backend_commitments(self):
        # Synthetic samples exercise the real helper and backend resolver. No
        # measured GPU profile or live GPU work is claimed by this CPU test.
        program = r'''
import builtins, hashlib, sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
verified = [False]
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.partition('.')[0] == 'torch' and not verified[0]:
        raise RuntimeError('Torch imported before owner verification')
    result = original_import(name, *args, **kwargs)
    if name == 'torch' and hasattr(result, 'cuda'):
        def forbidden_cuda_init(*args, **kwargs):
            raise RuntimeError('CUDA initialization attempted in CPU profile test')
        result.cuda._lazy_init = forbidden_cuda_init
    return result
builtins.__import__ = guarded_import
sys.path.extend((sys.argv[1], sys.argv[2], str(Path(sys.argv[1]) / 'tests')))
from looking_glass import gpu_service_process as child
from looking_glass import gpu_service, trusted_service_profile as profile, two_token_service as service
from looking_glass.device_identity import DeviceIdentity
from looking_glass.device_lease import DeviceLease
from looking_glass.resource_profile import ProfileRegistry
from looking_glass.supervisor import OwnerSession
import test_trusted_service_profile as fixtures
fixture = fixtures.TrustedServiceProfileTests()
live = fixture.make_identity()
measured = fixture.make_profiles(live)
pins = tuple(item.profile_id for item in measured)
registry = ProfileRegistry(measured, approved_profile_ids=pins)
assert measured[1].estimated_incremental_bytes < profile.JOB_COMMITMENT
assert measured[2].estimated_incremental_bytes < profile.JOB_COMMITMENT
config = child._parse_config(Path(sys.argv[3]).read_bytes())
config = replace(config, profiles=tuple((*row[:3], pin) for row, pin in zip(config.profiles, pins)))
source = SimpleNamespace(source_base=b'source', asset_manifest=b'assets')
config = replace(config, source_base_sha256=hashlib.sha256(source.source_base).hexdigest(),
                 asset_manifest_sha256=hashlib.sha256(source.asset_manifest).hexdigest())
identity = object.__new__(DeviceIdentity)
object.__setattr__(identity, 'workspace_root', config.workspace_root)
object.__setattr__(identity, 'gpu_uuid', fixtures.UUID)
lease = object()
session = Mock(identity=identity, device_lease_path=config.workspace_root / 'device.lock')
session.check.side_effect = lambda actual: verified.__setitem__(0, True)
def execute(actual_session, actual_lease, *, backend, registry, stop_event, announce):
    assert backend.live_keys(identity) == live.keys
    resolved = backend.resolve_profiles(registry, live.keys)
    assert resolved[0].profile_id == pins[0]
    assert tuple(item.profile_id for item in resolved[1:]) == pins[1:]
    assert tuple(item.estimated_incremental_bytes for item in resolved[1:]) == (
        profile.JOB_COMMITMENT, profile.JOB_COMMITMENT)
    return SimpleNamespace(status='clean', teardown_verified=True, error_type=None)
with patch.object(service, '_check_owner_contract') as gate, \
     patch.object(OwnerSession, 'from_environment', return_value=session), \
     patch.object(DeviceLease, 'acquire', return_value=lease), \
     patch.object(profile, 'prepare_trusted_source', return_value=source), \
     patch.object(profile, 'build_trusted_service_identity', return_value=live), \
     patch.object(gpu_service, '_load_profile_registry', return_value=(registry, live.keys)), \
     patch.object(gpu_service, '_run_bound_owner', side_effect=execute):
    assert child._owner(config) == 0
    gate.assert_called_once_with(config.repo_root)
assert verified[0]
'''
        path = self.root / 'config.json'
        path.write_bytes(self.data())
        repo = Path(__file__).resolve().parents[1]
        outcome = subprocess.run([sys._base_executable, '-B', '-S', '-P', '-c', program,
                                  str(repo), sysconfig.get_paths()['purelib'], str(path)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)


if __name__ == '__main__':
    unittest.main()
