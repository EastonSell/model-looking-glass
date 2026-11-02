"""CPU-only hard-exit integration for the source-bound timing owner.

The real supervisor, isolated bootstrap and owner wrapper execute in a child.
The first owner operation is replaced by a CPU-only admission/fault fixture;
this checks process shutdown and marker custody, not CUDA execution.
"""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import sysconfig

import pytest

from looking_glass.supervisor import pending_supervised_children, run_supervised


_ROOT = Path(__file__).resolve().parents[1]

# This runs before the isolated bootstrap without importing project or model
# modules. The trace intercepts the real owner wrapper and its first workload
# call, so the bootstrap's source checks and hard-exit branch remain actual.
_CHILD = r'''
import atexit, json, os, pathlib, runpy, sys, types, weakref

bootstrap, site, commit, git, directory, cache, study_run_id, phase, normal_exit = sys.argv[1:]
directory = pathlib.Path(directory)

def note(name, directory=directory):
    (directory / name).write_text('called', encoding='utf-8')

atexit.register(note, 'atexit-ran')
atexit.register(note, 'device-cleanup-ran')

class Buffer:
    def __del__(self, callback=note):
        callback('buffer-finalizer-ran')

if normal_exit == 'normal':
    # Negative control: replacing only the final process exit with normal
    # unwinding must make the registered shutdown callbacks observable.
    def unwind(code):
        raise SystemExit(code)
    os._exit = unwind

def exercise(caller_globals):
        from looking_glass.supervisor import OwnerSession
        from looking_glass.device_lease import DeviceLease
        from looking_glass.tap_repeated_admission import _Admission
        from looking_glass import tap_repeated_owner as operations_module

        session = OwnerSession.from_environment()
        lease = DeviceLease.acquire(session.device_lease_path, timeout=3)
        session.check(lease)
        gib = 1024 ** 3
        row = {'free_bytes': 6 * gib, 'total_bytes': 8 * gib,
               'allocated_bytes': 128 * 1024 ** 2, 'reserved_bytes': 512 * 1024 ** 2}
        admission = _Admission(check_owner=lambda: session.check(lease), snapshot=lambda: dict(row),
                               total_bytes=8 * gib, cap_bytes=2 * gib)
        admission.acquire()
        if phase == 'completed':
            import torch
            import transformers
            if torch.cuda.is_initialized() or str(torch.__version__) != '2.8.0+cu128':
                raise AssertionError('the CPU-only dependency import was not isolated')
            if str(transformers.__version__) != '4.57.3':
                raise AssertionError('the pinned Transformers import differs')

        custody = caller_globals['_CUSTODY']
        stage = {'after_admission': 'model_setup', 'after_operation': 'samples', 'completed': 'attested'}[phase]
        custody.update(directory=directory, session=session, device_lease=lease, admission=admission,
                       stage=stage)
        if phase != 'completed':
            value = Buffer()
            weakref.finalize(value, note, 'device-finalizer-ran')
        if phase == 'after_operation':
            class FakeIdentity:
                gpu_uuid = 'GPU-cpu-test'
                workspace_root = 'cpu-test'
            class FakeLease:
                pass
            class FakeSession:
                def __init__(self, bound):
                    self.identity = FakeIdentity()
                    self._bound_lease = bound
                    self._runtime_verified = True
                def check(self, candidate):
                    if candidate is not self._bound_lease:
                        raise RuntimeError('wrong fake lease')
                    session.check(lease)
            fake_lease = FakeLease()
            fake_session = FakeSession(fake_lease)
            operations_module._OPERATIONS_ENABLED = True
            operations_module.OwnerSession = FakeSession
            operations_module.DeviceLease = FakeLease
            operations_module.DeviceIdentity = FakeIdentity
            operations = operations_module._OwnerOperations(fake_session, fake_lease, admission)
            custody['operations'] = operations
            returned = operations.call(lambda: value, root='output')
            if returned is not value or operations.roots.output is not value:
                raise AssertionError('the assigned operation return was not rooted')
            journal = caller_globals['_Journal'](directory / 'partial.jsonl')
            journal.append({'stage': 'cpu-fake-1'})
            journal.append({'stage': 'cpu-fake-2'})
            custody['journal'] = journal
            custody['lean'] = types.SimpleNamespace(partial={
                'stage': 'running', 'mode': 'direct', 'started_ns': 42,
                'private_untrusted_key': 'must not appear', 'ended_ns': 2 ** 70})
        elif phase == 'after_admission':
            custody['pending_return'] = value
        proof = {'run_id': session.run_id, 'pid': os.getpid(), 'phase': phase,
                 'active_lease': admission.budget.active_leases() == (admission.lease,),
                 'trace_bytes': admission.lease.trace_bytes('gpu'),
                 'cuda_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'torch_imported': 'torch' in sys.modules}
        (directory / 'proof.json').write_text(json.dumps(proof), encoding='utf-8')
        if phase == 'completed':
            admission.lease.release_trace('gpu', 128 * 1024)
            admission.lease.release()
            session.attest_clean(lease)
            return 0
        if phase == 'after_operation':
            raise KeyboardInterrupt('CPU-only injected interruption')
        raise RuntimeError('CPU-only injected failure')

def trace(frame, event, arg):
    if event != 'call' or frame.f_globals.get('__name__') != 'looking_glass.tap_repeated_gpu_owner':
        return trace
    if frame.f_code.co_name == 'owner_main':
        frame.f_globals['_OWNER_ENABLED'] = True
        if phase == 'completed':
            caller_globals = frame.f_globals
            frame.f_globals['_execute_owner'] = lambda *_args: exercise(caller_globals)
    elif frame.f_code.co_name == '_execute_owner':
        sys.settrace(None)
        exercise(frame.f_globals)
    return trace

sys.settrace(trace)
sys.argv = [bootstrap, '--bootstrap-site', site, '--source-commit', commit,
            '--git-executable', git, '--owner', str(directory), cache,
            study_run_id, commit, git]
runpy.run_path(bootstrap, run_name='__main__')
'''


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:3] != (3, 11, 15),
                    reason='the source bootstrap requires Windows and pinned CPython 3.11.15')
@pytest.mark.parametrize(('phase', 'normal_exit'), (
    ('after_admission', False), ('after_operation', False),
    ('after_admission', True), ('after_operation', True), ('completed', False)))
def test_owner_failure_reaps_child_without_shutdown_or_device_cleanup(tmp_path, phase, normal_exit):
    if phase == 'completed':
        try:
            pinned = (metadata.version('torch') == '2.8.0+cu128'
                      and metadata.version('transformers') == '4.57.3')
        except metadata.PackageNotFoundError:
            pinned = False
        if not pinned:
            pytest.skip('pinned CPU dependency imports are unavailable')
    bootstrap = _ROOT / 'scripts' / 'tap_repeated_gpu_bootstrap.py'
    owner = _ROOT / 'looking_glass' / 'tap_repeated_gpu_owner.py'
    assert bootstrap.is_file() and owner.is_file(), 'integrated source-bound owner is not available'
    git = shutil.which('git')
    assert git is not None
    commit = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=_ROOT, text=True).strip()
    site = Path(sysconfig.get_paths()['purelib']).resolve(strict=True)
    prefix, cache, run_dir = (tmp_path / item for item in ('pycache', 'cache', 'run'))
    for item in (prefix, cache, run_dir):
        item.mkdir()
    # This is the real parent-side Git/blob verification. It runs before the
    # supervised Job starts; the child must only hash the resulting manifest.
    bootstrap_api = runpy.run_path(str(bootstrap), run_name='_test_bootstrap_support')
    git_path = Path(git).resolve(strict=True)
    paths = bootstrap_api['_verify_source'](_ROOT, commit, git_path)
    source_files = {}
    for relative in paths:
        entry = subprocess.check_output([git, 'ls-tree', commit, '--', relative], cwd=_ROOT).split(b' ', 1)[0]
        assert entry in (b'100644', b'100755')
        source_files[relative] = {'mode': entry.decode('ascii'),
                                  'sha256': hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()}
    study_run_id = 'b' * 32
    manifest = {'schema': 'tap-repeated-source-manifest-v1', 'run_id': study_run_id,
                'source_commit': commit, 'source_files': source_files}
    manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    protected = run_dir / 'protected-manifest'
    protected.mkdir()
    (protected / 'source-manifest.json').write_bytes(manifest_raw)
    device, marker = tmp_path / 'fake-device.lock', tmp_path / 'fake-device.unsafe.json'
    from scripts.v2_source_guards import WindowsJob
    class ObservedJob(WindowsJob):
        def terminate_and_wait(self, timeout=5):
            self.before_stop = self.process_counts()
            return super().terminate_and_wait(timeout=timeout)
    job = ObservedJob()
    env = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
           if key in os.environ}
    env['LOOKING_GLASS_TAP_REPEATED_MANIFEST_SHA256'] = hashlib.sha256(manifest_raw).hexdigest()
    uname = platform.uname()
    fields = {key: getattr(uname, key) for key in
              ('system', 'node', 'release', 'version', 'machine', 'processor')}
    env['LOOKING_GLASS_TAP_REPEATED_PLATFORM_UNAME'] = json.dumps(fields, sort_keys=True, separators=(',', ':'))
    command = [sys._base_executable, '-I', '-S', '-B', '-X', f'pycache_prefix={prefix}', '-c', _CHILD,
               str(bootstrap), str(site), commit, str(Path(git).resolve()), str(run_dir), str(cache),
               study_run_id, phase, 'normal' if normal_exit else 'hard']
    result = run_supervised(command, device_lease_path=device, marker_path=marker, timeout=90,
                            cwd=_ROOT, env=env, inherit_environment=False,
                            process_job=job, require_zero_exit=True)
    assert result.clean is (phase == 'completed') and result.timed_out is False
    assert result.exit_code == (0 if phase == 'completed' else 1)
    assert job.before_stop == (1, 0)
    proof = json.loads((run_dir / 'proof.json').read_text(encoding='utf-8'))
    assert proof == {'run_id': result.run_id, 'pid': result.pid, 'phase': phase, 'active_lease': True,
                     'trace_bytes': 128 * 1024, 'cuda_visibility': '',
                     'torch_imported': phase == 'completed'}
    if phase == 'completed':
        assert not marker.exists()
    else:
        assert json.loads(marker.read_text(encoding='utf-8'))['run_id'] == result.run_id
    assert not marker.with_name(marker.name + '.attestation').exists()
    assert not any(record.run_id == result.run_id for record in pending_supervised_children())
    if phase == 'completed':
        assert not (run_dir / 'failure.json').exists()
    else:
        failure = json.loads((run_dir / 'failure.json').read_text(encoding='utf-8'))
        expected = {'schema': 'tap-repeated-failure-v1', 'status': 'unverified',
                    'stage': 'samples' if phase == 'after_operation' else 'model_setup',
                    'error_type': 'KeyboardInterrupt' if phase == 'after_operation' else 'RuntimeError'}
        if phase == 'after_operation':
            expected.update(journal_rows_fsynced_before_failure=2, journal_close_confirmed=False,
                            partial={'source': 'lean', 'stage': 'running', 'mode': 'direct', 'started_ns': 42})
            assert len((run_dir / 'partial.jsonl').read_bytes().splitlines()) == 2
        assert failure == expected
    for name in ('atexit-ran', 'device-cleanup-ran', 'device-finalizer-ran', 'buffer-finalizer-ran'):
        assert (run_dir / name).exists() is (normal_exit and phase != 'completed'), name
    assert not (run_dir / 'candidate.json').exists()
    assert not (run_dir / 'final.json').exists()
