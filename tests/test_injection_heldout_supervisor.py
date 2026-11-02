"""Fabricated Windows process-only checks; no split or model."""
import os
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys

import pytest

from looking_glass import natural_heldout_supervisor as supervisor
from scripts import number_cpu_outer, v2_source_guards
from tools import injection_heldout_owner as owner


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object required')


def audit_no_tensor(directory, exit_code, reaped):
    assert exit_code == 0 and reaped is True
    return {'status': 'evidence_complete', 'inventory': {'retained_tensor_bytes': 0}}


def test_supervisor_accepts_finished_helpers_and_restores_the_guard(tmp_path):
    assert hasattr(owner, 'supervise_with_cpu_job'), 'injection supervisor adapter is missing'
    base = str(Path(sys._base_executable).resolve(strict=True))
    code = ("import subprocess,sys; assert sys.stdin.buffer.readline()==b'MLG-natural-discovery-start-v1\\n'; "
            "subprocess.run([sys.executable,'-I','-S','-B','-c','pass'],check=True,creationflags=subprocess.DETACHED_PROCESS)")
    original = v2_source_guards.WindowsJob
    result = owner.supervise_with_cpu_job([base, '-I', '-S', '-B', '-c', code], tmp_path / 'owner',
        dict(os.environ), audit_result=audit_no_tensor, supervisor=supervisor,
        guard_module=v2_source_guards, cpu_job_type=number_cpu_outer.WindowsJob, timeout=30)
    assert v2_source_guards.WindowsJob is original
    assert result['status'] == 'evidence_complete' and result['exit_code'] == 0
    assert result['child_reaped'] and result['job_drained'] and result['job_closed']
    assert result['postexit_active_processes'] == 0


def test_live_descendant_at_owner_exit_is_incomplete_and_never_audited(tmp_path):
    assert hasattr(owner, 'supervise_with_cpu_job'), 'injection supervisor adapter is missing'
    base = str(Path(sys._base_executable).resolve(strict=True))
    code = ("import subprocess,sys; assert sys.stdin.buffer.readline()==b'MLG-natural-discovery-start-v1\\n'; "
            "subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time; time.sleep(60)'])")
    calls = []
    def audit(*args):
        calls.append(args)
        return audit_no_tensor(*args)
    result = owner.supervise_with_cpu_job([base, '-I', '-S', '-B', '-c', code], tmp_path / 'owner',
        dict(os.environ), audit_result=audit, supervisor=supervisor,
        guard_module=v2_source_guards, cpu_job_type=number_cpu_outer.WindowsJob, timeout=30)
    assert result['status'] == 'incomplete' and not calls
    assert type(result['postexit_active_processes']) is int and result['postexit_active_processes'] > 0
    assert result['job_drained'] and result['job_closed'] and result['child_reaped']


def test_interrupted_supervisor_never_mutates_the_shared_guard(tmp_path):
    from types import ModuleType
    original = v2_source_guards.WindowsJob
    fake = ModuleType('_fabricated_supervisor')
    fake.__dict__['guard'] = v2_source_guards
    fake.__dict__['original'] = original
    exec("def supervise_heldout(*args, **kwargs):\n"
         "    from scripts.v2_source_guards import WindowsJob\n"
         "    assert WindowsJob is not original\n"
         "    assert guard.WindowsJob is original\n"
         "    raise KeyboardInterrupt()\n", fake.__dict__)
    with pytest.raises(KeyboardInterrupt):
        owner.supervise_with_cpu_job(['fake'], tmp_path / 'owner', {}, audit_result=audit_no_tensor,
            supervisor=fake, guard_module=v2_source_guards, cpu_job_type=number_cpu_outer.WindowsJob)
    assert v2_source_guards.WindowsJob is original


def test_source_bootstrap_rejects_changed_bytes_before_execution(tmp_path):
    assert hasattr(owner, 'load_verified_source'), 'source-first loader is missing'
    marker = tmp_path / 'marker'
    source = tmp_path / 'fixture.py'
    source.write_text('open(%r, "w").write("executed")\n' % str(marker), encoding='utf-8')
    with pytest.raises(ValueError):
        owner.load_verified_source(source, '0' * 64, '_fabricated_untrusted_injection_source')
    assert not marker.exists()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    owner.load_verified_source(source, digest, '_fabricated_verified_injection_source')
    assert marker.read_text() == 'executed'
    sys.modules.pop('_fabricated_verified_injection_source', None)


def test_source_bootstrap_rejects_relative_path_before_execution(tmp_path, monkeypatch):
    source = tmp_path / 'fixture.py'
    marker = tmp_path / 'marker'
    source.write_text('open(%r, "w").write("executed")\n' % str(marker), encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        owner.load_verified_source(Path('fixture.py'), digest, '_fabricated_relative_source')
    assert not marker.exists()
    assert '_fabricated_relative_source' not in sys.modules


def test_frozen_source_inventory_rechecks_files_and_untracked_package_entries(tmp_path):
    assert hasattr(owner, 'verify_frozen_inventory'), 'frozen source inventory verifier is missing'
    root = tmp_path / 'core'
    package = root / 'looking_glass'
    package.mkdir(parents=True)
    (package / '__init__.py').write_bytes(b'')
    (package / 'example.py').write_bytes(b'VALUE = 1\n')
    files = {f'looking_glass/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest()
             for p in package.iterdir()}
    record = {'git_commit': owner.SOURCE_COMMIT, 'git_tree': 'b' * 40, 'files': files}
    raw = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    expected = hashlib.sha256(raw).hexdigest()
    assert owner.verify_frozen_inventory(root, raw, expected) == record
    (package / 'example.py').write_bytes(b'VALUE = 2\n')
    with pytest.raises(ValueError):
        owner.verify_frozen_inventory(root, raw, expected)
    (package / 'example.py').write_bytes(b'VALUE = 1\n')
    (package / 'extra.py').write_bytes(b'')
    with pytest.raises(ValueError):
        owner.verify_frozen_inventory(root, raw, expected)


def test_shared_gate_requires_the_expected_root_and_lock_identity(tmp_path):
    assert hasattr(owner, 'verify_gate_identity'), 'shared gate identity check is missing'
    root = tmp_path / 'gate'
    root.mkdir()
    lock = root / 'gate.lock'
    lock.write_bytes(b'')
    root_stat, lock_stat = root.lstat(), lock.lstat()
    root_id = (root_stat.st_dev, root_stat.st_ino)
    lock_id = (lock_stat.st_dev, lock_stat.st_ino)
    assert owner.verify_gate_identity(root, expected_root=root, root_identity=root_id, lock_identity=lock_id)
    with pytest.raises(ValueError):
        owner.verify_gate_identity(root, expected_root=tmp_path, root_identity=root_id, lock_identity=lock_id)
    with pytest.raises(ValueError):
        owner.verify_gate_identity(root, expected_root=root, root_identity=root_id, lock_identity=(lock_id[0], lock_id[1] + 1))


def test_protected_empty_prefix_prevents_loading_matching_poisoned_bytecode(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    source = root / 'fabricated_import.py'
    source.write_text("VALUE = 'evil'\n", encoding='ascii')
    stamp = source.stat()
    py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text("VALUE = 'safe'\n", encoding='ascii')
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    base = str(Path(sys._base_executable).resolve(strict=True))
    load = "import sys;sys.path.insert(0,sys.argv[1]);import fabricated_import;print(fabricated_import.VALUE)"
    control = subprocess.run([base, '-I', '-S', '-B', '-c', load, str(root)],
                             capture_output=True, check=True, text=True)
    assert control.stdout.strip() == 'evil', 'fixture must demonstrate valid poisoned in-tree bytecode'
    child = ("import sys;assert sys.stdin.buffer.readline()==b'MLG-natural-discovery-start-v1\\n';" + load)
    guard = v2_source_guards.PrivateEmptyCache(tmp_path)
    with v2_source_guards.SourceFileLease(root, [source]) as source_lease, guard as prefix:
        with v2_source_guards.InheritableHandles(source_lease.handles() + guard.handles()) as handles:
            result = owner.supervise_with_cpu_job(
                [base, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix), '-c', child, str(root)],
                tmp_path / 'protected-owner', dict(os.environ), audit_result=audit_no_tensor,
                supervisor=supervisor, guard_module=v2_source_guards,
                cpu_job_type=number_cpu_outer.WindowsJob, timeout=30, inherited_handles=handles)
        assert not any(prefix.iterdir())
    assert result['status'] == 'evidence_complete' and result['postexit_active_processes'] == 0
    assert (tmp_path / 'protected-owner' / 'stdout.txt').read_text().strip() == 'safe'
