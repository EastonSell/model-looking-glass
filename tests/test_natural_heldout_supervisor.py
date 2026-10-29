import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from looking_glass import natural_heldout_supervisor as supervisor


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object')


@pytest.fixture(autouse=True)
def isolated_process_quarantine(monkeypatch):
    # Each test explicitly reaps or closes its own fabricated objects.
    monkeypatch.setattr(supervisor, '_QUARANTINED', [])


def _child(output, *, tail=''):
    root = Path(__file__).resolve().parents[1]
    code = ("import pathlib,sys; sys.path.insert(0,sys.argv[1]); "
            "from looking_glass.natural_agreement_controller import await_start; "
            "await_start(); d=pathlib.Path(sys.argv[2])/'attempt'/'tensors'; "
            "d.mkdir(parents=True); (d/'payload').write_bytes(b'partial'); " + tail)
    return [sys._base_executable, '-I', '-c', code, str(root), str(output)]


def _complete(attempt, code, reaped):
    assert code == 0 and reaped is True
    assert (attempt / 'tensors/payload').read_bytes() == b'partial'
    return {'status': 'evidence_complete', 'inventory': {'retained_tensor_bytes': 7}}


def test_success_audits_after_reap_and_rejects_overwrite(tmp_path):
    output = tmp_path / 'run'
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=_complete)
    assert result['status'] == 'evidence_complete'
    assert type(result['child_pid']) is int and 0 < result['child_pid'] <= 0xffffffff
    assert result['cleanup_verified'] is True and result['child_reaped'] is True
    assert result['job_drained'] is True and result['job_closed'] is True
    assert result['retry_safe'] is False and result['scientific_validation'].startswith('pending')
    assert result['retained_tensor_files']['total_bytes'] == 7
    assert json.loads((output / 'supervisor.json').read_text()) == result
    with pytest.raises(FileExistsError):
        supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=_complete)


def test_restricted_source_handles_are_inherited_by_the_gated_owner(tmp_path):
    from scripts.v2_source_guards import InheritableHandles, SourceFileLease
    source_root = tmp_path / 'source'
    source_root.mkdir()
    source = source_root / 'module.py'
    source.write_text('fabricated source')
    output = tmp_path / 'protected-run'
    with SourceFileLease(source_root, [source]) as lease:
        with InheritableHandles(lease.handles()) as inherited:
            code = ("import ctypes,json,pathlib,sys; sys.path.insert(0,sys.argv[1]); "
                    "from looking_glass.natural_agreement_controller import await_start; await_start(); "
                    "k=ctypes.WinDLL('kernel32',use_last_error=True); "
                    "k.GetHandleInformation.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)); "
                    "flags=ctypes.c_ulong(); "
                    "assert all(k.GetHandleInformation(int(h),ctypes.byref(flags)) for h in json.loads(sys.argv[3])); "
                    "d=pathlib.Path(sys.argv[2])/'attempt'/'tensors'; d.mkdir(parents=True); "
                    "(d/'payload').write_bytes(b'partial')")
            root = Path(__file__).resolve().parents[1]
            command = [sys._base_executable, '-I', '-c', code, str(root), str(output), json.dumps(inherited)]
            result = supervisor.supervise_heldout(command, output, os.environ.copy(), audit_result=_complete,
                                                   inherited_handles=inherited)
            assert result['status'] == 'evidence_complete' and result['cleanup_verified'] is True
            assert all(not os.get_handle_inheritable(handle) for handle in inherited)


@pytest.mark.parametrize('kind', ['job', 'process', 'stream', 'pump'])
def test_each_resource_is_rooted_before_its_initializer(kind, tmp_path, monkeypatch):
    custody = {'job': None, 'child': None, 'streams': [], 'pumps': []}
    created = []
    class Interrupted:
        def __init__(self, *args, **kwargs):
            created.append(self)
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        if kind == 'job':
            supervisor._new_job(custody, Interrupted)
        elif kind == 'process':
            monkeypatch.setattr(supervisor.subprocess, 'Popen', Interrupted)
            supervisor._spawn(custody, ['fabricated'])
        elif kind == 'stream':
            monkeypatch.setattr(supervisor.io, 'FileIO', Interrupted)
            supervisor._open_log(custody, tmp_path / 'log')
        else:
            monkeypatch.setattr(supervisor.threading, 'Thread', Interrupted)
            supervisor._start_pump(custody, lambda: None, ())
    stored = {'job': custody['job'], 'process': custody['child'],
              'stream': custody['streams'][0] if custody['streams'] else None,
              'pump': custody['pumps'][0] if custody['pumps'] else None}
    assert stored[kind] is created[0]


def test_nonzero_exit_retains_partial_and_never_audits(tmp_path):
    output = tmp_path / 'failed'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output, tail='sys.exit(7)'), output, os.environ.copy(),
                                          audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['exit_code'] == 7
    assert result['cleanup_verified'] is True and result['retained_tensor_files']['total_bytes'] == 7
    assert (output / 'attempt/tensors/payload').read_bytes() == b'partial'


def test_live_descendant_after_owner_exit_is_drained_but_never_audited(tmp_path):
    output = tmp_path / 'live-descendant'
    tail = ("import subprocess; subprocess.Popen([sys._base_executable,'-c','import time; time.sleep(30)'], "
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, "
            "creationflags=subprocess.DETACHED_PROCESS)")
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output, tail=tail), output, os.environ.copy(),
                                          audit_result=audit)
    audit.assert_not_called()
    assert result['exit_code'] == 0 and result['postexit_active_processes'] >= 1
    assert result['status'] == 'incomplete' and result['cleanup_verified'] is True
    assert 'active Job members' in result['supervisor_error']


def test_unreadable_postexit_job_count_never_audits(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job = guards.WindowsJob
    class UnreadableCount:
        def __init__(self):
            self.real = real_job()
        def assign(self, child):
            self.real.assign(child)
        def _accounting(self):
            raise OSError('fabricated count failure')
        def terminate_and_wait(self):
            self.real.terminate_and_wait()
        def close(self):
            self.real.close()
    monkeypatch.setattr(guards, 'WindowsJob', UnreadableCount)
    audit = Mock()
    output = tmp_path / 'unknown-postexit-count'
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['postexit_active_processes'] is None
    assert result['cleanup_verified'] is True and 'post-exit Job count' in result['supervisor_error']


def test_timeout_reaps_and_preserves_partial_without_audit(tmp_path):
    output = tmp_path / 'timeout'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output, tail='import time; time.sleep(30)'), output,
                                          os.environ.copy(), timeout=1, audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['cleanup_verified'] is True
    assert result['supervisor_error'] == 'TimeoutExpired'
    assert result['retained_tensor_files']['total_bytes'] == 7


def test_log_overflow_is_bounded_and_never_audits(tmp_path):
    output = tmp_path / 'overflow'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output, tail="sys.stdout.write('x'*3000000); sys.stdout.flush()"),
                                          output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['log_overflow'] is True
    assert (output / 'stdout.txt').stat().st_size <= 1024 * 1024


def test_drain_failure_closes_job_and_skips_audit_and_live_files(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job = guards.WindowsJob
    jobs = []

    class FailedDrain:
        def __init__(self):
            self.real = real_job()
            self.closed = False
            jobs.append(self)
        def assign(self, child):
            self.real.assign(child)
        def terminate_and_wait(self):
            raise OSError('simulated drain failure')
        def close(self):
            self.real.close()
            self.closed = True

    monkeypatch.setattr(guards, 'WindowsJob', FailedDrain)
    output = tmp_path / 'uncertain'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(),
                                          audit_result=audit)
    audit.assert_not_called()
    assert jobs[0].closed and result['child_reaped'] is True
    assert result['status'] == 'incomplete' and result['cleanup_verified'] is False
    assert result['retained_tensor_files']['accounting_complete'] is False
    assert result['retry_safe'] is False and 'unverified' in result['protection_state']


def test_audit_failure_and_mismatched_inventory_fail_closed(tmp_path):
    output = tmp_path / 'bad-audit'
    def bad(*args):
        raise ValueError('private case text must not appear in the supervisor')
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=bad)
    assert result['status'] == 'incomplete' and result['audit_error_type'] == 'ValueError'
    assert 'private case text' not in (output / 'supervisor.json').read_text()
    second = tmp_path / 'bad-count'
    result = supervisor.supervise_heldout(_child(second), second, os.environ.copy(),
                                          audit_result=lambda *args: {'status':'evidence_complete',
                                                                    'inventory': {'retained_tensor_bytes': 8}})
    assert result['status'] == 'incomplete' and 'accounting' in result['reason']


def test_interrupted_child_start_is_rooted_and_never_becomes_complete(tmp_path, monkeypatch):
    actual_spawn = supervisor._spawn
    children = []
    def interrupted(custody, *args, **kwargs):
        children.append(actual_spawn(custody, *args, **kwargs))
        raise RuntimeError('simulated interrupted return')
    monkeypatch.setattr(supervisor, '_spawn', interrupted)
    output = tmp_path / 'lost'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(),
                                          audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['child_reaped'] is True
    assert result['child_pid'] == children[0].pid
    assert result['cleanup_verified'] is False and result['retry_safe'] is False
    child = children[0]
    assert child.returncode != 0
    assert not (output / 'attempt').exists()
    child.stdout.close()
    child.stderr.close()


def test_unknown_process_initializer_outcome_cannot_produce_a_complete_receipt(tmp_path, monkeypatch):
    class PartialChild:
        stdin = None
        pid = None
        def poll(self):
            raise OSError('native process handle was not established')
    partial = PartialChild()
    def interrupted(custody, *args, **kwargs):
        custody['child'] = partial
        raise KeyboardInterrupt()
    monkeypatch.setattr(supervisor, '_spawn', interrupted)
    output = tmp_path / 'unknown-initializer'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['child_reaped'] is False
    assert result['cleanup_verified'] is False and result['quarantined'] is True
    assert supervisor._QUARANTINED[-1]['child'] is partial


def test_pump_close_failure_prevents_structural_audit(tmp_path, monkeypatch):
    actual_spawn = supervisor._spawn
    failures = []
    class CloseFailure:
        def __init__(self, source):
            self.source = source
            failures.append(self)
        def read1(self, size):
            return self.source.read1(size)
        def close(self):
            raise OSError('simulated pipe close failure')
    def start(custody, *args, **kwargs):
        child = actual_spawn(custody, *args, **kwargs)
        child.stdout = CloseFailure(child.stdout)
        return child
    monkeypatch.setattr(supervisor, '_spawn', start)
    audit = Mock()
    output = tmp_path / 'pipe-close'
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['log_errors'] == ['OSError']
    assert result['cleanup_verified'] is False and result['quarantined'] is True
    for stream in supervisor._QUARANTINED[-1]['streams']:
        stream.close()
    failures[0].source.close()


def test_cancellation_is_distinct_after_confirmed_cleanup(tmp_path, monkeypatch):
    actual_spawn = supervisor._spawn
    def start(custody, *args, **kwargs):
        child = actual_spawn(custody, *args, **kwargs)
        wait = child.wait
        first = [True]
        def interrupted(*a, **k):
            if first[0]:
                first[0] = False
                raise KeyboardInterrupt()
            return wait(*a, **k)
        child.wait = interrupted
        return child
    monkeypatch.setattr(supervisor, '_spawn', start)
    audit = Mock()
    output = tmp_path / 'cancelled'
    result = supervisor.supervise_heldout(_child(output, tail='import time; time.sleep(30)'), output,
                                          os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'cancelled' and result['cancellation_requested'] is True
    assert result['cleanup_verified'] is True and result['quarantined'] is False


def test_interrupted_drain_keeps_uncertainty_and_cancellation(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job = guards.WindowsJob
    class InterruptedDrain:
        def __init__(self):
            self.real = real_job()
        def assign(self, child):
            self.real.assign(child)
        def terminate_and_wait(self):
            self.real.terminate_and_wait()
            raise KeyboardInterrupt()
        def close(self):
            self.real.close()
    monkeypatch.setattr(guards, 'WindowsJob', InterruptedDrain)
    audit = Mock()
    output = tmp_path / 'interrupted-drain'
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['cancellation_requested'] is True
    assert result['cleanup_verified'] is False and result['quarantined'] is True
    assert supervisor._QUARANTINED[-1]['output'] == output


def test_failed_job_close_remains_rooted_for_diagnosis(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job = guards.WindowsJob
    jobs = []
    class FailedClose:
        def __init__(self):
            self.real = real_job()
            jobs.append(self)
        def assign(self, child):
            self.real.assign(child)
        def terminate_and_wait(self):
            self.real.terminate_and_wait()
        def close(self):
            raise OSError('simulated close failure')
    monkeypatch.setattr(guards, 'WindowsJob', FailedClose)
    audit = Mock()
    output = tmp_path / 'failed-close'
    try:
        result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
        audit.assert_not_called()
        assert result['status'] == 'incomplete' and result['cleanup_verified'] is False
        assert result['quarantined'] is True and supervisor._QUARANTINED[-1]['job'] is jobs[0]
        next_output = tmp_path / 'must-not-start'
        with pytest.raises(RuntimeError, match='custody'):
            supervisor.supervise_heldout(_child(next_output), next_output, os.environ.copy(), audit_result=audit)
        assert not next_output.exists()
    finally:
        # The injected failure is known, not an ambiguous real CloseHandle.
        jobs[0].real.close()


def test_interrupted_pump_join_keeps_streams_rooted_and_unread(tmp_path, monkeypatch):
    original_join = supervisor.threading.Thread.join
    first = [True]
    def interrupted(thread, *args, **kwargs):
        if first[0]:
            first[0] = False
            raise KeyboardInterrupt()
        return original_join(thread, *args, **kwargs)
    monkeypatch.setattr(supervisor.threading.Thread, 'join', interrupted)
    output = tmp_path / 'join-interrupted'
    audit = Mock()
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['status'] == 'incomplete' and result['cancellation_requested'] is True
    assert result['logs_complete'] is False and result['cleanup_verified'] is False
    assert result['stdout_sha256'] is None and result['stderr_sha256'] is None
    assert result['retained_tensor_files']['accounting_complete'] is False
    custody = supervisor._QUARANTINED[-1]
    assert all(not stream.closed for stream in custody['streams'])
    # The interrupted join is injected; complete its actual cleanup in test.
    for thread in custody['pumps']:
        original_join(thread, timeout=5)
    for stream in custody['streams']:
        stream.close()


def test_job_and_log_close_double_failure_keeps_custody(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job, real_open = guards.WindowsJob, supervisor._open_log
    jobs, wrappers = [], []
    class FailedJobClose:
        def __init__(self):
            self.real = real_job()
            jobs.append(self)
        def assign(self, child):
            self.real.assign(child)
        def terminate_and_wait(self):
            self.real.terminate_and_wait()
        def close(self):
            raise OSError('injected job close')
    class FailedStreamClose:
        def __init__(self, stream):
            self.stream = stream
            wrappers.append(self)
        def write(self, data):
            return self.stream.write(data)
        def flush(self):
            return self.stream.flush()
        def fileno(self):
            return self.stream.fileno()
        def close(self):
            raise OSError('injected stream close')
    def open_with_failure(custody, path):
        stream = real_open(custody, path)
        if path.name == 'stdout.txt':
            wrapper = FailedStreamClose(stream)
            custody['streams'][-1] = wrapper
            return wrapper
        return stream
    monkeypatch.setattr(guards, 'WindowsJob', FailedJobClose)
    monkeypatch.setattr(supervisor, '_open_log', open_with_failure)
    output = tmp_path / 'double-fault'
    audit = Mock()
    try:
        result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
        audit.assert_not_called()
        assert result['status'] == 'incomplete' and result['quarantined'] is True
        assert result['logs_complete'] is False and result['job_closed'] is False
        assert (output / 'supervisor.json').is_file()
        assert supervisor._QUARANTINED[-1]['job'] is jobs[0]
        assert wrappers[0] in supervisor._QUARANTINED[-1]['streams']
    finally:
        jobs[0].real.close()
        for wrapper in wrappers:
            wrapper.stream.close()


def test_supervisor_record_failure_retains_custody_and_blocks_next_owner(tmp_path, monkeypatch):
    real_open = Path.open
    def fail_record(path, *args, **kwargs):
        if path.name == 'supervisor.json':
            raise OSError('injected record failure')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_record)
    output = tmp_path / 'missing-terminal'
    with pytest.raises(OSError, match='record failure'):
        supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=_complete)
    assert not (output / 'supervisor.json').exists()
    assert supervisor._QUARANTINED[-1]['output'] == output
    next_output = tmp_path / 'must-not-start'
    with pytest.raises(RuntimeError, match='custody'):
        supervisor.supervise_heldout(_child(next_output), next_output, os.environ.copy(), audit_result=_complete)
    assert not next_output.exists()


@pytest.mark.parametrize('exit_code', [0, 7])
def test_external_preparation_is_accepted_only_after_clean_process_exit(tmp_path, exit_code):
    preparation = tmp_path / 'private-preparation'
    output = tmp_path / 'preparation-supervisor'
    code = ("import pathlib,sys; token=sys.stdin.buffer.readline(); "
            "assert token == b'MLG-natural-discovery-start-v1\\n'; "
            "p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "(p/'claim.json').write_bytes(b'claim'); "
            "(p/'preparation-complete.json').write_bytes(b'complete'); "
            "(p/'commitment.json').write_bytes(b'commitment'); "
            f"sys.exit({exit_code})")
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in
                [('claim.json', b'claim'), ('preparation-complete.json', b'complete'),
                 ('commitment.json', b'commitment')]}
    calls = []
    def audit_preparation(unused_attempt, code, reaped):
        calls.append((code, reaped))
        observed = {name: hashlib.sha256((preparation / name).read_bytes()).hexdigest() for name in expected}
        assert observed == expected
        assert not (preparation / 'failure.json').exists()
        return {'status':'evidence_complete', 'inventory':{'retained_tensor_bytes':0},
                'preparation':observed}
    result = supervisor.supervise_heldout([sys._base_executable, '-c', code, str(preparation)], output,
                                          os.environ.copy(), audit_result=audit_preparation)
    assert result['cleanup_verified'] is True and result['exit_code'] == exit_code
    if exit_code == 0:
        assert calls == [(0, True)] and result['status'] == 'evidence_complete'
        assert result['audit']['preparation'] == expected
    else:
        assert calls == [] and result['status'] == 'incomplete'
        assert (preparation / 'preparation-complete.json').read_bytes() == b'complete'


def test_malformed_pid_prevents_audit_after_real_job_cleanup(tmp_path, monkeypatch):
    import scripts.v2_source_guards as guards
    real_job = guards.WindowsJob
    class InvalidPidAfterDrain:
        def __init__(self):
            self.real = real_job()
        def assign(self, owner):
            self.owner = owner
            self.real.assign(owner)
        def terminate_and_wait(self):
            self.real.terminate_and_wait()
            self.owner.pid = True
        def close(self):
            self.real.close()
    monkeypatch.setattr(guards, 'WindowsJob', InvalidPidAfterDrain)
    audit = Mock()
    output = tmp_path / 'bad-pid'
    result = supervisor.supervise_heldout(_child(output), output, os.environ.copy(), audit_result=audit)
    audit.assert_not_called()
    assert result['cleanup_verified'] is True and result['status'] == 'incomplete'
    assert result['child_pid'] is None and 'invalid child PID' in result['supervisor_error']
