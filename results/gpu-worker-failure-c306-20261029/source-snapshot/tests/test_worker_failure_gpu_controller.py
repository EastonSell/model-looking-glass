"""CPU-only controller checks; no device is queried or launched."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from looking_glass.worker_failure_gpu_controller import (_RecordingJob, _source_identity,
                                                         _workspace_root, run_controller)


def test_supervisor_job_records_both_prelaunch_and_postexit_counts():
    class Job:
        def __init__(self):
            self.values = iter(((0, 0), (1, 0)))
        def process_counts(self):
            return next(self.values)
        def assign(self, process):
            pass
        def terminate_and_wait(self, timeout=5):
            pass
        def close(self):
            pass
    job = _RecordingJob(Job())
    assert job.process_counts() == (0, 0)
    assert job.process_counts() == (1, 0)
    assert job.counts == [(0, 0), (1, 0)]


def test_workspace_rejects_nested_work_lease_namespace(tmp_path):
    root = tmp_path / 'workspace'
    repo = root / 'work' / 'teams' / 'pilot'
    repo.mkdir(parents=True)
    if any(parent.name.casefold() == 'work' for parent in root.parents):
        with pytest.raises(ValueError):
            _workspace_root(repo, root)
    else:
        assert _workspace_root(repo, root) == root.resolve()
    actual_repo = Path(__file__).resolve().parents[1]
    ancestors = [parent for parent in actual_repo.parents if parent.name.casefold() == 'work']
    if len(ancestors) == 1 and ancestors[0].name == 'work':
        assert _workspace_root(actual_repo, ancestors[0].parent) == ancestors[0].parent
    nested = repo / 'work' / 'other'
    nested.mkdir(parents=True)
    with pytest.raises(ValueError):
        _workspace_root(nested, root)


def test_source_bytes_are_checked_even_when_git_hides_a_modified_file(tmp_path, monkeypatch):
    import looking_glass.worker_failure_gpu_controller as controller
    git = shutil.which('git')
    if not git:
        pytest.skip('Git unavailable')
    git = str(Path(git).resolve())
    source = tmp_path / 'source.py'
    source.write_text('value = 1\n', encoding='utf8')
    def invoke(*args):
        return subprocess.run([git, *args], cwd=tmp_path, check=True, capture_output=True).stdout
    invoke('init', '-q')
    invoke('config', 'core.autocrlf', 'false')
    invoke('add', 'source.py')
    invoke('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    commit = invoke('rev-parse', 'HEAD').decode().strip()
    monkeypatch.setattr(controller, '_SOURCE_PATHS', ('source.py',))
    _, files = _source_identity(tmp_path, Path(git), commit)
    assert files['source.py']['mode'] == '100644'
    invoke('update-index', '--assume-unchanged', 'source.py')
    source.write_text('value = 2\n', encoding='utf8')
    assert not invoke('status', '--porcelain', '--untracked-files=all')
    with pytest.raises(RuntimeError):
        _source_identity(tmp_path, Path(git), commit)


def test_disabled_controller_does_not_create_evidence(tmp_path, monkeypatch):
    import looking_glass.worker_failure_gpu_controller as controller
    monkeypatch.setattr(controller, '_ENABLED', False)
    target = tmp_path / 'result.json'
    with pytest.raises(RuntimeError):
        run_controller(workspace_root=tmp_path, output=target, coordinated_slot=True)
    assert not os.path.lexists(target)


def test_controller_publishes_sanitized_incomplete_after_reserved_output(tmp_path, monkeypatch):
    import json
    import looking_glass.worker_failure_gpu_controller as controller
    output = tmp_path / 'result.json'
    state = {'directory': None, 'output': None, 'stage': 'entry', 'failure': None, 'diagnostic': None}
    monkeypatch.setattr(controller, '_ROOTS', state)
    monkeypatch.setattr(controller, '_ENABLED', True)
    def fail(*args):
        state['output'] = output
        state['stage'] = 'device_preflight'
        raise OSError('private message')
    monkeypatch.setattr(controller, '_run_controller', fail)
    with pytest.raises(OSError):
        controller.run_controller(workspace_root=tmp_path, output=output, coordinated_slot=True)
    record = json.loads(output.read_text())
    assert record == {'schema': 'worker-failure-controller-incomplete-v1', 'status': 'incomplete',
                      'stage': 'device_preflight', 'error_type': 'OSError'}
    assert 'private message' not in output.read_text()


def test_isolated_module_dispatches_controller_help_without_loading_torch():
    repo = Path(__file__).resolve().parents[1]
    code = ("import runpy,sys; sys.path.insert(0,sys.argv[1]); "
            "sys.argv=['worker_failure_gpu','--help']; "
            "runpy.run_module('looking_glass.worker_failure_gpu',run_name='__main__')")
    child = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code, str(repo)],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 0
    assert '--workspace-root' in child.stdout and '--check-only' in child.stdout
