import os
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from looking_glass import tap_repeated_gpu as controller


def test_disabled_controller_never_reaches_device_work(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, '_ENABLED', False)
    with pytest.raises(RuntimeError, match='disabled'):
        controller.run_session(workspace_root=tmp_path, cache_dir=tmp_path, output=tmp_path / 'result',
            session=0, cpu_context='unknown', coordinated_slot=True)


def test_workspace_requires_the_single_canonical_work_ancestor(tmp_path):
    repo = tmp_path / 'work' / 'frozen' / 'repo'
    repo.mkdir(parents=True)
    assert controller._workspace_root(repo, tmp_path) == tmp_path.resolve()
    nested = repo / 'work' / 'another'
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match='one canonical'):
        controller._workspace_root(nested, tmp_path)
    with pytest.raises(ValueError, match='workspace'):
        controller._workspace_root(repo, tmp_path / 'work')


def test_child_environment_is_allowlisted_and_offline(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_TOKEN', 'private-sentinel')
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', 'unexpected')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'other-device')
    uname = {'system': 'Windows', 'node': 'private-node', 'release': '11',
             'version': '10.0.26200', 'machine': 'AMD64', 'processor': 'private-processor'}
    env = controller._child_environment(tmp_path, 'GPU-test', 2, 'a' * 64, 'b' * 64, uname, tmp_path / 'temp')
    assert 'HF_TOKEN' not in env and 'PYTORCH_ALLOC_CONF' not in env
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert env['LOOKING_GLASS_TAP_REPEATED_SESSION'] == '2'
    assert env['LOOKING_GLASS_TAP_REPEATED_MANIFEST_SHA256'] == 'b' * 64
    assert env['TEMP'] == env['TMP'] == str(tmp_path / 'temp')
    assert 'private-node' in env['LOOKING_GLASS_TAP_REPEATED_PLATFORM_UNAME']


def test_controller_telemetry_requires_programfiles_directory(monkeypatch, tmp_path):
    monkeypatch.delenv('PROGRAMFILES', raising=False)
    with pytest.raises(RuntimeError, match='PROGRAMFILES'):
        controller._require_smi_environment()
    monkeypatch.setenv('PROGRAMFILES', 'relative-directory')
    with pytest.raises(RuntimeError, match='PROGRAMFILES'):
        controller._require_smi_environment()
    monkeypatch.setenv('PROGRAMFILES', str(tmp_path))
    controller._require_smi_environment()


def test_resource_maxima_include_the_preflight_peak():
    rows = [{'stage': 'preflight', 'peak_allocated_bytes': 99, 'peak_reserved_bytes': 111,
             'memory_before': {'free_bytes': 400}},
            *({'stage': 'measured', 'mode': mode, 'guarded_wall_ns': 10,
               'peak_allocated_bytes': 20, 'peak_reserved_bytes': 30,
               'memory_before': {'free_bytes': 500}} for mode in ('none', 'direct', 'gradient'))]
    summary = controller._resource_summary(rows, {'free_bytes': 300}, {'free_bytes': 350})
    assert summary['maximum_recorded_reset_peak_allocated_bytes'] == 99
    assert summary['maximum_recorded_reset_peak_reserved_bytes'] == 111
    assert summary['minimum_free_bytes_in_admission_pretransfer_and_journal'] == 300


def test_resource_minimum_uses_postcheck_instead_of_unvalidated_duplicate():
    rows = [{'stage': 'preflight', 'peak_allocated_bytes': 20, 'peak_reserved_bytes': 30,
             'memory_before': {'free_bytes': 500}, 'memory_after': {'free_bytes': 1}},
            {'stage': 'postcheck', 'memory_after': {'free_bytes': 450}},
            *({'stage': 'measured', 'mode': mode, 'guarded_wall_ns': 10,
               'peak_allocated_bytes': 20, 'peak_reserved_bytes': 30,
               'memory_before': {'free_bytes': 500}} for mode in ('none', 'direct', 'gradient'))]
    summary = controller._resource_summary(rows, {'free_bytes': 600}, {'free_bytes': 550})
    assert summary['minimum_free_bytes_in_admission_pretransfer_and_journal'] == 450


def test_private_record_is_create_only_and_canonical(tmp_path):
    path = tmp_path / 'source-manifest.json'
    digest = controller._write_new(path, {'z': 1, 'a': True})
    assert path.read_bytes() == b'{"a":true,"z":1}\n'
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        controller._write_new(path, {'replacement': True})
    assert path.read_bytes() == b'{"a":true,"z":1}\n'


@pytest.mark.parametrize('suffix', ('', '.private.json', '.partial', '.incomplete.json'))
def test_result_name_and_all_related_records_must_be_fresh(tmp_path, suffix):
    output = tmp_path / 'result.json'
    assert controller._check_output_destination(output) == output
    Path(str(output) + suffix).write_text('existing', encoding='utf-8')
    with pytest.raises(ValueError, match='related records'):
        controller._check_output_destination(output)


@pytest.mark.skipif(os.name != 'nt', reason='Windows create-only rename contract')
def test_result_is_published_create_only_and_keeps_complete_bytes(tmp_path):
    output = tmp_path / 'result.json'
    controller._publish_result(output, {'status': 'completed_provisional'})
    expected = b'{"status":"completed_provisional"}\n'
    assert output.read_bytes() == expected
    with pytest.raises(FileExistsError):
        controller._publish_result(output, {'status': 'replacement'})
    assert output.read_bytes() == expected
    assert Path(str(output) + '.partial').exists()


def test_interrupted_visible_result_gets_distinct_incomplete_evidence(monkeypatch, tmp_path):
    controller._ROOTS.clear()
    monkeypatch.setattr(controller, '_ENABLED', True)
    monkeypatch.setattr(controller, '_ENTRY_CLAIMED', False)
    output = tmp_path / 'result.json'
    success = b'{"status":"completed_provisional"}\n'

    def interrupt_after_publication(**_kwargs):
        controller._ROOTS.update(run_dir=tmp_path, stage='publishing')
        output.write_bytes(success)
        raise KeyboardInterrupt('private-sentinel')

    monkeypatch.setattr(controller, '_run_session', interrupt_after_publication)
    with pytest.raises(KeyboardInterrupt):
        controller.run_session(workspace_root=tmp_path, cache_dir=tmp_path, output=output,
            session=0, cpu_context='unknown', coordinated_slot=True)
    assert output.read_bytes() == success
    sidecar = Path(str(output) + '.incomplete.json').read_bytes()
    assert json.loads(sidecar)['status'] == 'incomplete'
    assert b'private-sentinel' not in sidecar
    assert (tmp_path / 'controller-failure.json').read_bytes() == sidecar
    controller._ROOTS.clear()


def test_failure_preserves_partial_evidence_without_claiming_a_completed_sample(monkeypatch, tmp_path):
    controller._ROOTS.clear()
    monkeypatch.setattr(controller, '_ENABLED', True)
    monkeypatch.setattr(controller, '_ENTRY_CLAIMED', False)
    output = tmp_path / 'result.json'
    journal = tmp_path / 'journal.jsonl'
    rows = [{'stage': 'preflight', 'observation_id': 'preflight'},
            {'stage': 'postcheck', 'observation_id': 'preflight', 'status': 'passed'},
            {'stage': 'warmup', 'observation_id': 'warmup:0:0'}]
    journal.write_bytes(b''.join(controller._canonical(row) for row in rows))
    def fail(**_kwargs):
        controller._ROOTS.update(run_dir=tmp_path, stage='supervised_owner')
        raise RuntimeError('private-sentinel')
    monkeypatch.setattr(controller, '_run_session', fail)
    with pytest.raises(RuntimeError, match='private-sentinel'):
        controller.run_session(workspace_root=tmp_path, cache_dir=tmp_path, output=output,
            session=0, cpu_context='unknown', coordinated_slot=True)
    result = json.loads(output.read_bytes())
    assert result['status'] == 'incomplete' and result['unvalidated_present_pairs'] == 1
    assert b'private-sentinel' not in output.read_bytes()
    assert (tmp_path / 'controller-failure.json').read_bytes() == output.read_bytes()
    with pytest.raises(RuntimeError, match='single-use'):
        controller.run_session(workspace_root=tmp_path, cache_dir=tmp_path, output=tmp_path / 'second.json',
            session=0, cpu_context='unknown', coordinated_slot=True)
    assert not (tmp_path / 'second.json').exists()
    controller._ROOTS.clear()


@pytest.mark.skipif(not shutil.which('git'), reason='Git is unavailable')
def test_parent_source_manifest_uses_committed_regular_blob_bytes(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    path = repo / 'source.py'
    path.write_bytes(b'first\n')
    git = shutil.which('git')
    def run(*args):
        return subprocess.run([git, *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()
    run('init', '-q')
    run('config', 'core.autocrlf', 'false')
    run('add', '--', 'source.py')
    run('-c', 'user.name=Probe', '-c', 'user.email=probe@example.invalid', 'commit', '-qm', 'source')
    commit = run('rev-parse', 'HEAD')
    monkeypatch.setattr(controller, '_SOURCE_PATHS', ('source.py',))
    tree, files = controller._source_identity(repo, git, commit)
    assert len(tree) == 40 and files['source.py']['mode'] == '100644'
    run('update-index', '--assume-unchanged', 'source.py')
    path.write_bytes(b'changed\n')
    assert not run('status', '--porcelain', '--untracked-files=all')
    with pytest.raises(RuntimeError, match='source bytes'):
        controller._source_identity(repo, git, commit)
