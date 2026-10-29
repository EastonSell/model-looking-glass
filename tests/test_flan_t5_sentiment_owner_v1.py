"""Fabricated custody and process tests; no prospective rows or model calls."""
import copy
from collections import namedtuple
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from looking_glass import flan_t5_sentiment_owner_v1 as owner
from looking_glass import flan_t5_sentiment_selector_v1 as selector


_REAL_DISK_USAGE = shutil.disk_usage


@pytest.fixture(autouse=True)
def _fabricated_free_space(monkeypatch):
    usage = namedtuple('DiskUsage', 'total used free')
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage(8 * 1024**3, 4 * 1024**3, 4 * 1024**3))


def _selector_fixture():
    path = Path(__file__).with_name('test_flan_t5_sentiment_selector_v1.py')
    spec = importlib.util.spec_from_file_location('selector_fabricated_fixture', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _operator():
    path = Path(__file__).resolve().parents[1] / 'tools/run_flan_t5_sentiment_intervention_v1.py'
    spec = importlib.util.spec_from_file_location('sentiment_operator_fabricated', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result._load_stage_helper = lambda root: SimpleNamespace(
        _dependency_site=lambda venv: (Path(venv), 'a' * 64),
        _check_child_identity=lambda path, venv, pid: True)
    return result


def _tool(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / name
    spec = importlib.util.spec_from_file_location('fabricated_' + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fabricated_custody(tmp_path, monkeypatch):
    fixture = _selector_fixture()
    pool, tokenizer, exposures = fixture.pool(), fixture.FakeTokenizer(), fixture.source()
    pool_salt, selection_salt = bytes(range(32)), bytes(range(32, 64))
    pool_digest = selector.pool_commitment(pool, pool_salt)
    selection = selector.select(pool, tokenizer, exposures, pool_digest)
    selection_digest = selector.selection_commitment(selection, selection_salt)
    pool_card = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_digest,
                 'protocol_sha256': pool['protocol_sha256'], 'runner_revision': pool['runner_revision'],
                 'asset_manifest_sha256': pool['asset_manifest_sha256']}
    selection_card = {'schema': 'flan-sentiment-selection-card-v1', 'commitment': selection_digest,
                      'pool_commitment': pool_digest, 'protocol_sha256': pool['protocol_sha256'],
                      'runner_revision': pool['runner_revision'], 'asset_manifest_sha256': pool['asset_manifest_sha256']}
    versions = {'python': 'test', 'transformers': 'test', 'tokenizers': 'test'}
    stage = {'schema': 'flan-t5-sentiment-selector-stage-runtime-v1', 'stage': 'select',
             'status': 'completed', 'selector_revision': pool['runner_revision'],
             'source_sha256': {name: 'a' * 64 for name in owner.SELECTOR_SOURCE_PATHS},
             'versions': versions, 'pool_commitment': pool_digest, 'selection_commitment': selection_digest,
             'scope': 'Same-process version observation; external direct-exit and custody checks remain required.'}
    stage_raw = selector.canonical(stage) + b'\n'
    logs = {name: {'retained_bytes': 0, 'truncated': False, 'error_type': None, 'finished': True,
                   'sha256': hashlib.sha256(b'').hexdigest()} for name in ('stdout', 'stderr')}
    for name in ('stdout', 'stderr'):
        (tmp_path / (name + '.log')).write_bytes(b'')
    exit_record = {'schema': 'flan-t5-sentiment-selector-stage-exit-v1', 'stage': 'select',
                   'status': 'completed', 'stage_receipt_sha256': hashlib.sha256(stage_raw).hexdigest(),
                   'reaped': True, 'exit_code': 0, 'timed_out': False, 'exception_type': None,
                   'selector_revision': pool['runner_revision'], 'child_prefix_empty_after': True,
                   'logs': logs, 'scope': 'Local direct-child observation; no execution authentication.'}
    exit_raw = selector.canonical(exit_record) + b'\n'
    review = {'schema': owner.REVIEW_SCHEMA, 'status': 'consistent', 'pool_commitment': pool_digest,
              'selection_commitment': selection_digest, 'protocol_sha256': pool['protocol_sha256'],
              'runner_revision': pool['runner_revision'], 'asset_manifest_sha256': pool['asset_manifest_sha256'],
              'unit_count': 12, 'token_rows': 96, 'selection_count': 4, 'enough_selected': True,
              'source_count': 1, 'source_rows': 0, 'screen_runtime': versions,
              'limitations': ['fabricated observation'], 'replay_script_sha256': 'f' * 64,
              'replay_source_revision': 'c' * 40, 'replay_source_path': 'tools/fake-replay.py',
              'selector_source_revision': pool['runner_revision'],
              'screen_runtime_receipt_sha256': hashlib.sha256(stage_raw).hexdigest(),
              'screen_exit_receipt_sha256': hashlib.sha256(exit_raw).hexdigest()}
    scientific_pin = {'schema': 'flan-t5-sentiment-scientific-reader-pin-v1', 'revision': 'd' * 40,
                      'sha256': {name: 'a' * 64 for name in owner.SCIENTIFIC_READER_PATHS}}
    data = {'pool': selector.canonical(pool), 'pool_salt': pool_salt,
            'pool_card': selector.canonical(pool_card) + b'\n', 'pool_observation': b'{}',
            'selection': selector.canonical(selection), 'selection_salt': selection_salt,
            'selection_card': selector.canonical(selection_card) + b'\n', 'selection_observation': b'{}',
            'screen_runtime_receipt': stage_raw,
            'screen_exit_observation': exit_raw,
            'screen_review': selector.canonical(review),
            'scientific_reader_pin': selector.canonical(scientific_pin) + b'\n'}
    paths = {}
    for name, raw in data.items():
        paths[name] = tmp_path / name
        paths[name].write_bytes(raw)
    monkeypatch.setattr(selector, 'load_exposures', lambda *a, **k: exposures)
    monkeypatch.setattr(selector, '_require_commit', lambda *a, **k: None)
    monkeypatch.setattr(owner, '_check_stage_source', lambda *a, **k: None)
    monkeypatch.setattr(owner, '_check_replay_source', lambda *a, **k: None)
    monkeypatch.setattr(owner, '_check_scientific_reader', lambda *a, **k: None)
    return paths, pool, selection


def test_custody_binds_actual_selector_output_and_both_salted_cards(tmp_path, monkeypatch):
    paths, pool, selection = fabricated_custody(tmp_path, monkeypatch)
    observed = []
    binding = owner.bind_custody(tmp_path, pool['runner_revision'], paths,
                                 observe=lambda repo, raw, observation, kind: observed.append(kind))
    assert observed == ['pool', 'selection']
    assert binding['pool'] == pool and binding['selection'] == selection
    assert len(selection['unit_screens']) == 12 and len(selection['selected_units']) == 4


@pytest.mark.parametrize('change', ['noncanonical-pool', 'noncanonical-selection', 'wrong-salt',
                                     'wrong-card', 'bad-review-count', 'wrong-runner',
                                     'stage-nonzero-exit', 'stage-log-drift'])
def test_custody_rejects_changed_or_incomplete_evidence(tmp_path, monkeypatch, change):
    paths, pool, selection = fabricated_custody(tmp_path, monkeypatch)
    if change == 'noncanonical-pool':
        paths['pool'].write_text(json.dumps(pool, indent=2), encoding='utf8')
    elif change == 'noncanonical-selection':
        paths['selection'].write_bytes(paths['selection'].read_bytes() + b'\n')
    elif change == 'wrong-salt':
        paths['selection_salt'].write_bytes(b'x' * 32)
    elif change == 'wrong-card':
        card = selector.strict_json(paths['selection_card'].read_bytes())
        card['commitment'] = 'f' * 64
        paths['selection_card'].write_bytes(selector.canonical(card) + b'\n')
    elif change == 'bad-review-count':
        review = selector.strict_json(paths['screen_review'].read_bytes())
        review['token_rows'] = 95
        paths['screen_review'].write_bytes(selector.canonical(review))
    elif change == 'wrong-runner':
        pool['runner_revision'] = 'b' * 40
        paths['pool'].write_bytes(selector.canonical(pool))
    elif change == 'stage-nonzero-exit':
        observed = selector.strict_json(paths['screen_exit_observation'].read_bytes())
        observed['exit_code'] = 1
        paths['screen_exit_observation'].write_bytes(selector.canonical(observed) + b'\n')
    elif change == 'stage-log-drift':
        (tmp_path / 'stderr.log').write_bytes(b'drift')
    with pytest.raises(ValueError):
        owner.bind_custody(tmp_path, 'a' * 40, paths, observe=lambda *args: None)


def test_observation_checks_the_actual_committed_card_bytes(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    raw = b'{"card":"fabricated"}\n'
    (repo / 'card.json').write_bytes(raw)
    subprocess.run(['git', 'add', 'card.json'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fabricated card'], cwd=repo, check=True)
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    observation = {'schema': 'flan-sentiment-pool-observation-v1', 'path': 'card.json',
                   'revision': rev, 'sha256': hashlib.sha256(raw).hexdigest()}
    owner._observe_card(repo, raw, observation, 'pool')
    changed = copy.deepcopy(observation)
    changed['sha256'] = hashlib.sha256(b'changed').hexdigest()
    with pytest.raises(ValueError):
        owner._observe_card(repo, raw, changed, 'pool')


def test_source_snapshot_ignores_inherited_git_directory_redirect(tmp_path, monkeypatch):
    repositories = [tmp_path / 'checkout-a', tmp_path / 'checkout-b']
    revisions = []
    for index, repo in enumerate(repositories):
        repo.mkdir()
        (repo / 'source.py').write_text('same source\n', encoding='utf8')
        subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
        subprocess.run(['git', 'add', 'source.py'], cwd=repo, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', f'fabricated {index}'], cwd=repo, check=True)
        revisions.append(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())
    assert revisions[0] != revisions[1]
    monkeypatch.setattr(owner, 'SOURCE_PATHS', ('source.py',))
    monkeypatch.setenv('GIT_DIR', str(repositories[1] / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(repositories[0]))
    with pytest.raises(ValueError, match='expected commit'):
        owner.source_snapshot(repositories[0], revisions[1])
    assert owner.source_snapshot(repositories[0], revisions[0])['revision'] == revisions[0]


def _operator_args(tmp_path):
    root = Path(_operator().__file__).resolve().parents[1]
    paths = {name: tmp_path / name for name in _operator().PATH_OPTIONS if name not in ('repo', 'attempt')}
    args = ['--repo', str(root), '--attempt', str(tmp_path / 'attempt'), '--revision', 'a' * 40]
    for name, path in paths.items():
        args.extend(['--' + name, str(path)])
    return args


@pytest.mark.parametrize('identity_matches', [True, False])
def test_operator_accepts_only_reaped_child_with_matching_terminal(tmp_path, monkeypatch, identity_matches):
    op = _operator()
    monkeypatch.setattr(op, '_load_stage_helper', lambda root: SimpleNamespace(
        _dependency_site=lambda venv: (Path(venv), 'a' * 64),
        _check_child_identity=lambda path, venv, pid: identity_matches))
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    fake_owner = SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision})
    monkeypatch.setattr(op, '_load_owner', lambda root, revision: fake_owner)
    attempt = tmp_path / 'attempt'
    class FakeChild:
        pid = 123
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def wait(self, timeout):
            child = attempt / 'child'
            child.mkdir()
            (child / 'events.jsonl').write_bytes(b'events')
            terminal = b'{"status":"complete_candidate"}'
            (child / 'terminal.json').write_bytes(terminal)
            receipt = {'status': 'complete_candidate', 'source_revision': 'a' * 40,
                       'terminal_sha256': hashlib.sha256(terminal).hexdigest(),
                       'events_sha256': hashlib.sha256(b'events').hexdigest()}
            (attempt / 'owner-receipt.json').write_text(json.dumps(receipt))
            return 0
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *a, **k: FakeChild())
    if identity_matches:
        op.main(_operator_args(tmp_path))
    else:
        with pytest.raises(RuntimeError, match='incomplete'):
            op.main(_operator_args(tmp_path))
    result = json.loads((attempt / 'operator-exit.json').read_text())
    assert result['direct_child_exit_code'] == 0 and result['child_reaped']
    assert result['terminal_consistent'] is identity_matches and not result['timed_out']


def test_operator_timeout_kills_reaps_and_retains_incomplete_receipt(tmp_path, monkeypatch):
    op = _operator()
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision:
                        SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
    class FakeChild:
        pid = 123
        killed = False
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired('fabricated', timeout)
            return 9
        def kill(self):
            self.killed = True
    child = FakeChild()
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *a, **k: child)
    with pytest.raises(RuntimeError, match='incomplete'):
        op.main(_operator_args(tmp_path))
    result = json.loads((tmp_path / 'attempt' / 'operator-exit.json').read_text())
    assert child.killed and result['child_reaped'] and result['timed_out']
    assert not result['terminal_consistent']


@pytest.mark.parametrize('selector_stage', [False, True])
def test_low_realistic_free_space_rejects_before_spawn(tmp_path, monkeypatch, selector_stage):
    op = _operator()
    root = Path(op.__file__).resolve().parents[1]
    revision = 'a' * 40
    if selector_stage:
        launcher = _tool('observe_flan_sentiment_selector_stage_v1.py')
        stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                                _dependency_site=lambda venv: (Path(venv), 'a' * 64))
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_bootstrap', lambda *args: None)
        monkeypatch.setattr(launcher, '_load', lambda path, name: stage if name == '_flan_stage_helper' else op)
        attempt = tmp_path / 'selector-stage'
        arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                     '--', 'select', '--repo', str(root)]
    else:
        launcher = op
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_load_owner', lambda root, revision:
                            SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
        attempt = tmp_path / 'attempt'
        arguments = _operator_args(tmp_path)
    usage = namedtuple('DiskUsage', 'total used free')
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage(4 * 1024**3, 2 * 1024**3 + 1, 2 * 1024**3 - 1))
    spawned = []
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: spawned.append(True))
    with pytest.raises(ValueError, match='less than 2 GiB free'):
        launcher.main(arguments)
    assert not spawned and not attempt.exists()


def test_log_drain_bounds_saved_bytes_and_still_consumes_the_pipe(tmp_path):
    op = _operator()
    original = b'x' * (op.LOG_LIMIT + 10000)
    pipe = io.BytesIO(original)
    state = {'retained_bytes': 0, 'truncated': False, 'error_type': None, 'finished': False}
    op._drain_log(pipe, tmp_path / 'output.log', state)
    assert (tmp_path / 'output.log').read_bytes() == original[:op.LOG_LIMIT]
    assert state == {'retained_bytes': op.LOG_LIMIT, 'truncated': True, 'error_type': None, 'finished': True}
    assert pipe.closed


def test_zero_exit_without_a_terminal_is_incomplete(tmp_path, monkeypatch):
    op = _operator()
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision:
                        SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
    class FakeChild:
        pid = 123
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *a, **k: FakeChild())
    with pytest.raises(RuntimeError, match='incomplete'):
        op.main(_operator_args(tmp_path))
    result = json.loads((tmp_path / 'attempt' / 'operator-exit.json').read_text())
    assert result['child_reaped'] and result['direct_child_exit_code'] == 0
    assert not result['terminal_consistent'] and result['terminal_sha256'] is None


def test_spawn_record_failure_kills_and_reaps_direct_child(tmp_path, monkeypatch):
    op = _operator()
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision:
                        SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
    class FakeChild:
        pid = 123
        killed = False
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def poll(self):
            return 9 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            return 9
    child = FakeChild()
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *a, **k: child)
    original = op._write_new
    def fail_spawn(path, value):
        if Path(path).name == 'operator-spawn.json':
            raise OSError('fabricated spawn receipt failure')
        return original(path, value)
    monkeypatch.setattr(op, '_write_new', fail_spawn)
    with pytest.raises(OSError, match='spawn receipt'):
        op.main(_operator_args(tmp_path))
    result = json.loads((tmp_path / 'attempt' / 'operator-exit.json').read_text())
    assert child.killed and result['child_reaped'] and result['direct_child_exit_code'] == 9
    assert result['operator_exception_type'] == 'OSError' and not result['terminal_consistent']


def test_selector_stage_records_same_process_versions_and_public_cards(tmp_path, monkeypatch):
    import tokenizers
    import transformers
    monkeypatch.setattr(transformers, '__file__', str(tmp_path / 'fake-site' / 'transformers.py'))
    monkeypatch.setattr(tokenizers, '__file__', str(tmp_path / 'fake-site' / 'tokenizers.py'))
    stage = _tool('run_flan_sentiment_selector_stage_v1.py')
    monkeypatch.setattr(stage, '_empty_prefix', lambda: None)
    monkeypatch.setattr(stage, '_receive_grant', lambda expected: None)
    monkeypatch.setattr(stage, '_dependency_site', lambda venv: (tmp_path / 'fake-site', 'a' * 64))
    monkeypatch.setattr(stage.sys, 'path', list(stage.sys.path))
    monkeypatch.setattr(stage, '_snapshot', lambda root, revision: {'fabricated': 'a' * 64})
    pool_card = tmp_path / 'pool-card.json'
    selected_card = tmp_path / 'selection-card.json'
    pool_card.write_text(json.dumps({'commitment': 'b' * 64}))
    selected_card.write_text(json.dumps({'commitment': 'c' * 64}))
    called = []
    class FakeLoader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            module.main = lambda: called.append(list(stage.sys.argv))
    spec = importlib.machinery.ModuleSpec('_fabricated_selector', FakeLoader())
    monkeypatch.setattr(stage.importlib.util, 'spec_from_file_location', lambda *a, **k: spec)
    receipt = tmp_path / 'stage.json'
    root = Path(stage.__file__).resolve().parents[1]
    stage.main(['--revision', 'a' * 40, '--stage-receipt', str(receipt), '--venv', str(tmp_path),
                '--grant-sha256', 'a' * 64, '--', 'select',
                '--repo', str(root), '--pool-card', str(pool_card), '--selection-card', str(selected_card)])
    value = json.loads(receipt.read_bytes())
    assert len(called) == 1 and called[0][1] == 'select'
    assert value['status'] == 'completed' and value['versions']['transformers'] == transformers.__version__
    assert value['versions']['tokenizers'] == tokenizers.__version__
    assert value['pool_commitment'] == 'b' * 64 and value['selection_commitment'] == 'c' * 64


def test_selector_stage_failure_is_retained_without_a_success_receipt(tmp_path, monkeypatch):
    stage = _tool('run_flan_sentiment_selector_stage_v1.py')
    monkeypatch.setattr(stage, '_empty_prefix', lambda: None)
    monkeypatch.setattr(stage, '_receive_grant', lambda expected: None)
    monkeypatch.setattr(stage, '_dependency_site', lambda venv: (tmp_path / 'fake-site', 'a' * 64))
    monkeypatch.setattr(stage.sys, 'path', list(stage.sys.path))
    monkeypatch.setattr(stage, '_snapshot', lambda root, revision: {'fabricated': 'a' * 64})
    class FakeLoader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            def fail():
                raise RuntimeError('fabricated selector failure')
            module.main = fail
    spec = importlib.machinery.ModuleSpec('_fabricated_selector', FakeLoader())
    monkeypatch.setattr(stage.importlib.util, 'spec_from_file_location', lambda *a, **k: spec)
    receipt = tmp_path / 'stage.json'
    root = Path(stage.__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match='fabricated selector'):
        stage.main(['--revision', 'a' * 40, '--stage-receipt', str(receipt), '--venv', str(tmp_path),
                    '--grant-sha256', 'a' * 64, '--', 'commit-pool',
                    '--repo', str(root), '--pool-card', str(tmp_path / 'unused')])
    value = json.loads(receipt.read_bytes())
    assert value['stage'] == 'commit-pool' and value['status'] == 'incomplete'
    assert value['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('identity_matches', [True, False])
def test_selector_observer_binds_the_reaped_child_and_stage_receipt(tmp_path, monkeypatch, identity_matches):
    observer = _tool('observe_flan_sentiment_selector_stage_v1.py')
    output = _operator()
    stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                            _dependency_site=lambda venv: (Path(venv), 'a' * 64),
                            _check_child_identity=lambda path, venv, pid: identity_matches)
    monkeypatch.setattr(observer, '_empty_prefix', lambda: None)
    monkeypatch.setattr(observer, '_load', lambda path, name: stage if name == '_flan_stage_helper' else output)
    root = Path(observer.__file__).resolve().parents[1]
    revision = 'a' * 40
    def fake_git(command, cwd):
        if command[-2:] == ['rev-parse', 'HEAD']:
            return (revision + '\n').encode()
        if command[-3:-1] == ['cat-file', '-t']:
            return b'commit\n'
        if command[-2] == 'show':
            return (root / command[-1].split(':', 1)[1]).read_bytes()
        raise AssertionError(command)
    monkeypatch.setattr(observer.subprocess, 'check_output', fake_git)
    attempt = tmp_path / 'selector-stage'
    class FakeChild:
        pid = 123
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'public stage status\n')
        stderr = io.BytesIO(b'')
        def wait(self, timeout):
            receipt = {'schema': 'flan-t5-sentiment-selector-stage-runtime-v1', 'stage': 'select',
                       'status': 'completed', 'selector_revision': revision}
            (attempt / 'stage-runtime.json').write_text(json.dumps(receipt))
            return 0
    monkeypatch.setattr(observer.subprocess, 'Popen', lambda *a, **k: FakeChild())
    arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                 '--', 'select', '--repo', str(root)]
    if identity_matches:
        observer.main(arguments)
    else:
        with pytest.raises(RuntimeError, match='incomplete'):
            observer.main(arguments)
    result = json.loads((attempt / 'stage-exit.json').read_bytes())
    assert result['status'] == ('completed' if identity_matches else 'incomplete')
    assert result['reaped'] and result['exit_code'] == 0
    assert result['stage_receipt_sha256'] == hashlib.sha256((attempt / 'stage-runtime.json').read_bytes()).hexdigest()
    assert result['logs']['stdout']['retained_bytes'] == len(b'public stage status\n')
    assert result['logs']['stderr']['finished'] and not result['logs']['stderr']['truncated']


@pytest.mark.parametrize('free_at_model_load', [3 * 1024**3, 2 * 1024**3 - 1])
def test_owner_retokenizes_verified_selection_before_loading_model_and_calls_bound_core(
        tmp_path, monkeypatch, free_at_model_load):
    import torch
    import transformers
    from looking_glass import flan_t5_feasibility, flan_t5_sentiment_runner

    paths, pool, selection = fabricated_custody(tmp_path, monkeypatch)
    original_bind = owner.bind_custody
    monkeypatch.setattr(owner, 'bind_custody', lambda repo, revision, supplied:
                        original_bind(repo, revision, supplied, observe=lambda *args: None))
    root = Path(owner.__file__).resolve().parents[1]
    monkeypatch.setattr(owner, 'source_snapshot', lambda repo, revision: {'revision': revision, 'sha256': {}})
    asset = tmp_path / 'assets'
    asset.mkdir()
    (asset / 'asset-manifest.json').write_bytes(b'fabricated')
    monkeypatch.setattr(flan_t5_feasibility, 'verify_assets', lambda path: {'files': {}})
    fixture = _selector_fixture()
    class T5TokenizerFast(fixture.FakeTokenizer):
        pass
    tokenizer = T5TokenizerFast()
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        lambda *args, **kwargs: calls.append('tokenizer') or tokenizer)
    class T5ForConditionalGeneration:
        def __init__(self):
            self.config = SimpleNamespace(d_model=512, vocab_size=32128, decoder_start_token_id=0,
                                          pad_token_id=0, eos_token_id=1, _attn_implementation='eager')
        def eval(self):
            return self
    monkeypatch.setattr(transformers.T5ForConditionalGeneration, 'from_pretrained',
                        lambda *args, **kwargs: (calls.append('model') or T5ForConditionalGeneration(), {}))
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda *a, **k: None)
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda value: None)
    monkeypatch.setattr(torch, 'manual_seed', lambda value: None)
    monkeypatch.setattr(torch, 'set_flush_denormal', lambda value: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}.items():
        monkeypatch.setenv(key, value)
    def fake_core(model, selected_tokenizer, groups, journal, *, custody_check):
        calls.append('core')
        assert selected_tokenizer is tokenizer and len(groups) == 8
        assert groups[0]['cases']['P1'] == selection['unit_screens'][0]['instructions'][0]['cases'][0]
        custody_check()
        journal.finish('complete_candidate', {'testing_only': False})
        return {'fake': True}
    monkeypatch.setattr(flan_t5_sentiment_runner, 'run_cpu_core', fake_core)
    usage = namedtuple('DiskUsage', 'total used free')
    space_checks = []
    def disk_usage(path):
        space_checks.append(path)
        available = 3 * 1024**3 if len(space_checks) == 1 else free_at_model_load
        return usage(4 * 1024**3, 4 * 1024**3 - available, available)
    monkeypatch.setattr(shutil, 'disk_usage', disk_usage)
    attempt = tmp_path / 'owner-attempt'
    attempt.mkdir()
    args = SimpleNamespace(repo=root, venv=Path(sys.prefix).resolve(), asset_dir=asset,
                           attempt=attempt, revision=pool['runner_revision'], **paths)
    if free_at_model_load < 2 * 1024**3:
        with pytest.raises(ValueError, match='free space before model loading'):
            owner.run(args)
        assert calls == ['tokenizer'] and len(space_checks) == 2
        assert json.loads((attempt / 'owner-receipt.json').read_bytes())['status'] == 'incomplete'
        assert json.loads((attempt / 'child' / 'terminal.json').read_bytes())['status'] == 'incomplete'
        return
    assert owner.run(args) == {'fake': True}
    assert len(space_checks) == 2
    assert calls == ['tokenizer', 'model', 'core']
    receipt = json.loads((attempt / 'owner-receipt.json').read_bytes())
    assert receipt['status'] == 'complete_candidate'
    assert json.loads((attempt / 'child' / 'terminal.json').read_bytes())['status'] == 'complete_candidate'


def test_invalid_custody_stops_before_tokenizer_or_model_loading(tmp_path, monkeypatch):
    import transformers
    paths, pool, selection = fabricated_custody(tmp_path, monkeypatch)
    paths['selection_salt'].write_bytes(b'x' * 32)
    original_bind = owner.bind_custody
    monkeypatch.setattr(owner, 'bind_custody', lambda repo, revision, supplied:
                        original_bind(repo, revision, supplied, observe=lambda *args: None))
    monkeypatch.setattr(owner, 'source_snapshot', lambda repo, revision: {'revision': revision, 'sha256': {}})
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: calls.append('tokenizer'))
    monkeypatch.setattr(transformers.T5ForConditionalGeneration, 'from_pretrained',
                        lambda *a, **k: calls.append('model'))
    for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}.items():
        monkeypatch.setenv(key, value)
    attempt = tmp_path / 'owner-attempt'
    attempt.mkdir()
    args = SimpleNamespace(repo=Path(owner.__file__).resolve().parents[1], asset_dir=tmp_path,
                           attempt=attempt, revision=pool['runner_revision'], **paths)
    with pytest.raises(ValueError, match='selection card'):
        owner.run(args)
    assert calls == []
    assert json.loads((attempt / 'owner-receipt.json').read_bytes())['status'] == 'incomplete'


@pytest.mark.parametrize('time_out', [False, True])
def test_real_isolated_stage_observer_records_exit_and_ignores_checkout_shadow(tmp_path, time_out):
    if _REAL_DISK_USAGE(tmp_path).free < 2 * 1024**3 + 256 * 1024**2:
        pytest.skip('real direct-child rehearsal requires the actual free-space reserve')
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / 'fixture-repo'
    repo.mkdir()
    paths = ['tools/observe_flan_sentiment_selector_stage_v1.py',
             'tools/run_flan_sentiment_selector_stage_v1.py',
             'tools/run_flan_t5_sentiment_intervention_v1.py']
    for name in paths:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    fake_launcher = repo / 'tools/run_flan_sentiment_selector_v1.py'
    fake_launcher.write_text(
        "import json, os, pathlib, sys, time\n"
        "def main():\n"
        "    if os.environ.get('FAKE_STAGE_TIMEOUT') == '1': time.sleep(60)\n"
        "    args = sys.argv\n"
        "    path = pathlib.Path(args[args.index('--selection-card') + 1])\n"
        "    path.write_text(json.dumps({'commitment':'c'*64}))\n", encoding='utf8')
    for name in ('looking_glass/__init__.py', 'looking_glass/flan_t5_sentiment_selector_v1.py',
                 'looking_glass/flan_t5_sentiment_screen.py', 'looking_glass/flan_t5_feasibility.py',
                 'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md', 'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md'):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fabricated\n', encoding='utf8')
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fabricated stage'], cwd=repo, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    foreign = tmp_path / 'foreign-git-repo'
    foreign.mkdir()
    (foreign / 'other.txt').write_text('other\n', encoding='utf8')
    subprocess.run(['git', 'init', '-q'], cwd=foreign, check=True)
    subprocess.run(['git', 'add', 'other.txt'], cwd=foreign, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'foreign'], cwd=foreign, check=True)
    sentinel = tmp_path / 'untrusted-module-executed'
    (repo / 'numbers.py').write_text(
        "from pathlib import Path\nPath(" + repr(str(sentinel)) + ").write_text('executed')\n",
        encoding='utf8')
    pool_card, selection_card = tmp_path / 'pool-card.json', tmp_path / 'selection-card.json'
    pool_card.write_text(json.dumps({'commitment': 'b' * 64}), encoding='utf8')
    prefix = tmp_path / 'observer-prefix'
    prefix.mkdir()
    attempt = tmp_path / 'stage-observation'
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(repo / paths[0]), '--revision', revision, '--attempt', str(attempt),
               '--venv', str(Path(sys.prefix).resolve()), '--timeout-seconds', '10' if time_out else '45', '--',
               'select', '--repo', str(repo), '--pool-card', str(pool_card),
               '--selection-card', str(selection_card)]
    environment = dict(os.environ, FAKE_STAGE_TIMEOUT='1' if time_out else '0',
                       GIT_DIR=str(foreign / '.git'), GIT_WORK_TREE=str(repo))
    result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, timeout=120, check=False)
    diagnostic = result.stderr.decode(errors='replace')
    if not attempt.exists() and 'less than 2 GiB free' in diagnostic:
        pytest.skip('actual free space fell below the reserve before stage allocation')
    for name in ('stage-exit.json', 'stderr.log'):
        path = attempt / name
        if path.is_file():
            diagnostic += '\n' + path.read_text(errors='replace')
    assert (result.returncode != 0) == time_out, diagnostic
    assert not sentinel.exists() and not any(prefix.iterdir())
    exit_record = json.loads((attempt / 'stage-exit.json').read_bytes())
    identity = json.loads((attempt / 'stdout.log').read_bytes().splitlines()[0])
    spawn = json.loads((attempt / 'stage-spawn.json').read_bytes())
    assert identity['pid'] == spawn['pid'] and identity['parent_pid'] == spawn['parent_pid']
    site_relative = ('Lib/site-packages' if os.name == 'nt' else
                     'lib/python%d.%d/site-packages' % sys.version_info[:2])
    expected_site = Path(sys.prefix).resolve() / site_relative
    assert identity['dependency_site_path_sha256'] == hashlib.sha256(str(expected_site.resolve()).encode()).hexdigest()
    assert identity['prefix_path_sha256'] == hashlib.sha256(str(Path(sys.base_prefix).resolve()).encode()).hexdigest()
    if time_out:
        assert exit_record['reaped'] and exit_record['timed_out'] and exit_record['status'] == 'incomplete'
        assert not (attempt / 'stage-runtime.json').exists()
        return
    runtime = json.loads((attempt / 'stage-runtime.json').read_bytes())
    assert runtime['stage'] == 'select' and runtime['status'] == 'completed'
    assert runtime['pool_commitment'] == 'b' * 64 and runtime['selection_commitment'] == 'c' * 64
    import tokenizers
    import transformers
    assert runtime['versions']['transformers'] == transformers.__version__
    assert runtime['versions']['tokenizers'] == tokenizers.__version__
    assert exit_record['reaped'] and exit_record['exit_code'] == 0 and exit_record['status'] == 'completed'
    assert exit_record['stage_receipt_sha256'] == hashlib.sha256((attempt / 'stage-runtime.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('time_out', [False, True])
def test_real_isolated_owner_operator_observes_the_working_child(tmp_path, time_out):
    if _REAL_DISK_USAGE(tmp_path).free < 2 * 1024**3 + 256 * 1024**2:
        pytest.skip('real direct-child rehearsal requires the actual free-space reserve')
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / 'owner-fixture-repo'
    repo.mkdir()
    for name in ('tools/run_flan_t5_sentiment_intervention_v1.py',
                 'tools/run_flan_sentiment_selector_stage_v1.py'):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    package = repo / 'looking_glass'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf8')
    (package / 'flan_t5_sentiment_owner_v1.py').write_text(
        "import hashlib, json, os, time\n"
        "from pathlib import Path\n"
        "def source_snapshot(root, revision): return {'revision': revision}\n"
        "def run(args):\n"
        "    if os.environ.get('FAKE_OWNER_TIMEOUT') == '1': time.sleep(60)\n"
        "    attempt = Path(args.attempt); child = attempt / 'child'; child.mkdir()\n"
        "    events = b'events'; terminal = b'{\\\"status\\\":\\\"complete_candidate\\\"}'\n"
        "    (child / 'events.jsonl').write_bytes(events)\n"
        "    (child / 'terminal.json').write_bytes(terminal)\n"
        "    receipt = {'status':'complete_candidate','source_revision':args.revision,\n"
        "               'terminal_sha256':hashlib.sha256(terminal).hexdigest(),\n"
        "               'events_sha256':hashlib.sha256(events).hexdigest()}\n"
        "    (attempt / 'owner-receipt.json').write_text(json.dumps(receipt))\n", encoding='utf8')
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fabricated owner'], cwd=repo, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    prefix = tmp_path / 'outer-prefix'
    prefix.mkdir()
    attempt = tmp_path / 'owner-observation'
    paths = {name: tmp_path / name for name in _operator().PATH_OPTIONS}
    paths.update({'repo': repo, 'venv': Path(sys.prefix).resolve(), 'attempt': attempt})
    command = [sys._base_executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
               str(repo / 'tools/run_flan_t5_sentiment_intervention_v1.py'),
               '--revision', revision, '--timeout-seconds', '10']
    for name, path in paths.items():
        command.extend(['--' + name, str(path)])
    environment = dict(os.environ, FAKE_OWNER_TIMEOUT='1' if time_out else '0')
    result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, timeout=120, check=False)
    if not attempt.exists() and b'less than 2 GiB free' in result.stderr:
        pytest.skip('actual free space fell below the reserve before owner allocation')
    assert (result.returncode != 0) == time_out, result.stderr.decode(errors='replace')
    record = json.loads((attempt / 'operator-exit.json').read_bytes())
    identity = json.loads((attempt / 'stdout.log').read_bytes().splitlines()[0])
    spawn = json.loads((attempt / 'operator-spawn.json').read_bytes())
    assert identity['pid'] == spawn['pid'] and identity['parent_pid'] == spawn['parent_pid']
    assert record['child_reaped'] and record['timed_out'] is time_out
    assert record['terminal_consistent'] is (not time_out)


@pytest.mark.parametrize('tool_name', ['run_flan_t5_sentiment_intervention_v1.py',
                                       'run_flan_sentiment_selector_stage_v1.py'])
def test_actual_child_gate_waits_and_rejects_partial_grant_before_work(tmp_path, tool_name):
    tool = Path(__file__).resolve().parents[1] / 'tools' / tool_name
    sentinel = tmp_path / ('entered-' + tool_name)
    grant = bytes(range(32))
    code = (
        'import importlib.util,pathlib; '
        f"spec=importlib.util.spec_from_file_location('gate', {str(tool)!r}); "
        'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
        f"module._receive_grant({hashlib.sha256(grant).hexdigest()!r}); "
        f"pathlib.Path({str(sentinel)!r}).write_text('entered')"
    )
    child = subprocess.Popen([sys._base_executable, '-I', '-S', '-B', '-c', code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.5)
        assert not sentinel.exists()
        child.stdin.write(grant[:12])
        child.stdin.close()
        assert child.wait(timeout=15) != 0 and not sentinel.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
        child.stdout.close()
        child.stderr.close()
    completed = subprocess.run([sys._base_executable, '-I', '-S', '-B', '-c', code], input=grant,
                               capture_output=True, timeout=15, check=False)
    assert completed.returncode == 0 and sentinel.read_text() == 'entered'


def test_direct_interpreter_uses_the_selected_venv_without_site_startup(tmp_path):
    stage = Path(__file__).resolve().parents[1] / 'tools/run_flan_sentiment_selector_stage_v1.py'
    code = (
        'import importlib.util,json,sys; '
        f"spec=importlib.util.spec_from_file_location('direct_runtime', {str(stage)!r}); "
        'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
        'site,_=module._dependency_site(sys.argv[1]); sys.path.append(str(site)); '
        'import torch,transformers,tokenizers; '
        "print(json.dumps({'prefix':sys.prefix,'torch_origin':torch.__file__,"
        "'torch':torch.__version__,'transformers':transformers.__version__,"
        "'tokenizers':tokenizers.__version__,'cuda_initialized':torch.cuda.is_initialized()}))"
    )
    command = [sys._base_executable, '-I', '-S', '-B', '-c', code]
    completed = subprocess.run([*command, str(Path(sys.prefix).resolve())], capture_output=True,
                               timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr.decode(errors='replace')
    value = json.loads(completed.stdout)
    site_relative = ('Lib/site-packages' if os.name == 'nt' else
                     'lib/python%d.%d/site-packages' % sys.version_info[:2])
    site = (Path(sys.prefix).resolve() / site_relative).resolve()
    assert Path(value['torch_origin']).resolve().is_relative_to(site)
    assert Path(value['prefix']).resolve() == Path(sys.base_prefix).resolve()
    assert value['torch'] and value['transformers'] and value['tokenizers']
    assert value['cuda_initialized'] is False
    wrong = tmp_path / 'wrong-venv'
    wrong.mkdir()
    (wrong / 'pyvenv.cfg').write_text(
        f'home = {tmp_path}\nimplementation = CPython\n'
        f'version_info = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n'
        'include-system-site-packages = false\n', encoding='utf8')
    rejected = subprocess.run([*command, str(wrong)], capture_output=True, timeout=30, check=False)
    assert rejected.returncode != 0 and b'venv does not match' in rejected.stderr


def test_operator_roots_child_before_interrupted_popen_constructor(tmp_path, monkeypatch):
    op = _operator()
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision:
                        SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
    class InterruptedPopen:
        last = None
        def __init__(self, *args, **kwargs):
            type(self).last = self
            self.pid = 123
            self.killed = False
            raise KeyboardInterrupt('fabricated lost constructor return')
        def poll(self):
            return 9 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            return 9
    monkeypatch.setattr(op.subprocess, 'Popen', InterruptedPopen)
    with pytest.raises(KeyboardInterrupt, match='constructor return'):
        op.main(_operator_args(tmp_path))
    record = json.loads((tmp_path / 'attempt' / 'operator-exit.json').read_text())
    assert InterruptedPopen.last.killed and record['child_reaped']
    assert record['operator_exception_type'] == 'KeyboardInterrupt' and not record['terminal_consistent']


def test_lost_grant_write_return_stops_and_reaps_known_child(tmp_path, monkeypatch):
    op = _operator()
    monkeypatch.setattr(op, '_empty_prefix', lambda: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision:
                        SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
    class LostWrite:
        delivered = None
        def write(self, value):
            self.delivered = value
            raise OSError('fabricated lost grant return')
    class FakeChild:
        pid = 123
        killed = False
        stdin = LostWrite()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def poll(self):
            return 9 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            return 9
    child = FakeChild()
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *a, **k: child)
    with pytest.raises(OSError, match='lost grant return'):
        op.main(_operator_args(tmp_path))
    record = json.loads((tmp_path / 'attempt' / 'operator-exit.json').read_text())
    assert len(child.stdin.delivered) == 32 and child.killed and record['child_reaped']
    assert record['operator_exception_type'] == 'OSError' and not record['terminal_consistent']


@pytest.mark.parametrize('selector_stage', [False, True])
def test_interrupted_drainer_start_withholds_grant_and_reaps_child(tmp_path, monkeypatch, selector_stage):
    op = _operator()
    root = Path(op.__file__).resolve().parents[1]
    revision = 'a' * 40
    if selector_stage:
        launcher = _tool('observe_flan_sentiment_selector_stage_v1.py')
        stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                                _dependency_site=lambda venv: (Path(venv), 'a' * 64),
                                _check_child_identity=lambda path, venv, pid: True)
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_bootstrap', lambda *args: None)
        monkeypatch.setattr(launcher, '_load', lambda path, name: stage if name == '_flan_stage_helper' else op)
        attempt = tmp_path / 'selector-stage'
        arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                     '--', 'select', '--repo', str(root)]
        receipt_name, spawn_name = 'stage-exit.json', 'stage-spawn.json'
    else:
        launcher = op
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_load_owner', lambda root, revision:
                            SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
        attempt = tmp_path / 'attempt'
        arguments = _operator_args(tmp_path)
        receipt_name, spawn_name = 'operator-exit.json', 'operator-spawn.json'

    class FakeChild:
        pid = 123
        killed = False
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def poll(self):
            return 9 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            return 9

    child = FakeChild()
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: child)
    real_start = threading.Thread.start
    def interrupted_start(thread):
        real_start(thread)
        raise KeyboardInterrupt('fabricated interrupted drainer start')
    monkeypatch.setattr(threading.Thread, 'start', interrupted_start)
    with pytest.raises(KeyboardInterrupt, match='drainer start'):
        launcher.main(arguments)
    record = json.loads((attempt / receipt_name).read_bytes())
    assert child.stdin.getvalue() == b'' and child.killed
    assert not (attempt / spawn_name).exists()
    assert record['reaped' if selector_stage else 'child_reaped']
    assert record['status' if selector_stage else 'terminal_consistent'] == ('incomplete' if selector_stage else False)


@pytest.mark.parametrize('selector_stage', [False, True])
def test_parent_cleanup_cancellation_takes_precedence_over_spawn_record_error(tmp_path, monkeypatch, selector_stage):
    op = _operator()
    root = Path(op.__file__).resolve().parents[1]
    revision = 'a' * 40
    if selector_stage:
        launcher = _tool('observe_flan_sentiment_selector_stage_v1.py')
        stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                                _dependency_site=lambda venv: (Path(venv), 'a' * 64),
                                _check_child_identity=lambda path, venv, pid: True)
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_bootstrap', lambda *args: None)
        monkeypatch.setattr(launcher, '_load', lambda path, name: stage if name == '_flan_stage_helper' else op)
        attempt = tmp_path / 'selector-stage'
        arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                     '--', 'select', '--repo', str(root)]
        receipt_name, spawn_name = 'stage-exit.json', 'stage-spawn.json'
    else:
        launcher = op
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_load_owner', lambda root, revision:
                            SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
        attempt = tmp_path / 'attempt'
        arguments = _operator_args(tmp_path)
        receipt_name, spawn_name = 'operator-exit.json', 'operator-spawn.json'

    class FakeChild:
        pid = 123
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'')
        stderr = io.BytesIO(b'')
        def poll(self):
            raise KeyboardInterrupt('fabricated cleanup cancellation')

    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: FakeChild())
    write_new = op._write_new
    def fail_spawn(path, value):
        if Path(path).name == spawn_name:
            raise OSError('fabricated spawn record error')
        return write_new(path, value)
    monkeypatch.setattr(op, '_write_new', fail_spawn)
    with pytest.raises(KeyboardInterrupt, match='cleanup cancellation') as exc:
        launcher.main(arguments)
    assert isinstance(exc.value.__cause__, OSError)
    record = json.loads((attempt / receipt_name).read_bytes())
    assert not record['reaped' if selector_stage else 'child_reaped']
    assert record['exception_type' if selector_stage else 'operator_exception_type'] == 'KeyboardInterrupt'


@pytest.mark.parametrize('selector_stage', [False, True])
@pytest.mark.parametrize('cancel_first', [False, True])
def test_parent_finalization_preserves_cancellation_priority(tmp_path, monkeypatch, selector_stage, cancel_first):
    op = _operator()
    root = Path(op.__file__).resolve().parents[1]
    revision = 'a' * 40
    if selector_stage:
        launcher = _tool('observe_flan_sentiment_selector_stage_v1.py')
        stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                                _dependency_site=lambda venv: (Path(venv), 'a' * 64),
                                _check_child_identity=lambda path, venv, pid: True)
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_bootstrap', lambda *args: None)
        monkeypatch.setattr(launcher, '_load', lambda path, name: stage if name == '_flan_stage_helper' else op)
        attempt = tmp_path / 'selector-stage'
        arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                     '--', 'select', '--repo', str(root)]
        spawn_name, exit_name = 'stage-spawn.json', 'stage-exit.json'
    else:
        launcher = op
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_load_owner', lambda root, revision:
                            SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
        attempt = tmp_path / 'attempt'
        arguments = _operator_args(tmp_path)
        spawn_name, exit_name = 'operator-spawn.json', 'operator-exit.json'

    class FakeChild:
        pid = 123
        killed = False
        stdin, stdout, stderr = io.BytesIO(), io.BytesIO(b''), io.BytesIO(b'')
        def poll(self): return 9 if self.killed else None
        def kill(self): self.killed = True
        def wait(self, timeout): return 9

    child = FakeChild()
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: child)
    original_write = op._write_new
    def fail_records(path, value):
        if Path(path).name == spawn_name:
            if cancel_first:
                raise KeyboardInterrupt('original cancellation')
            raise OSError('original spawn error')
        if Path(path).name == exit_name:
            if cancel_first:
                raise OSError('later receipt error')
            raise KeyboardInterrupt('later cancellation')
        return original_write(path, value)
    monkeypatch.setattr(op, '_write_new', fail_records)
    with pytest.raises(KeyboardInterrupt, match='original cancellation' if cancel_first else 'later cancellation') as exc:
        launcher.main(arguments)
    assert isinstance(exc.value.__cause__, OSError) and child.killed
    assert not (attempt / exit_name).exists()


@pytest.mark.parametrize('selector_stage', [False, True])
def test_parent_finalization_does_not_revive_ambient_exception(tmp_path, monkeypatch, selector_stage):
    op = _operator()
    root = Path(op.__file__).resolve().parents[1]
    revision = 'a' * 40
    if selector_stage:
        launcher = _tool('observe_flan_sentiment_selector_stage_v1.py')
        stage = SimpleNamespace(_snapshot=lambda root, revision: {'fabricated': 'a' * 64},
                                _dependency_site=lambda venv: (Path(venv), 'a' * 64),
                                _check_child_identity=lambda path, venv, pid: True)
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_bootstrap', lambda *args: None)
        monkeypatch.setattr(launcher, '_load', lambda path, name: stage if name == '_flan_stage_helper' else op)
        attempt = tmp_path / 'selector-stage'
        arguments = ['--revision', revision, '--attempt', str(attempt), '--venv', str(tmp_path),
                     '--', 'select', '--repo', str(root)]
        exit_name = 'stage-exit.json'
    else:
        launcher = op
        monkeypatch.setattr(launcher, '_empty_prefix', lambda: None)
        monkeypatch.setattr(launcher, '_load_owner', lambda root, revision:
                            SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}))
        arguments = _operator_args(tmp_path)
        exit_name = 'operator-exit.json'
    class FakeChild:
        pid = 123
        stdin, stdout, stderr = io.BytesIO(), io.BytesIO(b''), io.BytesIO(b'')
        def wait(self, timeout): return 0
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *args, **kwargs: FakeChild())
    original_write = op._write_new
    def fail_exit(path, value):
        if Path(path).name == exit_name:
            raise OSError('actual receipt error')
        return original_write(path, value)
    monkeypatch.setattr(op, '_write_new', fail_exit)
    try:
        raise ValueError('unrelated outer exception')
    except ValueError:
        with pytest.raises(OSError, match='actual receipt error'):
            launcher.main(arguments)


@pytest.mark.parametrize('cancel_first', [False, True])
def test_owner_child_final_check_preserves_cancellation_priority(tmp_path, monkeypatch, cancel_first):
    op = _operator()
    calls = []
    def check_prefix():
        calls.append('prefix')
        if len(calls) == 2:
            if cancel_first:
                raise OSError('later source check')
            raise KeyboardInterrupt('later cancellation')
    def run(args):
        if cancel_first:
            raise KeyboardInterrupt('original cancellation')
        raise OSError('original owner error')
    fake_owner = SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}, run=run)
    monkeypatch.setattr(op, '_empty_prefix', check_prefix)
    monkeypatch.setattr(op, '_receive_grant', lambda value: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision: fake_owner)
    monkeypatch.setattr(op, '_load_stage_helper', lambda root: SimpleNamespace(
        _dependency_site=lambda venv: (tmp_path / 'fake-site', 'a' * 64),
        _process_identity=lambda *args: {'schema': 'fabricated'}))
    monkeypatch.setattr(op.sys, 'path', list(op.sys.path))
    arguments = _operator_args(tmp_path) + ['--child', '--grant-sha256', 'a' * 64]
    with pytest.raises(KeyboardInterrupt, match='original cancellation' if cancel_first else 'later cancellation') as exc:
        op.main(arguments)
    assert isinstance(exc.value.__cause__, OSError)


def test_owner_child_final_check_does_not_revive_ambient_exception(tmp_path, monkeypatch):
    op = _operator()
    calls = []
    def check_prefix():
        calls.append('prefix')
        if len(calls) == 2:
            raise OSError('actual prefix error')
    fake_owner = SimpleNamespace(source_snapshot=lambda root, revision: {'revision': revision}, run=lambda args: None)
    monkeypatch.setattr(op, '_empty_prefix', check_prefix)
    monkeypatch.setattr(op, '_receive_grant', lambda value: None)
    monkeypatch.setattr(op, '_load_owner', lambda root, revision: fake_owner)
    monkeypatch.setattr(op, '_load_stage_helper', lambda root: SimpleNamespace(
        _dependency_site=lambda venv: (tmp_path / 'fake-site', 'a' * 64),
        _process_identity=lambda *args: {'schema': 'fabricated'}))
    monkeypatch.setattr(op.sys, 'path', list(op.sys.path))
    arguments = _operator_args(tmp_path) + ['--child', '--grant-sha256', 'a' * 64]
    try:
        raise ValueError('unrelated outer exception')
    except ValueError:
        with pytest.raises(OSError, match='actual prefix error'):
            op.main(arguments)


@pytest.mark.parametrize('cancel_first', [False, True])
def test_selector_child_receipt_preserves_cancellation_priority(tmp_path, monkeypatch, cancel_first):
    stage = _tool('run_flan_sentiment_selector_stage_v1.py')
    monkeypatch.setattr(stage, '_empty_prefix', lambda: None)
    monkeypatch.setattr(stage, '_receive_grant', lambda value: None)
    monkeypatch.setattr(stage, '_dependency_site', lambda venv: (tmp_path / 'fake-site', 'a' * 64))
    monkeypatch.setattr(stage, '_snapshot', lambda *args: {'fabricated': 'a' * 64})
    monkeypatch.setattr(stage.sys, 'path', list(stage.sys.path))
    class FakeLoader:
        def create_module(self, spec): return None
        def exec_module(self, module):
            def run():
                if cancel_first:
                    raise KeyboardInterrupt('original cancellation')
                raise OSError('original selector error')
            module.main = run
    spec = importlib.machinery.ModuleSpec('_fabricated_selector', FakeLoader())
    monkeypatch.setattr(stage.importlib.util, 'spec_from_file_location', lambda *args, **kwargs: spec)
    def fail_receipt(path, value):
        if cancel_first:
            raise OSError('later receipt error')
        raise KeyboardInterrupt('later cancellation')
    monkeypatch.setattr(stage, '_write_new', fail_receipt)
    root = Path(stage.__file__).resolve().parents[1]
    arguments = ['--revision', 'a' * 40, '--stage-receipt', str(tmp_path / 'stage.json'),
                 '--venv', str(tmp_path), '--grant-sha256', 'a' * 64, '--', 'commit-pool',
                 '--repo', str(root), '--pool-card', str(tmp_path / 'unused')]
    with pytest.raises(KeyboardInterrupt, match='original cancellation' if cancel_first else 'later cancellation') as exc:
        stage.main(arguments)
    assert isinstance(exc.value.__cause__, OSError)


def test_failure_receipt_does_not_swallow_new_cancellation(tmp_path, monkeypatch):
    import transformers
    paths, pool, selection = fabricated_custody(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, 'source_snapshot', lambda repo, revision: {'revision': revision, 'sha256': {}})
    monkeypatch.setattr(owner, 'bind_custody', lambda *args: (_ for _ in ()).throw(ValueError('binding failure')))
    monkeypatch.setattr(owner, '_write_new', lambda *args: (_ for _ in ()).throw(KeyboardInterrupt('receipt')))
    for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'}.items():
        monkeypatch.setenv(key, value)
    attempt = tmp_path / 'owner-attempt'
    attempt.mkdir()
    args = SimpleNamespace(repo=Path(owner.__file__).resolve().parents[1], asset_dir=tmp_path,
                           attempt=attempt, revision=pool['runner_revision'], **paths)
    with pytest.raises(KeyboardInterrupt, match='receipt') as exc:
        owner.run(args)
    assert isinstance(exc.value.__cause__, ValueError)
