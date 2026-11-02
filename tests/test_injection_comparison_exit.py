"""Synthetic comparison operator-exit records; no private data is read."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from looking_glass import injection_outer_observer as outer


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


here = Path(__file__).parent
exit_check = load(here.parent / 'tools' / 'injection_comparison_exit.py', '_comparison_exit')
comparison = load(here.parent / 'tools' / 'injection_split_comparison.py', '_comparison_exit_inputs')
outer_fixtures = load(here / 'test_injection_outer_observer.py', '_comparison_exit_outer_fixtures')


@pytest.fixture(autouse=True)
def isolated_synthetic_outer_custody():
    with outer_fixtures.synthetic_observer_context():
        yield


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def config_bytes(root, receipt):
    value = {key: str(root / key) for key in
             ('source_root', 'custody_source', 'pycache_prefix', 'site_nonce', 'site_record')}
    value.update({'schema': 'mlg-injection-split-comparison-config-v1', 'output': str(receipt),
                  'natural': {key: str(root / ('natural-' + key)) for key in comparison._SPLIT_KEYS},
                  'injection': {key: str(root / ('injection-' + key)) for key in comparison._SPLIT_KEYS},
                  'injection_commitment_sha256': 'a' * 64,
                  'injection_commitment_url': comparison._RAW + 'a' * 40 + '/results/commitment.json'})
    return canonical(value)


@pytest.fixture
def case():
    receipt = canonical({'schema': 'fabricated-receipt', 'status': 'clear'})
    config_sha, operation_sha, command_sha, operator_sha = ('a' * 64, 'b' * 64, 'c' * 64, 'd' * 64)
    inner = canonical({'schema': 'mlg-injection-outer-process-observation-v1',
        'purpose': 'execution', 'status': 'completed', 'exit_code': 0,
        'parent_reaped': True, 'parent_handle_closed': True, 'outer_job_drained': True,
        'outer_job_closed': True, 'active_job_members': 0, 'parent_pid': 100,
        'parent_birth_100ns': 200, 'config_sha256': config_sha, 'operation_sha256': operation_sha,
        'command_sha256': command_sha, 'supervisor_source_sha256': exit_check._CALLER_SHA,
        'supervisor_terminal_sha256': sha(receipt)})
    result = {'comparison_status': 'clear', 'receipt_sha256': sha(receipt),
              'observer_observation_sha256': sha(inner)}
    options = dict(config_sha256=config_sha, operation_sha256=operation_sha,
                   comparator_command_sha256=command_sha, operator_source_sha256=operator_sha,
                   pid=123, birth=456)
    terminal = exit_check.make_terminal(result, **options)
    return {'receipt': receipt, 'inner': inner, 'terminal': terminal, 'options': options}


def test_terminal_binds_observed_process_and_clean_comparison(case):
    checked = exit_check.verify_terminal(case['terminal'], inner_observation=case['inner'],
                                         receipt=case['receipt'], receipt_check=lambda raw: True,
                                         **case['options'])
    assert checked == {'status': 'clear', 'receipt_sha256': sha(case['receipt'])}


@pytest.mark.skipif(os.name != 'nt', reason='the production process identity uses Windows handles')
def test_current_identity_matches_the_native_process_birth():
    pid, birth = exit_check.current_identity()
    assert pid == os.getpid()
    assert birth == outer._birth_time(SimpleNamespace(_handle=-1))


@pytest.mark.skipif(os.name != 'nt', reason='the production process identity uses Windows handles')
def test_operator_writes_fresh_terminal_only_after_verified_adapter(case, tmp_path):
    inner_dir = tmp_path / 'inner'
    inner_dir.mkdir()
    receipt = tmp_path / 'receipt.json'
    terminal = tmp_path / 'operator-terminal.json'
    config = tmp_path / 'config.json'
    config.write_bytes(config_bytes(tmp_path, receipt))
    options = {'output': inner_dir, 'config_path': config, 'config_sha256': sha(config.read_bytes()),
               'operation_sha256': case['options']['operation_sha256']}
    inner_value = json.loads(case['inner'])
    inner_value['config_sha256'] = options['config_sha256']
    inner = canonical(inner_value)
    result = {'comparison_status': 'clear', 'receipt_sha256': sha(case['receipt']),
              'observer_observation_sha256': sha(inner)}
    class Adapter:
        def observe_comparison(self, **kwargs):
            receipt.write_bytes(case['receipt'])
            (inner_dir / 'observation.json').write_bytes(inner)
            return result
    actual = exit_check.run_operator(adapter=Adapter(), adapter_options=options, comparison=comparison,
        terminal_path=terminal, receipt_path=receipt, source_check=lambda: True,
        receipt_check=lambda raw: raw == case['receipt'],
        comparator_command_sha256=case['options']['comparator_command_sha256'],
        operator_source_sha256=case['options']['operator_source_sha256'])
    value = json.loads(terminal.read_bytes())
    assert value['operator_pid'] == os.getpid()
    assert actual['status'] == 'clear'
    with pytest.raises(exit_check.ExitError, match='fresh'):
        exit_check.run_operator(adapter=Adapter(), adapter_options=options, comparison=comparison,
            terminal_path=terminal, receipt_path=receipt, source_check=lambda: True,
            receipt_check=lambda raw: True, comparator_command_sha256=case['options']['comparator_command_sha256'],
            operator_source_sha256=case['options']['operator_source_sha256'])


@pytest.mark.parametrize('mutation', ['pid', 'birth', 'receipt', 'inner', 'source', 'unclean', 'unchecked'])
def test_changed_or_uncertain_evidence_cannot_authorize(case, mutation):
    options = dict(case['options'])
    receipt, inner, check = case['receipt'], case['inner'], lambda raw: True
    if mutation == 'pid':
        options['pid'] += 1
    elif mutation == 'birth':
        options['birth'] += 1
    elif mutation == 'receipt':
        receipt += b' '
    elif mutation == 'inner':
        inner += b' '
    elif mutation == 'source':
        options['operator_source_sha256'] = 'e' * 64
    elif mutation == 'unclean':
        value = json.loads(inner)
        value['outer_job_closed'] = False
        inner = canonical(value)
    else:
        check = lambda raw: False
    with pytest.raises(exit_check.ExitError):
        exit_check.verify_terminal(case['terminal'], inner_observation=inner, receipt=receipt,
                                   receipt_check=check, **options)


def test_collision_is_retained_as_a_negative_result(case):
    receipt = canonical({'schema': 'fabricated-receipt', 'status': 'collision'})
    inner_value = json.loads(case['inner'])
    inner_value['supervisor_terminal_sha256'] = sha(receipt)
    inner = canonical(inner_value)
    result = {'comparison_status': 'collision', 'receipt_sha256': sha(receipt),
              'observer_observation_sha256': sha(inner)}
    terminal = exit_check.make_terminal(result, **case['options'])
    checked = exit_check.verify_terminal(terminal, inner_observation=inner, receipt=receipt,
                                         receipt_check=lambda raw: True, **case['options'])
    assert checked['status'] == 'collision'


@pytest.mark.parametrize('exit_code', [0, 7])
def test_outer_observer_binds_fresh_operator_terminal_and_exit(case, tmp_path, exit_code):
    config = tmp_path / 'config.json'
    operation = tmp_path / 'operation.json'
    receipt_path = tmp_path / 'receipt.json'
    config.write_bytes(config_bytes(tmp_path, receipt_path))
    operation.write_bytes(b'{"synthetic":"operation"}')
    options = dict(case['options'], config_sha256=sha(config.read_bytes()),
                   operation_sha256=sha(operation.read_bytes()), birth=1)
    inner_value = json.loads(case['inner'])
    inner_value['config_sha256'] = options['config_sha256']
    inner_value['operation_sha256'] = options['operation_sha256']
    inner = canonical(inner_value)
    result = {'comparison_status': 'clear', 'receipt_sha256': sha(case['receipt']),
              'observer_observation_sha256': sha(inner)}
    terminal_template = json.loads(exit_check.make_terminal(result, **options))
    inner_path = tmp_path / 'inner.json'
    terminal_path = tmp_path / 'terminal.json'
    program = ("import json,os,pathlib,sys; sys.stdin.buffer.readline(); "
               "p=pathlib.Path; p(sys.argv[1]).write_bytes(bytes.fromhex(sys.argv[4])); "
               "p(sys.argv[2]).write_bytes(bytes.fromhex(sys.argv[5])); "
               "t=json.loads(sys.argv[6]); t['operator_pid']=os.getpid(); "
               "p(sys.argv[3]).write_text(json.dumps(t,sort_keys=True,separators=(',',':'))); "
               "sys.exit(int(sys.argv[7]))")
    command = [str(Path(sys._base_executable).resolve(strict=True)), '-B', '-c', program, str(receipt_path), str(inner_path),
               str(terminal_path), case['receipt'].hex(), inner.hex(), json.dumps(terminal_template), str(exit_code)]
    def observer(**kwargs):
        return outer._observe(**kwargs, job_type=outer_fixtures.FakeJob,
                              configure_job=lambda job: None, birth_time=lambda process: 1)
    kwargs = dict(comparison=comparison, command=command, output=tmp_path / 'outer',
                  environment=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), config_path=config,
                  operation_path=operation, terminal_path=terminal_path, inner_path=inner_path,
                  receipt_path=receipt_path, start_token=b'FAKE-GATE\n', timeout=10,
                  source_check=lambda: True, protect_process=lambda process: True,
                  receipt_check=lambda raw: raw == case['receipt'], observer=observer,
                  config_sha256=options['config_sha256'], operation_sha256=options['operation_sha256'],
                  comparator_command_sha256=options['comparator_command_sha256'],
                  operator_source_sha256=options['operator_source_sha256'])
    if exit_code == 0:
        result = exit_check.observe_operator(**kwargs)
        assert result['status'] == 'clear'
        assert result['receipt_sha256'] == sha(case['receipt'])
        assert (tmp_path / 'outer' / 'observation.json').is_file()
    else:
        with pytest.raises(outer.ObserverError):
            exit_check.observe_operator(**kwargs)
        assert receipt_path.read_bytes() == case['receipt']
        assert not (tmp_path / 'outer' / 'observation.json').exists()


def test_preexisting_receipt_is_rejected_before_operator_start(case, tmp_path):
    receipt_path = tmp_path / 'receipt.json'
    receipt_path.write_bytes(case['receipt'])
    called = []
    with pytest.raises(exit_check.ExitError, match='fresh'):
        exit_check.observe_operator(comparison=comparison, command=['fabricated'], output=tmp_path / 'outer',
            environment={}, config_path=tmp_path / 'config', operation_path=tmp_path / 'operation',
            terminal_path=tmp_path / 'terminal', inner_path=tmp_path / 'inner', receipt_path=receipt_path,
            start_token=b'FAKE', timeout=10, source_check=lambda: True, protect_process=lambda process: True,
            receipt_check=lambda raw: True, observer=lambda **kwargs: called.append(True),
            config_sha256='a' * 64, operation_sha256='b' * 64, comparator_command_sha256='c' * 64,
            operator_source_sha256='d' * 64)
    assert called == []


def test_configured_receipt_path_must_match_before_operator_starts(case, tmp_path):
    configured = tmp_path / 'configured-receipt.json'
    other = tmp_path / 'other-receipt.json'
    config = tmp_path / 'config.json'
    config.write_bytes(config_bytes(tmp_path, configured))
    called = []
    adapter = SimpleNamespace(observe_comparison=lambda **kwargs: called.append(True))
    options = {'output': tmp_path / 'inner', 'config_path': config,
               'config_sha256': sha(config.read_bytes()), 'operation_sha256': 'b' * 64}
    with pytest.raises(exit_check.ExitError, match='receipt'):
        exit_check.run_operator(adapter=adapter, adapter_options=options, comparison=comparison,
            terminal_path=tmp_path / 'terminal.json', receipt_path=other,
            source_check=lambda: True, receipt_check=lambda raw: True,
            comparator_command_sha256='c' * 64, operator_source_sha256='d' * 64)
    assert called == []


def test_changed_config_after_comparison_cannot_create_terminal(case, tmp_path):
    receipt = tmp_path / 'receipt.json'
    terminal = tmp_path / 'terminal.json'
    inner_dir = tmp_path / 'inner'
    inner_dir.mkdir()
    config = tmp_path / 'config.json'
    config.write_bytes(config_bytes(tmp_path, receipt))
    config_sha = sha(config.read_bytes())
    inner_value = json.loads(case['inner'])
    inner_value['config_sha256'] = config_sha
    inner = canonical(inner_value)
    result = {'comparison_status': 'clear', 'receipt_sha256': sha(case['receipt']),
              'observer_observation_sha256': sha(inner)}
    class Adapter:
        def observe_comparison(self, **kwargs):
            receipt.write_bytes(case['receipt'])
            (inner_dir / 'observation.json').write_bytes(inner)
            config.write_bytes(config.read_bytes() + b' ')
            return result
    with pytest.raises(exit_check.ExitError, match='configuration'):
        exit_check.run_operator(adapter=Adapter(), comparison=comparison, terminal_path=terminal,
            receipt_path=receipt, source_check=lambda: True, receipt_check=lambda raw: True,
            adapter_options={'output': inner_dir, 'config_path': config, 'config_sha256': config_sha,
                             'operation_sha256': case['options']['operation_sha256']},
            comparator_command_sha256=case['options']['comparator_command_sha256'],
            operator_source_sha256=case['options']['operator_source_sha256'])
    assert not terminal.exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job integration')
def test_real_windows_job_observes_native_operator_identity(case, tmp_path):
    try:
        source = subprocess.check_output(['git', 'show',
            '479b41fa3ab1b3131644fa14d68d761ce9672f42:scripts/v2_source_guards.py'],
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('reviewed Job source is unavailable in this archive')
    job_module = ModuleType('_fabricated_comparison_exit_job')
    exec(compile(source, 'reviewed-job.py', 'exec', dont_inherit=True), job_module.__dict__)
    receipt = tmp_path / 'receipt.json'
    config = tmp_path / 'config.json'
    operation = tmp_path / 'operation.json'
    config.write_bytes(config_bytes(tmp_path, receipt))
    operation.write_bytes(b'{"synthetic":"operation"}')
    options = dict(case['options'], config_sha256=sha(config.read_bytes()),
                   operation_sha256=sha(operation.read_bytes()))
    inner_value = json.loads(case['inner'])
    inner_value.update(config_sha256=options['config_sha256'], operation_sha256=options['operation_sha256'])
    inner = canonical(inner_value)
    result = {'comparison_status': 'clear', 'receipt_sha256': sha(case['receipt']),
              'observer_observation_sha256': sha(inner)}
    template = json.loads(exit_check.make_terminal(result, **options))
    inner_path, terminal_path = tmp_path / 'inner.json', tmp_path / 'terminal.json'
    program = ("import json,pathlib,runpy,sys; sys.stdin.buffer.readline(); p=pathlib.Path; "
               "p(sys.argv[1]).write_bytes(bytes.fromhex(sys.argv[4])); "
               "p(sys.argv[2]).write_bytes(bytes.fromhex(sys.argv[5])); "
               "t=json.loads(sys.argv[6]); pid,birth=runpy.run_path(sys.argv[7])['current_identity'](); "
               "t['operator_pid']=pid; t['operator_birth_100ns']=birth; "
               "p(sys.argv[3]).write_text(json.dumps(t,sort_keys=True,separators=(',',':')))")
    command = [str(Path(sys._base_executable).resolve(strict=True)), '-B', '-c', program, str(receipt),
               str(inner_path), str(terminal_path), case['receipt'].hex(), inner.hex(),
               json.dumps(template), str(here.parent / 'tools' / 'injection_comparison_exit.py')]
    def observer(**kwargs):
        return outer._observe(**kwargs, job_type=job_module.WindowsJob,
                              configure_job=outer._configure_windows_job, birth_time=outer._birth_time)
    result = exit_check.observe_operator(comparison=comparison, command=command, output=tmp_path / 'outer',
        environment=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), config_path=config, operation_path=operation,
        terminal_path=terminal_path, inner_path=inner_path, receipt_path=receipt, start_token=b'FAKE-GATE\n',
        timeout=20, source_check=lambda: True, protect_process=lambda process: True,
        receipt_check=lambda raw: raw == case['receipt'], observer=observer,
        config_sha256=options['config_sha256'], operation_sha256=options['operation_sha256'],
        comparator_command_sha256=options['comparator_command_sha256'],
        operator_source_sha256=options['operator_source_sha256'])
    assert result['status'] == 'clear'
    observed = json.loads((tmp_path / 'outer' / 'observation.json').read_bytes())
    assert observed['parent_pid'] == json.loads(terminal_path.read_bytes())['operator_pid']
    assert observed['parent_birth_100ns'] == json.loads(terminal_path.read_bytes())['operator_birth_100ns']
    assert observed['outer_job_drained'] is observed['outer_job_closed'] is True
