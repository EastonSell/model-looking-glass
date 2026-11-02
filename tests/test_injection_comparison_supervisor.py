"""Synthetic checks for the independently observed comparison operator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parents[1]
operator = load('_comparison_supervisor_operator', root / 'tools/injection_comparison_operator.py')
supervisor = load('_comparison_supervisor', root / 'tools/injection_comparison_supervisor.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_direct_operator_command_hashes_source_before_executing_it(tmp_path):
    config = {'base_python': str(Path(sys._base_executable).resolve(strict=True)),
              'pycache_prefix': str(tmp_path / 'prefix')}
    command = supervisor.operator_command(operator, config, root / 'tools/injection_comparison_operator.py',
        tmp_path / 'operator-config.json', '1' * 64, '2' * 64,
        operator._RAW + 'a' * 40 + '/results/manifest.json')
    assert command[1:5] == ['-I', '-S', '-B', '-X']
    assert command[5] == 'pycache_prefix=' + config['pycache_prefix']
    assert supervisor._OPERATOR_SHA in command
    assert command[-4:] == [str(tmp_path / 'operator-config.json'), '1' * 64, '2' * 64,
                            operator._RAW + 'a' * 40 + '/results/manifest.json']
    assert 'hashlib.sha256' in command[7] and "'--run'" in command[7]


def test_command_rejects_an_executable_other_than_direct_base(tmp_path):
    config = {'base_python': str(tmp_path / 'not-python.exe'), 'pycache_prefix': str(tmp_path / 'prefix')}
    with pytest.raises(supervisor.SupervisorError):
        supervisor.operator_command(operator, config, root / 'tools/injection_comparison_operator.py',
            tmp_path / 'operator-config.json', '1' * 64, '2' * 64,
            operator._RAW + 'a' * 40 + '/results/manifest.json')


def test_supervisor_holds_protections_until_observed_exit_and_preserves_collision(tmp_path, monkeypatch):
    source = tmp_path / 'comparison.py'
    source.write_bytes(b'value = 1\n')
    source_url = operator._RAW + 'c' * 40 + '/tools/comparison.py'
    monkeypatch.setattr(operator, '_PINNED_SOURCES', (('comparison', sha(source.read_bytes()), source_url),))
    captured = (('comparison', '_fake_comparison', '', source, source.read_bytes()),)
    op_source = root / 'tools/injection_comparison_operator.py'
    manifest_url = operator._RAW + 'd' * 40 + '/results/manifest.json'
    own_url = operator._RAW + 'e' * 40 + '/tools/supervisor.py'
    public_dir, phase, evidence = (tmp_path/'public', tmp_path/'injection'/'phase', tmp_path/'injection'/'evidence')
    for path in (public_dir, phase, evidence, tmp_path/'operator-cache', tmp_path/'comparison-cache'):
        path.mkdir(parents=True)
    paths = {name: public_dir / name for name in ('config','operation','manifest','natural-freeze',
        'natural-commitment','natural-inventory','injection-freeze','injection-commitment')}
    for name, path in paths.items(): path.write_bytes(name.encode())
    outer_config_path = tmp_path/'injection'/'operator-config.json'
    outer_config_path.write_bytes(b'operator-config')
    comparison_config = {'pycache_prefix':str(tmp_path/'comparison-cache'), 'output':str(evidence/'receipt.json'),
        'natural':{'freeze':str(paths['natural-freeze']), 'commitment':str(paths['natural-commitment']),
                   'inventory':str(paths['natural-inventory'])},
        'injection':{'freeze':str(paths['injection-freeze']), 'commitment':str(paths['injection-commitment'])},
        'injection_commitment_sha256':'1'*64, 'injection_commitment_url':operator._RAW+'b'*40+'/commitment.json'}
    outer_config = {'source_paths':{'comparison':str(source)}, 'comparison_config_path':str(paths['config']),
        'comparison_config_sha256':sha(paths['config'].read_bytes()),
        'operation_path':str(paths['operation']), 'operation_sha256':sha(paths['operation'].read_bytes()),
        'operation_url':operator._RAW+'a'*40+'/operation.json', 'manifest_path':str(paths['manifest']),
        'natural_parent':str(tmp_path/'natural'), 'injection_parent':str(tmp_path/'injection'),
        'site_parent':str(tmp_path/'site'), 'identities':{}, 'terminal_path':str(evidence/'terminal.json'),
        'inner_output':str(evidence/'inner'), 'base_python':str(Path(sys._base_executable).resolve(strict=True)),
        'pycache_prefix':str(tmp_path/'operator-cache'), 'timeout':60}
    private = {'injection_parent':tmp_path/'injection','injection_phase':phase,
        'natural_parent':tmp_path/'natural','natural_phase':tmp_path/'natural'/'phase',
        'site_parent':tmp_path/'site','site_nonce':tmp_path/'site'/'nonce.bin'}
    events = []
    own_bytes = Path(supervisor.__file__).read_bytes()
    manifest = {'sources':{'operator':{'sha256':supervisor._OPERATOR_SHA,'url':supervisor._OPERATOR_URL},
        'supervisor':{'sha256':sha(own_bytes),'url':own_url}}}
    monkeypatch.setattr(operator, 'parse_operator_config', lambda *args: outer_config)
    monkeypatch.setattr(operator, 'validate_source_manifest', lambda *args: manifest)
    monkeypatch.setattr(operator, 'verify_operator_manifest', lambda *args: True)
    monkeypatch.setattr(operator, 'validate_public_binding', lambda **kwargs: events.append('public') or {})
    monkeypatch.setattr(operator, '_private_paths', lambda *args: private)
    monkeypatch.setattr(operator, 'verify_private_inputs', lambda *args, **kwargs: events.append('private') or True)
    comparison = SimpleNamespace(_read_file=lambda path, limit: Path(path).read_bytes(),
                                 parse_config=lambda raw: comparison_config)
    class Lease:
        def __init__(self, parent, files): self.parent, self.files = parent, files
        def __enter__(self): events.append('lease'); return self
        def close(self): events.append('close')
        def handles(self): return (id(self),)
    class Guard:
        SourceFileLease = Lease
        @staticmethod
        def duplicate_to_process(handles, process): events.append('duplicate'); return handles
    def guarded(leases, precheck, launch, postcheck):
        try:
            for lease in leases: lease.__enter__()
            assert precheck(leases) is True
            return postcheck(launch(leases))
        finally:
            for lease in reversed(leases): lease.close()
    def observe(**kwargs):
        assert kwargs['source_check']() is True
        assert kwargs['protect_process'](object()) is True
        assert kwargs['receipt_check'](b'public receipt') is True
        events.append('observed_exit')
        return {'status':'collision','receipt_sha256':'f'*64,'operator_observation_sha256':'e'*64}
    exit_check = SimpleNamespace(observe_operator=observe, verify_public_receipt=lambda **kwargs: True)
    modules = {'comparison':comparison,'custody':SimpleNamespace(), 'source_guards':Guard,
        'replay_owner':SimpleNamespace(run_guarded=guarded), 'natural_query':SimpleNamespace(),
        'adapter':SimpleNamespace(_BOOTSTRAP='comparison bootstrap'), 'exit':exit_check,
        'observer':SimpleNamespace(observe_supervisor=lambda **kwargs: None)}
    public = {supervisor._OPERATOR_URL:op_source.read_bytes(), own_url:own_bytes}
    args = dict(operator=operator, modules=modules, captured=captured,
        operator_source=op_source, operator_config_path=outer_config_path,
        operator_config_sha256=sha(outer_config_path.read_bytes()), manifest_sha256=sha(paths['manifest'].read_bytes()),
        manifest_url=manifest_url, output=evidence/'outer', environment={'CUDA_VISIBLE_DEVICES':'-1'},
        timeout=90, reader=public.__getitem__)
    result = supervisor.observe_protected_operator(**args)
    assert result['status'] == 'collision'
    assert events.index('private') < events.index('observed_exit') < len(events)-1
    assert 'duplicate' in events and events[-1] == 'close'
    events.clear()
    with pytest.raises(supervisor.SupervisorError):
        supervisor.observe_protected_operator(**dict(args, output=evidence/'inner'/'nested'))
    assert 'observed_exit' not in events
    events.clear()
    public[supervisor._OPERATOR_URL] = b'wrong published source'
    with pytest.raises(supervisor.SupervisorError):
        supervisor.observe_protected_operator(**args)
    assert 'observed_exit' not in events
