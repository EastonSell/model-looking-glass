"""Fabricated comparison-observer handoff tests; no live private inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from looking_glass import injection_split_custody as custody
from looking_glass import injection_outer_observer as outer


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


here = Path(__file__).parent
fixtures = load(here / 'test_injection_split_comparison.py', '_comparison_observation_fixtures')
comparison = fixtures.comparison
adapter = load(here.parent / 'tools' / 'injection_comparison_observation.py', '_comparison_observation')
outer_fixtures = load(here / 'test_injection_outer_observer.py', '_fabricated_outer_observer_fixtures')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def example(tmp_path, monkeypatch):
    factory = fixtures.pair.__wrapped__(tmp_path)
    prepared = next(factory)
    try:
        paths = prepared.build(prepared.clear)
        comparison.compare_files(custody, **paths, publication_check=lambda name, raw: True)
        clear = paths['output'].read_bytes()
        collision = json.loads(clear)
        collision['status'] = 'collision'
        collision = json.dumps(collision, sort_keys=True, separators=(',', ':')).encode()
        monkeypatch.setattr(comparison, '_NATURAL_FREEZE', paths['expected']['natural_freeze_sha256'])
        monkeypatch.setattr(comparison, '_NATURAL_COMMITMENT', paths['expected']['natural_commitment_sha256'])
        monkeypatch.setattr(comparison, '_INJECTION_FREEZE', paths['expected']['injection_freeze_sha256'])
        commitment_url = comparison._RAW + 'b' * 40 + '/results/commitment.json'
        config = {'schema': 'mlg-injection-split-comparison-config-v1',
                  'source_root': str(tmp_path), 'custody_source': str(tmp_path / 'custody.py'),
                  'pycache_prefix': str(tmp_path / 'prefix'), 'output': str(tmp_path / 'fresh-receipt'),
                  'site_nonce': str(paths['site_nonce']), 'site_record': str(paths['site_record']),
                  'natural': {key: str(value) for key, value in paths['natural'].items()},
                  'injection': {key: str(value) for key, value in paths['injection'].items()},
                  'injection_commitment_sha256': paths['expected']['injection_commitment_sha256'],
                  'injection_commitment_url': commitment_url}
        config_path = tmp_path / 'config.json'
        config_path.write_bytes(json.dumps(config, sort_keys=True, separators=(',', ':')).encode())
        operation = tmp_path / 'operation.json'
        operation.write_bytes(b'{"fabricated":"operation"}')
        caller = here.parent / 'tools' / 'injection_split_comparison.py'
        public = {comparison._PUBLIC_URLS['natural_commitment']: paths['natural']['commitment'].read_bytes(),
                  commitment_url: paths['injection']['commitment'].read_bytes(),
                  adapter._CALLER_URL: caller.read_bytes()}
        options = dict(comparison=comparison, custody=custody, caller_source=caller,
                       config_path=config_path, config_sha256=sha(config_path.read_bytes()),
                       operation_path=operation, operation_sha256=sha(operation.read_bytes()),
                       output=tmp_path / 'observation', base_python=Path(sys.executable),
                       environment={'CUDA_VISIBLE_DEVICES': '-1'}, timeout=30,
                       source_check=lambda: True, protect_process=lambda process: True,
                       reader=lambda url: public[url])
        yield {'clear': clear, 'collision': collision, 'options': options}
    finally:
        factory.close()


@pytest.mark.parametrize('status', ['clear', 'collision'])
def test_observer_accepts_only_verified_receipt_after_process_completion(example, status):
    raw = example[status]
    def fake_observer(**kwargs):
        assert kwargs['terminal_check'](raw, 'execution', 123, 456) is True
        return {'schema': 'mlg-injection-outer-process-observation-v1', 'purpose': 'execution',
                'status': 'completed', 'exit_code': 0, 'parent_reaped': True,
                'parent_handle_closed': True, 'outer_job_drained': True, 'outer_job_closed': True,
                'active_job_members': 0, 'parent_pid': 123, 'parent_birth_100ns': 456,
                'config_sha256': kwargs['config_sha256'], 'operation_sha256': kwargs['operation_sha256'],
                'command_sha256': kwargs['command_sha256'], 'supervisor_source_sha256': kwargs['supervisor_source_sha256'],
                'supervisor_terminal_sha256': sha(raw)}
    result = adapter.observe_comparison(**example['options'], observer=fake_observer)
    assert result['comparison_status'] == status
    assert result['receipt_sha256'] == sha(raw)


def test_wrong_commitment_or_unverified_terminal_cannot_be_promoted(example):
    raw = example['clear'].replace(b'"natural_case_count":16', b'"natural_case_count":15')
    def fake_observer(**kwargs):
        assert kwargs['terminal_check'](raw, 'execution', 123, 456) is False
        return {'status': 'completed'}
    with pytest.raises(adapter.ObservationError):
        adapter.observe_comparison(**example['options'], observer=fake_observer)


def test_unobserved_source_stops_before_starting_comparison(example):
    options = dict(example['options'], source_check=lambda: False)
    calls = []
    with pytest.raises(adapter.ObservationError):
        adapter.observe_comparison(**options, observer=lambda **kwargs: calls.append(True))
    assert calls == []


@pytest.mark.parametrize('exit_code', [0, 7])
def test_real_synthetic_child_requires_zero_exit_after_receipt(example, monkeypatch, exit_code):
    raw = example['clear']
    bootstrap = ("import json,pathlib,sys; sys.stdin.buffer.readline(); "
                 "config=json.loads(pathlib.Path(sys.argv[3]).read_bytes()); "
                 "pathlib.Path(config['output']).write_bytes(bytes.fromhex(" + repr(raw.hex()) + ")); "
                 "sys.exit(" + str(exit_code) + ")")
    monkeypatch.setattr(adapter, '_BOOTSTRAP', bootstrap)
    options = dict(example['options'], environment=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
    def synthetic_observer(**kwargs):
        return outer._observe(**kwargs, job_type=outer_fixtures.FakeJob,
                              configure_job=lambda job: None, birth_time=lambda process: 1)
    if exit_code == 0:
        result = adapter.observe_comparison(**options, observer=synthetic_observer)
        assert result['comparison_status'] == 'clear'
        assert result['receipt_sha256'] == sha(raw)
        assert (options['output'] / 'observation.json').is_file()
    else:
        with pytest.raises(outer.ObserverError):
            adapter.observe_comparison(**options, observer=synthetic_observer)
        assert not (options['output'] / 'observation.json').exists()
        assert Path(json.loads(options['config_path'].read_bytes())['output']).read_bytes() == raw
