"""Only fabricated data is used to test the independent held-out auditor."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch

from looking_glass import agreement_heldout as heldout
from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_patch_runner import NaturalAgreementRunner

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / 'tools' / 'audit_natural_agreement_heldout.py'
fixture_spec = importlib.util.spec_from_file_location('toy_heldout_fixture', ROOT / 'tests' / 'test_agreement_heldout.py')
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def load_auditor():
    assert AUDITOR.is_file(), 'independent held-out auditor is missing'
    spec = importlib.util.spec_from_file_location('independent_heldout_audit', AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_families_compare_frozen_prediction_with_zero():
    audit = load_auditor()
    result = audit.prediction_errors(2.0, (1.0, 2.0, -1.0, 0.0))
    assert result == {
        'errors': [-1.0, 0.0, -3.0, -2.0],
        'frozen_predictor': {'mae': 1.5, 'rmse': math.sqrt(3.5)},
        'zero_predictor': {'mae': 1.0, 'rmse': math.sqrt(1.5)},
    }


@pytest.mark.parametrize('prediction,effects', [
    (2.0, (1.0, 2.0, -1.0)),
    (float('nan'), (1.0, 2.0, -1.0, 0.0)),
    (2.0, (1.0, float('inf'), -1.0, 0.0)),
    (1e308, (1e308, -1e308, 0.0, 0.0)),
])
def test_invalid_or_overflowing_prediction_inputs_fail_closed(prediction, effects):
    with pytest.raises(ValueError):
        load_auditor().prediction_errors(prediction, effects)


@pytest.fixture(scope='module')
def fabricated_heldout(tmp_path_factory):
    """The inputs and weights are fabricated; no pretrained model is used."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source = fixture.fake_study.__wrapped__()
    study = next(source)
    try:
        path = tmp_path_factory.mktemp('fabricated-heldout') / 'attempt'
        model = fixture.cpu_toy(123)
        runner = NaturalAgreementRunner(model, AgreementJournal(path, {'protocol': 'fabricated-only'}),
                                        expected_quartets=4)
        heldout.run_committed_heldout(runner, **study)
        assert model.calls == 608
        yield path, study
    finally:
        source.close()
        torch.set_num_threads(previous_threads)


def audit_args(study):
    return {'prediction_bytes': study['prediction_bytes'], 'freeze_bytes': study['freeze'].record_bytes,
            'heldout_bytes': study['record_bytes'], 'split_nonce': study['split_nonce'],
            'expected_freeze_sha256': study['freeze'].sha256,
            'expected_commitment': study['heldout_commitment'], 'synthetic': True}


def test_complete_fabricated_journal_uses_frozen_sites(fabricated_heldout):
    path, study = fabricated_heldout
    audit = load_auditor()
    result = audit.inspect(path, **audit_args(study))
    assert result['status'] == 'independent_saved_evidence_pass'
    assert result['forwards'] == 608
    assert result['scope'] == 'Fabricated synthetic data only.'
    assert result['selected_site'] == '4:mlp:last'
    assert result['sites'][0]['frozen_prediction'] == 0.125
    assert len(result['sites']) == 8
    assert len(result['clean_cases']) == 16
    assert len(result['present_metrics']) == len(result['secondary_past_metrics']) == 8
    assert len(result['present_metrics'][0]['metrics']['pairs']) == 64
    assert len(result['present_metrics'][0]['metrics']['quartets']) == 4


def test_frozen_cpu_schema_without_device_rng_is_accepted(fabricated_heldout):
    path, study = fabricated_heldout
    event_path, terminal_path = path / 'events.jsonl', path / 'terminal.json'
    original_events, original_terminal = event_path.read_bytes(), terminal_path.read_bytes()
    rows = [json.loads(line) for line in original_events.splitlines()]
    del rows[0]['event']['device_rng']
    del rows[-1]['event']['result']['device_rng']
    raw = b''.join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode() + b'\n' for row in rows)
    terminal = json.loads(original_terminal)
    terminal['event_log_bytes'] = len(raw)
    try:
        event_path.write_bytes(raw)
        terminal_path.write_bytes(json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        result = load_auditor().inspect(path, **audit_args(study))
        assert result['forwards'] == 608
    finally:
        event_path.write_bytes(original_events)
        terminal_path.write_bytes(original_terminal)


def test_phase_limits_reject_underfunded_saved_journals(fabricated_heldout, monkeypatch):
    path, study = fabricated_heldout
    audit = load_auditor()
    assert audit.MAX_EVENT_LOG_BYTES == 48 * 1024 * 1024
    assert audit.MAX_RETAINED_BYTES == 160 * 1024 * 1024
    with monkeypatch.context() as patch:
        patch.setattr(audit, 'MAX_EVENT_LOG_BYTES', (path / 'events.jsonl').stat().st_size - 1)
        with pytest.raises(AssertionError):
            audit.inspect(path, **audit_args(study))
    with monkeypatch.context() as patch:
        retained = json.loads((path / 'terminal.json').read_bytes())['retained_tensor_bytes']
        patch.setattr(audit, 'MAX_RETAINED_BYTES', retained - 1)
        with pytest.raises(AssertionError):
            audit.inspect(path, **audit_args(study))


def test_pending_terminal_is_not_a_complete_journal(fabricated_heldout):
    path, study = fabricated_heldout
    pending = path / '.terminal-incomplete.pending'
    pending.write_bytes(b'partial')
    try:
        with pytest.raises(AssertionError):
            load_auditor().inspect(path, **audit_args(study))
    finally:
        pending.unlink()


def test_hard_linked_artifact_is_not_independent_evidence(tmp_path):
    original = tmp_path / 'original'
    alias = tmp_path / 'alias'
    original.write_bytes(b'example')
    try:
        os.link(original, alias)
    except OSError as error:
        pytest.skip(f'hard links are unavailable: {error}')
    with pytest.raises(AssertionError):
        load_auditor().read_bounded(original, 100)


def test_frozen_abstention_is_preserved(fabricated_heldout, tmp_path):
    del fabricated_heldout  # Keep the fabricated-only fixture's thread limit active.
    original = fixture.discovery_prediction
    with mock.patch.object(fixture, 'discovery_prediction',
                           lambda: original(effect_sign=-1, magnitude_tie=True)):
        source = fixture.fake_study.__wrapped__()
        study = next(source)
        try:
            path = tmp_path / 'abstained'
            model = fixture.cpu_toy(321)
            runner = NaturalAgreementRunner(model, AgreementJournal(path, {'protocol': 'fabricated-only'}),
                                            expected_quartets=4)
            heldout.run_committed_heldout(runner, **study)
            result = load_auditor().inspect(path, **audit_args(study))
        finally:
            source.close()
    assert model.calls == 608
    assert result['selected_site'] is None
    assert result['magnitude_site'] == result['earliest_site']
    assert 'ranking' not in result


def test_cli_writes_once_for_revealed_fabricated_journal(fabricated_heldout, tmp_path):
    path, study = fabricated_heldout
    files = {'prediction': study['prediction_bytes'], 'freeze': study['freeze'].record_bytes,
             'reveal': study['record_bytes'], 'nonce': study['split_nonce']}
    for name, contents in files.items():
        (tmp_path / name).write_bytes(contents)
    output = tmp_path / 'audit.json'
    command = [sys.executable, str(AUDITOR), str(path), '--prediction', str(tmp_path / 'prediction'),
               '--freeze', str(tmp_path / 'freeze'), '--reveal', str(tmp_path / 'reveal'),
               '--nonce-file', str(tmp_path / 'nonce'), '--freeze-sha256', study['freeze'].sha256,
               '--commitment', study['heldout_commitment'], '--synthetic', '--output', str(output)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'}
    first = subprocess.run(command, capture_output=True, env=env, timeout=30)
    assert first.returncode == 0, first.stderr.decode('utf-8', 'replace')
    original = output.read_bytes()
    assert json.loads(original)['scope'] == 'Fabricated synthetic data only.'
    second = subprocess.run(command, capture_output=True, env=env, timeout=30)
    assert second.returncode != 0
    assert output.read_bytes() == original


def test_cli_refuses_output_inside_the_immutable_journal(fabricated_heldout, tmp_path):
    path, study = fabricated_heldout
    files = {'prediction': study['prediction_bytes'], 'freeze': study['freeze'].record_bytes,
             'reveal': study['record_bytes'], 'nonce': study['split_nonce']}
    for name, contents in files.items():
        (tmp_path / name).write_bytes(contents)
    output = path / 'must-not-be-added.json'
    command = [sys.executable, str(AUDITOR), str(path), '--prediction', str(tmp_path / 'prediction'),
               '--freeze', str(tmp_path / 'freeze'), '--reveal', str(tmp_path / 'reveal'),
               '--nonce-file', str(tmp_path / 'nonce'), '--freeze-sha256', study['freeze'].sha256,
               '--commitment', study['heldout_commitment'], '--synthetic', '--output', str(output)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run(command, capture_output=True, env=env, timeout=30)
    try:
        assert result.returncode != 0
        assert not output.exists()
    finally:
        if output.exists():
            output.unlink()


@pytest.mark.parametrize('tamper', ['frozen_prediction', 'freeze_pin', 'nonce', 'commitment',
                                    'selected_site', 'prediction_error', 'zero_error',
                                    'clean_margin', 'pair_effect', 'landing', 'metric_order',
                                    'bad_digest_path', 'failed_terminal', 'terminal_extra',
                                    'configuration_hash', 'device_rng'])
def test_contradictory_fabricated_evidence_never_passes(fabricated_heldout, monkeypatch, tamper):
    path, study = fabricated_heldout
    args = audit_args(study)
    event_path, terminal_path = path / 'events.jsonl', path / 'terminal.json'
    original_events, original_terminal = event_path.read_bytes(), terminal_path.read_bytes()
    rows = [json.loads(line) for line in original_events.splitlines()]
    terminal = json.loads(original_terminal)
    summary = rows[-1]['event']['result']['heldout_evaluation']
    escaped_reads = []
    escaped = path / 'tensors' / '../../escape.f32le'
    if tamper == 'frozen_prediction':
        record = json.loads(args['prediction_bytes'])
        record['selection']['predictions']['1:attn:last'] += 1.0
        args['prediction_bytes'] = load_auditor().canonical(record)
    elif tamper == 'freeze_pin':
        args['expected_freeze_sha256'] = '0' * 64
    elif tamper == 'nonce':
        args['split_nonce'] = bytes(32)
    elif tamper == 'commitment':
        args['expected_commitment'] = '0' * 64
    elif tamper == 'selected_site':
        summary['selected_site'] = '1:attn:last'
    elif tamper == 'prediction_error':
        summary['sites'][0]['families'][0]['prediction_error'] += 1.0
    elif tamper == 'zero_error':
        summary['sites'][0]['zero_predictor']['mae'] += 1.0
    elif tamper == 'clean_margin':
        summary['clean_cases'][0]['clean_margin'] += 1.0
    elif tamper == 'pair_effect':
        event = next(row['event'] for row in rows if row['event']['kind'] == 'present_site_metrics')
        event['metrics']['pairs'][0]['signed_target_shift'] += 1.0
    elif tamper == 'landing':
        next(row['event'] for row in rows if row['event']['kind'] == 'patch')['post_action_matches_donor_bits'] = False
    elif tamper == 'metric_order':
        rows[-17], rows[-16] = rows[-16], rows[-17]
        rows[-17]['index'], rows[-16]['index'] = rows[-16]['index'], rows[-17]['index']
    elif tamper == 'bad_digest_path':
        next(row for row in rows if row['event']['kind'] == 'clean_capture')['tensors']['last_logits']['sha256'] = '../../escape'
        (path.parent / 'escape.f32le').write_bytes(b'private sentinel')
    elif tamper == 'terminal_extra':
        terminal['unexpected'] = 'contradictory'
    elif tamper == 'configuration_hash':
        rows[0]['event']['execution_configuration_sha256'] = '0' * 64
    elif tamper == 'device_rng':
        rows[-1]['event']['result']['device_rng'] = {'unexpected': True}
    else:
        terminal['status'] = 'failed'
    raw = b''.join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode() + b'\n' for row in rows)
    terminal['event_log_bytes'] = len(raw)
    terminal_raw = json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    original_open, original_lstat = Path.open, Path.lstat
    def monitored_open(target, mode='r', *args, **kwargs):
        if mode == 'rb' and target == escaped:
            escaped_reads.append(target)
        return original_open(target, mode, *args, **kwargs)
    def monitored_lstat(target, *args, **kwargs):
        if target == escaped:
            escaped_reads.append(target)
        return original_lstat(target, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', monitored_open)
    monkeypatch.setattr(Path, 'lstat', monitored_lstat)
    try:
        event_path.write_bytes(raw)
        terminal_path.write_bytes(terminal_raw)
        with pytest.raises((AssertionError, KeyError, ValueError)):
            load_auditor().inspect(path, **args)
        assert not escaped_reads
    finally:
        event_path.write_bytes(original_events)
        terminal_path.write_bytes(original_terminal)
        if tamper == 'bad_digest_path':
            (path.parent / 'escape.f32le').unlink()
