import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from looking_glass import year_label_control_v2 as study
from scripts import year_label_control_v2 as runner


def test_empty_bytecode_prefix_rejects_existing_entries_before_import(tmp_path, monkeypatch):
    prefix = tmp_path/'pycache'
    prefix.mkdir()
    monkeypatch.setattr(sys, 'pycache_prefix', str(prefix))
    assert runner.checked_empty_pycache_prefix() == str(prefix.resolve())
    (prefix/'stale.pyc').write_bytes(b'stale')
    with pytest.raises(RuntimeError, match='empty'):
        runner.checked_empty_pycache_prefix()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    child = subprocess.run([sys.executable, '-B', '-X', f'pycache_prefix={prefix}', runner.__file__, '--help'],
                           env=env, text=True, capture_output=True)
    assert child.returncode != 0 and 'pycache prefix must be empty' in child.stderr


def test_runtime_rejects_changed_bytecode_prefix(tmp_path, monkeypatch):
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(runner, 'PYCACHE_AT_IMPORT', str(first.resolve()))
    monkeypatch.setattr(sys, 'pycache_prefix', str(second))
    with pytest.raises(RuntimeError, match='pycache prefix changed'):
        runner.environment()


def test_historical_reservation_must_bind_exact_unique_rows(monkeypatch):
    import importlib
    rows = study.select_cases(study.v1.make_inventory()['cases'])
    baseline = SimpleNamespace(INVENTORY_VERSION=study.BASELINE_VERSION, INVENTORY_SHA256=study.BASELINE_SHA256)
    payload = {'predecessor': {'version': 'distilgpt2-development-exposures-v25',
                              'sha256': '32cc4bcbe4b482ded122715c5f96c6c5c765a76a1ef3532360cdf9890827f144'},
               'records': [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                            'token_ids': row['input_ids']} for row in rows]}
    all_rows = tuple(tuple(row['input_ids']) for row in study.v1.make_inventory()['cases'])
    reservation = SimpleNamespace(INVENTORY_VERSION='test-reservation', YEAR_LABEL_CONTROL_ROWS=all_rows,
                                  _checked_inventory=lambda: (payload, 'a'*64))
    monkeypatch.setattr(runner, 'RESERVATION_VERSION', 'test-reservation')
    monkeypatch.setattr(runner, 'RESERVATION_SHA256', 'a'*64)
    monkeypatch.setattr(runner, 'RESERVATION_MODULE', 'test-reservation-module')
    modules = {'looking_glass.pretrained_exposures_v24': baseline, 'test-reservation-module': reservation}
    monkeypatch.setattr(importlib, 'import_module', lambda name: modules[name])
    assert runner.check_exposure(rows)['reservation_sha256'] == 'a'*64
    payload['records'][0]['token_ids'] = [0]
    with pytest.raises(RuntimeError, match='unique and exact'):
        runner.check_exposure(rows)
    payload['records'][0]['token_ids'] = rows[0]['input_ids']
    payload['records'].append(dict(payload['records'][0]))
    with pytest.raises(RuntimeError, match='unique and exact'):
        runner.check_exposure(rows)
    payload['records'].pop()
    reservation._checked_inventory = lambda: (payload, 'b'*64)
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(rows)


def test_prior_segment_rejects_bad_range_and_changed_hash(tmp_path):
    path = tmp_path/'sample.f32'
    path.write_bytes(b'prefixabcdefghsuffix')
    slot = {'offset': 6, 'bytes': 8, 'sha256': hashlib.sha256(b'abcdefgh').hexdigest()}
    assert runner._prior_segment(path, slot, 8) == b'abcdefgh'
    with pytest.raises(ValueError, match='differs'):
        runner._prior_segment(path, {**slot, 'offset': 7}, 8)
    with pytest.raises(ValueError, match='invalid'):
        runner._prior_segment(path, {**slot, 'bytes': 9}, 8)


def test_prior_receipt_mismatch_stops_before_using_records(tmp_path):
    (tmp_path/'supervisor.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='receipt differs'):
        runner.load_prior(tmp_path, study.select_cases(study.v1.make_inventory()['cases']))


def _prepare(monkeypatch, tmp_path, model):
    rows = study.select_cases(study.v1.make_inventory()['cases'])
    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'fabricated'})
    monkeypatch.setattr(runner, 'check_exposure', lambda cases: {'fabricated': True})
    monkeypatch.setattr(runner, 'load_prior', lambda path, cases: {
        r['case_id']: {'clean': bytes(runner.LOGITS_BYTES),
                       'before': bytes(len(r['input_ids'])*runner.intervention.ROW_BYTES)} for r in rows})
    monkeypatch.setattr(runner, 'environment', lambda: {'fabricated': True})
    monkeypatch.setattr(runner, 'checked_empty_pycache_prefix', lambda: str(tmp_path))
    monkeypatch.setattr(runner, 'runtime', lambda model: {'fabricated': True})
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', lambda model: None)
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'state_hash', lambda model: 'state')
    monkeypatch.setattr(runner.base, 'buffer_hash', lambda model: 'buffers')
    monkeypatch.setattr(runner.base, 'load_distilgpt2', lambda **kwargs:
                        SimpleNamespace(model=model, tokenizer=None,
                                        metadata={'weights_sha256': runner.base.WEIGHTS_SHA256}))
    monkeypatch.setattr(study, 'validate_tokenizer', lambda tokenizer, cases: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['year_label_control_v2.py', '--cache-dir', str(tmp_path),
                                     '--prior-attempt-dir', str(tmp_path), '--output-dir', str(output)])
    return rows, output


@pytest.mark.parametrize('fault', ['post_return', 'return_count', 'helper_count', 'partial_write'])
def test_returned_payload_and_counter_survive_failure(tmp_path, monkeypatch, fault):
    import inspect
    calls = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append(1)
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    _, output = _prepare(monkeypatch, tmp_path, Model())
    lines, first = inspect.getsourcelines(runner.main)
    markers = {'post_return': "phase = 'persist_and_validate'", 'return_count': 'returned = returned_before + 1',
               'helper_count': 'payload_returned = True'}
    target = (first + next(i for i, line in enumerate(lines) if line.strip() == markers[fault])
              if fault != 'partial_write' else None)
    fired = []
    def trace(frame, event, argument):
        if (event == 'line' and frame.f_code.co_name == 'perform' and frame.f_code.co_filename == runner.__file__
                and frame.f_lineno == target and not fired):
            fired.append(True)
            raise MemoryError('synthetic post-return failure')
        return trace
    if fault != 'partial_write':
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with pytest.raises(MemoryError, match='synthetic post-return failure'):
                runner.main()
        finally:
            sys.settrace(previous)
    else:
        original = runner.base._write_payload
        def fail_once(stream, payload):
            if not fired:
                fired.append(True)
                stream.write(payload['logits'][:128])
                stream.flush()
                raise OSError('synthetic partial write')
            return original(stream, payload)
        monkeypatch.setattr(runner.base, '_write_payload', fail_once)
        with pytest.raises(OSError, match='synthetic partial write'):
            runner.main()
    failure = json.loads((output/'failure.json').read_bytes())
    assert fired and len(calls) == 1
    assert failure['model_call_returns_observed'] == failure['returned_forwards'] == 1
    assert failure['model_call_return_observed'] is True and failure['payload_returned'] is True
    assert failure['partial_payload_recorded'] is True
    assert next(output.glob('*.partial.f32')).read_bytes() == bytes(runner.LOGITS_BYTES)
    if fault == 'partial_write':
        assert len(failure['raw_files']) == 1 and failure['raw_files'][0]['bytes'] == 128


def test_post_return_fidelity_failure_retains_the_complete_raw_payload(tmp_path, monkeypatch):
    by_input = {tuple(row['input_ids']): row for row in study.v1.make_inventory()['cases']}
    calls = []
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)
    class Model:
        def __init__(self):
            self.block = Block()
            self.transformer = SimpleNamespace(h=[None, None, None, self.block])
        def __call__(self, input_ids, **kwargs):
            calls.append(1)
            row = by_input[tuple(input_ids[0].tolist())]
            length = input_ids.shape[1]
            hidden = torch.zeros((1, length, 768), dtype=torch.float32)
            hidden[0, -1, 5] = 10.0 * row['anchor_index']
            hidden[0, -1, 9] = 5.0 * row['label_index']
            self.block(hidden)
            return SimpleNamespace(logits=torch.zeros((1, length, runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    torch.set_num_threads(1)
    model = Model()
    _, output = _prepare(monkeypatch, tmp_path, model)
    original = runner.intervention.BlockOutputWitness.verify_complete
    def reject_after_check(self):
        original(self)
        if self.plan is not None:
            raise ValueError('synthetic actual-fidelity failure')
    monkeypatch.setattr(runner.intervention.BlockOutputWitness, 'verify_complete', reject_after_check)
    with pytest.raises(ValueError, match='synthetic actual-fidelity failure'):
        runner.main()
    failure = json.loads((output/'failure.json').read_bytes())
    events = [json.loads(line) for line in (output/'attempt.jsonl').read_bytes().splitlines()]
    assert len(calls) == failure['submitted_forwards'] == failure['model_call_returns_observed'] == 73
    assert failure['completed_records'] == 72 and failure['payload_recorded'] is True
    assert len([e for e in events if e['event'] == 'raw_recorded']) == 73
    assert len([e for e in events if e['event'] == 'complete']) == 72
    assert not model.block._forward_hooks


def test_full_fabricated_schedule_checks_all_cases_and_records_fidelity(tmp_path, monkeypatch):
    all_rows = study.v1.make_inventory()['cases']
    by_input = {tuple(row['input_ids']): row for row in all_rows}
    torch.set_num_threads(1)
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)
    class Model:
        def __init__(self):
            self.block = Block()
            self.transformer = SimpleNamespace(h=[None, None, None, self.block])
        def __call__(self, input_ids, **kwargs):
            row = by_input[tuple(input_ids[0].tolist())]
            length = input_ids.shape[1]
            hidden = torch.zeros((1, length, 768), dtype=torch.float32)
            hidden[0, -1, 5] = 10.0 * row['anchor_index']
            hidden[0, -1, 9] = 5.0 * row['label_index']
            observed = self.block(hidden)[0]
            logits = torch.zeros((1, length, runner.VOCAB), dtype=torch.float32)
            logits[0, -1, row['middle_ids']] = observed[0, -1, 5] * 0.03 + observed[0, -1, 9] * 0.05
            return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)
    model = Model()
    rows, output = _prepare(monkeypatch, tmp_path, model)
    runner.main()
    result = json.loads((output/'result.json').read_bytes())
    manifest = json.loads((output/'manifest.json').read_bytes())
    assert result['status'] == 'complete' and result['submitted_forwards'] == 288
    assert result['raw_payload_bytes'] == 77_802_624 and len(result['records']) == 288
    assert manifest['planned_forwards'] == 288 and len(manifest['cases']) == 36
    assert [x['family_index'] for x in result['family_doses']] == [1, 2, 3]
    assert [x['family_index'] for x in result['intervention_summary']['families']] == [1, 2, 3]
    assert len(result['cross_run_baseline_comparison']) == 36
    assert all(type(x['clean_logits_match_v1']) is bool and type(x['noop_block_output_matches_v1']) is bool
               for x in result['cross_run_baseline_comparison'])
    expected = [(row['case_id'], action) for row in rows for action in ('clean', 'noop')]
    for row in rows:
        shift = row['rotation']
        expected.extend((row['case_id'], action)
                        for action in study.ACTIONS[shift:] + study.ACTIONS[:shift] + ('clean_after',))
    assert [(item['case_id'], item['action']) for item in result['records']] == expected
    nonself = [r for r in result['records'] if r['action'] in study.ACTIONS[1:]]
    assert len(nonself) == 144
    assert all(x['intervention']['certificate'] == x['intervention']['observed_certificate'] for x in nonself)
    assert all(len(x['intervention']['displacement_f64_hex']) == 768*16 for x in nonself)
    events = [json.loads(line)['event'] for line in (output/'attempt.jsonl').read_bytes().splitlines()]
    assert all(events.count(event) == 288 for event in ('start', 'raw_recorded', 'complete'))
    assert events.count('cross_run_baseline_comparison') == 1 and events.count('family_dose') == 3
    assert not model.block._forward_hooks
