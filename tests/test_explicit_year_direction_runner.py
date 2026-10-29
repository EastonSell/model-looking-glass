import json
import struct
import sys
from types import SimpleNamespace

import pytest
import torch

from looking_glass import explicit_year_direction as study
from scripts import explicit_year_direction_v1 as runner


def test_unbound_reservation_fails_before_model_work(monkeypatch):
    monkeypatch.setattr(runner, 'RESERVATION_VERSION', None)
    monkeypatch.setattr(runner, 'RESERVATION_SHA256', None)
    monkeypatch.setattr(runner, 'RESERVATION_MODULE', None)
    with pytest.raises(RuntimeError, match='reservation is not yet bound'):
        runner.check_exposure(study.make_inventory()['cases'])


def test_reservation_check_rejects_changed_digest_and_order(monkeypatch):
    import importlib
    rows = study.make_inventory()['cases']
    baseline = SimpleNamespace(INVENTORY_VERSION=study.EXPOSURE_VERSION,
                               inventory_digest=lambda: study.EXPOSURE_SHA256)
    reserved = SimpleNamespace(INVENTORY_VERSION='test-reservation', inventory_digest=lambda: 'a' * 64,
                               EXPLICIT_YEAR_DIRECTION_ROWS=tuple(tuple(r['input_ids']) for r in rows))
    monkeypatch.setattr(runner, 'RESERVATION_VERSION', 'test-reservation')
    monkeypatch.setattr(runner, 'RESERVATION_SHA256', 'a' * 64)
    monkeypatch.setattr(runner, 'RESERVATION_MODULE', 'test-reservation-module')
    modules = {'looking_glass.pretrained_exposures_v17': baseline, 'test-reservation-module': reserved}
    monkeypatch.setattr(importlib, 'import_module', lambda name: modules[name])
    assert runner.check_exposure(rows)['reservation_sha256'] == 'a' * 64
    reserved.inventory_digest = lambda: 'b' * 64
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(rows)
    reserved.inventory_digest = lambda: 'a' * 64
    reserved.EXPLICIT_YEAR_DIRECTION_ROWS = reserved.EXPLICIT_YEAR_DIRECTION_ROWS[::-1]
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(rows)


def test_nonfinite_returned_logits_are_saved_before_rejection(tmp_path):
    row = study.make_inventory()['cases'][0]
    payload = bytearray(runner.LOGITS_BYTES)
    struct.pack_into('<I', payload, 0, 0x7f800000)
    marker = {'recorded': False}
    with (tmp_path/'data').open('xb') as raw, (tmp_path/'log').open('xb') as log:
        with pytest.raises(ValueError):
            runner.persist_action(log, raw, row, 'clean', {'logits': bytes(payload)},
                                  {}, {}, lambda: None, marker)
    assert marker['recorded'] and (tmp_path/'data').read_bytes() == bytes(payload)


@pytest.mark.parametrize('fault', ['post_return', 'partial_write', 'return_count', 'helper_count'])
def test_interrupted_return_retains_available_logits_and_correct_counts(tmp_path, monkeypatch, fault):
    import inspect
    calls = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append(input_ids.shape[1])
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'fabricated'})
    monkeypatch.setattr(runner, 'check_exposure', lambda cases: {'fabricated': True})
    monkeypatch.setattr(runner, 'runtime', lambda model: {'fabricated': True})
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', lambda model: None)
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'state_hash', lambda model: 'state')
    monkeypatch.setattr(runner.base, 'buffer_hash', lambda model: 'buffers')
    monkeypatch.setattr(runner.base, 'load_distilgpt2', lambda **kwargs:
                        SimpleNamespace(model=Model(), tokenizer=None,
                                        metadata={'weights_sha256': runner.base.WEIGHTS_SHA256}))
    monkeypatch.setattr(study, 'validate_tokenizer', lambda tokenizer, cases: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['explicit_year_direction_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    lines, first = inspect.getsourcelines(runner.main)
    marker_lines = {'post_return': "phase = 'persist_and_validate'", 'return_count': 'returned = returned_before + 1',
                    'helper_count': 'payload_returned = True'}
    target = (first + next(i for i, line in enumerate(lines) if line.strip() == marker_lines[fault])
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
        def fail_first(stream, payload):
            if not fired:
                fired.append(True)
                stream.write(payload['logits'][:128])
                stream.flush()
                raise OSError('synthetic partial write')
            return original(stream, payload)
        monkeypatch.setattr(runner.base, '_write_payload', fail_first)
        with pytest.raises(OSError, match='synthetic partial write'):
            runner.main()
    failure = json.loads((output/'failure.json').read_bytes())
    assert fired and len(calls) == 1
    assert failure['model_call_return_observed'] is True and failure['payload_returned'] is True
    assert failure['model_call_returns_observed'] == 1 and failure['returned_forwards'] == 1
    assert failure['payload_recorded'] is False and failure['partial_payload_recorded'] is True
    partial = list(output.glob('*.partial.f32'))
    assert len(partial) == 1 and partial[0].read_bytes() == bytes(runner.LOGITS_BYTES)
    if fault == 'partial_write':
        assert len(failure['raw_files']) == 1 and failure['raw_files'][0]['bytes'] == 128


def test_family_planner_maps_the_three_named_wordings():
    def raw(scale):
        values = [0.25] * (13 * 768)
        values[-768:] = [scale * (i + 1) / 1024 for i in range(768)]
        return struct.pack(f'<{len(values)}f', *values)
    noops = {c: {'low': raw(0.0), 'high': raw(i + 1.0)} for i, c in enumerate(study.CONDITIONS)}
    plan = runner.plan_family(noops)
    assert set(plan['norms']) == set(study.CONDITIONS)
    assert set(plan['plans']) == set(study.CONDITIONS)
    for c in study.CONDITIONS:
        for role in study.ROLES:
            assert set(plan['plans'][c][role]) == set(study.ACTIONS)
            assert plan['plans'][c][role]['year_final']['requested'] == plan['dose']


@pytest.mark.parametrize('gate_pass', [False, True])
def test_full_synthetic_schedule_runs_for_both_clean_gate_outcomes(tmp_path, monkeypatch, gate_pass):
    rows = study.make_inventory()['cases']
    by_input = {tuple(row['input_ids']): row for row in rows}
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
            if row['year_role'] == 'high':
                hidden[0, -1] = torch.arange(1, 769, dtype=torch.float32) / 1024
            self.block(hidden)
            logits = torch.zeros((1, length, runner.VOCAB), dtype=torch.float32)
            if gate_pass:
                if row['condition'] == 'after':
                    logits[0, -1, row['local_above_ids']] = 2.0
                    if row['year_role'] == 'low':
                        logits[0, -1, row['middle_ids']] = 2.0
                elif row['condition'] == 'before':
                    logits[0, -1, row['local_below_ids']] = 2.0
                    if row['year_role'] == 'high':
                        logits[0, -1, row['middle_ids']] = 2.0
            return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)

    model = Model()
    def no_hooks(value):
        if value.block._forward_hooks:
            raise RuntimeError('unexpected test hook')
    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'synthetic'})
    monkeypatch.setattr(runner, 'check_exposure', lambda cases: {'synthetic': True})
    monkeypatch.setattr(runner, 'runtime', lambda model: {'synthetic': True})
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', no_hooks)
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'synthetic': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {'synthetic': True})
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
    monkeypatch.setattr(sys, 'argv', ['explicit_year_direction_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    runner.main()
    result = json.loads((output/'result.json').read_bytes())
    assert result['status'] == 'complete' and result['submitted_forwards'] == 336
    assert result['raw_payload_bytes'] == 86_714_688
    assert result['interventions_performed'] is True and len(result['records']) == 336
    assert result['clean_summary']['comparator_gate_passed'] is gate_pass
    assert result['intervention_summary']['generic_pattern_supported'] is False
    assert result['intervention_summary']['comparator_direction_supported'] is (False if gate_pass else None)
    assert len(result['family_doses']) == 8
    expected = [(row['case_id'], action) for row in rows for action in ('clean', 'noop')]
    for row in rows:
        shift = row['rotation']
        expected.extend((row['case_id'], action)
                        for action in study.ACTIONS[shift:] + study.ACTIONS[:shift] + ('clean_after',))
    assert [(r['case_id'], r['action']) for r in result['records']] == expected
    events = [json.loads(line)['event'] for line in (output/'attempt.jsonl').read_bytes().splitlines()]
    assert all(events.count(event) == 336 for event in ('start', 'raw_recorded', 'complete'))
    assert events.count('family_dose') == 8 and not model.block._forward_hooks
