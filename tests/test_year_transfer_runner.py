import json
import struct
from types import SimpleNamespace

import pytest

from looking_glass import year_transfer as study


def test_reservation_binding_rejects_changed_digest_or_rows(monkeypatch):
    import importlib
    from scripts import year_transfer_v1 as runner
    cases = study.make_inventory()['cases']
    baseline = SimpleNamespace(INVENTORY_VERSION=study.EXPOSURE_VERSION,
                               inventory_digest=lambda: study.EXPOSURE_SHA256)
    reservation = SimpleNamespace(
        INVENTORY_VERSION='distilgpt2-development-exposures-v16',
        inventory_digest=lambda: '275a16d49a386b1de247bff0f9031e92105d1b76f2493352090bf27c37d58843',
        YEAR_TRANSFER_ROWS=tuple(tuple(row['input_ids']) for row in cases),
    )
    modules = {'looking_glass.pretrained_exposures_v15': baseline,
               'looking_glass.pretrained_exposures_v16': reservation}
    monkeypatch.setattr(importlib, 'import_module', lambda name: modules[name])
    exposure = runner.check_exposure(cases)
    assert exposure['reservation_version'] == reservation.INVENTORY_VERSION
    assert exposure['reservation_sha256'] == reservation.inventory_digest()
    reservation.inventory_digest = lambda: '0' * 64
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(cases)
    reservation.inventory_digest = lambda: exposure['reservation_sha256']
    reservation.YEAR_TRANSFER_ROWS = reservation.YEAR_TRANSFER_ROWS[:-1]
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(cases)


def test_returned_nonfinite_logits_are_saved_before_rejection(tmp_path):
    from scripts import year_transfer_v1 as runner
    row = study.make_inventory()['cases'][0]
    payload = bytearray(runner.LOGITS_BYTES)
    struct.pack_into('<I', payload, 0, 0x7f800000)
    marker = {'recorded': False}
    with (tmp_path/'data').open('xb') as raw, (tmp_path/'log').open('xb') as log:
        with pytest.raises(ValueError):
            runner.persist_action(log, raw, row, 'clean', {'logits': bytes(payload)},
                                  {}, {}, lambda: None, marker)
    assert marker['recorded']
    assert (tmp_path/'data').read_bytes() == bytes(payload)
    assert [json.loads(line)['event'] for line in (tmp_path/'log').read_bytes().splitlines()] == ['raw_recorded']


def test_post_return_helper_failure_preserves_available_logits(monkeypatch):
    import torch
    from scripts import year_transfer_v1 as runner
    class Model:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    calls = []
    def check(_):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('synthetic post-return hook check failure')
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', check)
    observation = {}
    with pytest.raises(RuntimeError):
        runner.run_forward(Model(), torch.zeros((1, 12), dtype=torch.long), 'clean', observation=observation)
    assert observation['model_call_return_observed']
    assert len(observation['payload']['logits']) == runner.LOGITS_BYTES


@pytest.mark.parametrize('fault', ['post_return', 'partial_write', 'return_count', 'helper_count'])
def test_interrupted_after_payload_return_preserves_bounded_logits(tmp_path, monkeypatch, fault):
    import inspect
    import sys
    import torch
    from scripts import year_transfer_v1 as runner
    calls = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append(input_ids.shape[1])
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    model = Model()
    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'fabricated'})
    monkeypatch.setattr(runner, 'check_exposure', lambda cases: {'fabricated': True})
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
    monkeypatch.setattr(sys, 'argv', ['year_transfer_v1.py', '--cache-dir', str(tmp_path),
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
    assert any(item['file'] == partial[0].name for item in failure['partial_files'])
    if fault == 'partial_write':
        assert len(failure['raw_files']) == 1 and failure['raw_files'][0]['bytes'] == 128


def test_partial_helper_payload_is_journaled_and_preserved(tmp_path):
    from scripts import year_transfer_v1 as runner
    data = bytes([17, 0, 0, 0]) * 8
    marker = {'partial': False}
    observation = {'model_call_return_observed': False, 'payload': {'block_before': data}}
    with (tmp_path/'log').open('xb') as log:
        saved = runner.persist_partial(log, tmp_path/'partial.f32', 'test-case', 'noop', observation, marker)
    events = [json.loads(line) for line in (tmp_path/'log').read_bytes().splitlines()]
    assert saved == len(data) and marker['partial']
    assert (tmp_path/'partial.f32').read_bytes() == data
    assert events[0]['event'] == 'raw_recorded_partial'
    assert events[0]['model_call_return_observed'] is False


def test_position_zero_check_uses_each_same_wording_noop_pair():
    from scripts import year_transfer_v1 as runner
    rows = study.make_inventory()['cases']
    caches = {row['case_id']: {'before': bytes(len(row['input_ids']) * 768 * 4)} for row in rows}
    assert runner.check_position_zero(rows, caches) == 24
    changed = bytearray(caches[rows[1]['case_id']]['before'])
    changed[0] = 1
    caches[rows[1]['case_id']]['before'] = bytes(changed)
    with pytest.raises(ValueError):
        runner.check_position_zero(rows, caches)


def test_fresh_random_eager_gpt2_executes_clean_noop_and_replacement(monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from scripts import year_transfer_v1 as runner
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, 'VOCAB', 32)
    monkeypatch.setattr(runner, 'LOGITS_BYTES', 32 * 4)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(202)
        config = GPT2Config(vocab_size=32, n_embd=768, n_head=12, n_layer=4, n_positions=16,
                            use_cache=False, attn_implementation='eager')
        model = GPT2LMHeadModel(config).eval()
        low = torch.tensor([list(range(12))], dtype=torch.long)
        high = low.clone()
        high[0, 5] = 14
        clean, _ = runner.run_forward(model, low, 'clean')
        low_noop, low_witness = runner.run_forward(model, low, 'noop')
        high_noop, high_witness = runner.run_forward(model, high, 'noop')
        low_witness.verify_complete()
        high_witness.verify_complete()
        assert clean['logits'] == low_noop['logits']
        assert low_noop['block_before'][:768*4] == high_noop['block_before'][:768*4]
        noops = {condition: {'low': low_noop['block_before'], 'high': high_noop['block_before']}
                 for condition in study.CONDITIONS}
        planned = runner.intervention.plan_family(noops)['plans']['forward']['low']
        self_out, self_witness = runner.run_forward(model, low, 'self',
                                                   expected_before=low_noop['block_before'], plan=planned['self'])
        self_witness.verify_complete()
        assert self_out['logits'] == clean['logits']
        patched, witness = runner.run_forward(model, low, 'year_final',
                                              expected_before=low_noop['block_before'], plan=planned['year_final'])
        witness.verify_complete()
        assert patched['block_before'] == low_noop['block_before']
        assert witness.observed['achieved'] == planned['year_final']['achieved']
        runner.assert_no_forward_hooks(model)


@pytest.mark.parametrize('scenario', ['neither', 'preference_only', 'comparator_only'])
def test_complete_synthetic_schedules_preserve_both_gate_branches(tmp_path, monkeypatch, scenario):
    import sys
    import torch
    from scripts import year_transfer_v1 as runner
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
            if scenario != 'neither' and row['year_role'] == 'low':
                values = ({'forward': 2.0, 'reverse': 1.0, 'neutral': 1.0} if scenario == 'preference_only'
                          else {'forward': 2.0, 'reverse': -1.0, 'neutral': 0.0})
                value = values[row['condition']]
                logits[0, -1, row['middle_ids']] = value
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
    monkeypatch.setattr(runner.study, 'validate_tokenizer', lambda tokenizer, cases: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['year_transfer_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    runner.main()
    result = json.loads((output/'result.json').read_bytes())
    gate_pass = scenario != 'neither'
    assert result['status'] == 'complete'
    assert result['interventions_performed'] is gate_pass
    assert result['submitted_forwards'] == (336 if gate_pass else 96)
    assert result['raw_payload_bytes'] == (87206208 if gate_pass else 23230848)
    assert result['clean_summary']['preference_gate_passed'] is (scenario == 'preference_only')
    assert result['clean_summary']['comparator_gate_passed'] is (scenario == 'comparator_only')
    assert len(result['records']) == (336 if gate_pass else 96)
    expected = [(row['case_id'], action) for row in rows for action in ('clean', 'noop')]
    if gate_pass:
        assert result['intervention_summary']['preference_pattern_supported'] is (False if scenario == 'preference_only' else None)
        assert result['intervention_summary']['comparator_pattern_supported'] is (False if scenario == 'comparator_only' else None)
        assert len(result['family_doses']) == 8
        item = next(row for row in result['records'] if row['action'] == 'year_final')
        assert len(bytes.fromhex(item['intervention']['expected_before_hex'])) == 768 * 4
        assert len(bytes.fromhex(item['intervention']['expected_after_hex'])) == 768 * 4
        arms = ('self', 'year_final', 'year_at_zero', 'rotated_final')
        for row in rows:
            rotation = row['rotation']
            expected.extend((row['case_id'], action)
                            for action in arms[rotation:] + arms[:rotation] + ('clean_after',))
    else:
        assert result['intervention_summary'] is None
        assert result['family_doses'] == []
    assert [(record['case_id'], record['action']) for record in result['records']] == expected
    records = [json.loads(line) for line in (output/'attempt.jsonl').read_bytes().splitlines()]
    events = [record['event'] for record in records]
    assert events.count('start') == len(expected)
    assert events.count('raw_recorded') == len(expected)
    assert events.count('complete') == len(expected)
    assert events.count('clean_gates') == events.count('position_zero_check') == 1
    assert events.count('family_dose') == (8 if gate_pass else 0)
    if gate_pass:
        start = next(record for record in records if record['event'] == 'start' and record['action'] == 'year_final')
        assert start['intervention']['expected_before_hex'] == item['intervention']['expected_before_hex']
        assert start['intervention']['expected_after_hex'] == item['intervention']['expected_after_hex']
    assert not model.block._forward_hooks
