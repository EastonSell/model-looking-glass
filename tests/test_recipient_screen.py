import json
import math
from pathlib import Path
import struct

import pytest

from looking_glass import recipient_screen as screen


def fixed_cases():
    path = Path(__file__).resolve().parents[1] / 'experiments/distilgpt2_recipient_screen_v1.json'
    return json.loads(path.read_bytes())


def test_public_cases_are_complete_and_structurally_fixed():
    data = fixed_cases()
    cases = screen.validate_cases(data)
    assert len(cases) == 64
    broken = fixed_cases()
    broken['cases'][0]['recipient_id'] = broken['cases'][0]['giver_id']
    with pytest.raises(ValueError):
        screen.validate_cases(broken)
    broken = fixed_cases()
    broken['cases'][0]['input_ids'][2] += 1
    with pytest.raises(ValueError):
        screen.validate_cases(broken)


def test_behavior_gate_requires_bidirectional_success_and_reports_all_cells():
    cases = screen.validate_cases(fixed_cases())
    margins = {c['case_id']: 1.0 for c in cases}
    result = screen.summarize(cases, margins)
    assert result['gate_passed'] is True
    assert result['both_positive_pairs'] == 32
    assert result['template_both_positive'] == [16, 16]
    assert result['name_pairs_meeting_requirement'] == 8
    # Fail one direction in every template-0 pair: 48/64 individual margins
    # remain positive, but only one template supports paired behavior.
    for c in cases:
        if c['template'] == 0 and c['giver_index'] == 0:
            margins[c['case_id']] = -1.0
    result = screen.summarize(cases, margins)
    assert result['positive_cases'] == 48
    assert result['both_positive_pairs'] == 16
    assert result['gate_passed'] is False
    margins[cases[0]['case_id']] = 0.0
    assert screen.summarize(cases, margins)['gate_passed'] is False
    with pytest.raises(ValueError):
        screen.summarize(cases, {**margins, cases[0]['case_id']: float('nan')})


def test_gate_exact_paired_boundary_is_inclusive():
    cases = screen.validate_cases(fixed_cases())
    margins = {c['case_id']: 1.0 for c in cases}
    for c in cases:
        if c['giver_index'] == 0 and c['order'] == 0 and c['template'] == c['pair'] % 2:
            margins[c['case_id']] = -1.0
    result = screen.summarize(cases, margins)
    assert (result['both_positive_pairs'], result['template_both_positive']) == (24, [12, 12])
    assert result['gate_passed'] is True
    another = next(c for c in cases if c['pair'] == 0 and c['template'] == 1 and c['order'] == 1
                   and c['giver_index'] == 1)
    margins[another['case_id']] = 0.0
    assert screen.summarize(cases, margins)['gate_passed'] is False


def test_logits_have_deterministic_rank_and_signed_margin():
    result = screen.score_logits([1.0, 3.0, 3.0, -1.0], recipient=2, giver=0)
    assert result['margin'] == 2.0
    assert result['recipient_rank'] == 2  # exact tie goes to lower token ID
    assert result['giver_rank'] == 3
    assert result['top_token_id'] == 1
    assert result['recipient_log_probability'] > result['giver_log_probability']
    with pytest.raises(ValueError):
        screen.score_logits([float('nan'), 1.0], recipient=0, giver=1)


def test_log_probability_preserves_normalizer_for_large_equal_logits():
    score = screen.score_logits([2.0**100, 2.0**100], recipient=1, giver=0)
    assert score['recipient_log_probability'] == pytest.approx(-math.log(2))
    assert score['recipient_rank'] == 2


def test_bounded_returned_nonfinite_logits_are_saved_before_rejection(tmp_path):
    from scripts.recipient_screen_v1 import persist_logits
    case = screen.validate_cases(fixed_cases())[0]
    payload = bytearray(50257 * 4)
    struct.pack_into('<f', payload, 0, float('nan'))
    with (tmp_path / 'raw').open('xb') as raw, (tmp_path / 'journal').open('xb') as journal:
        with pytest.raises(ValueError):
            persist_logits(raw, journal, case, bytes(payload))
    assert (tmp_path / 'raw').read_bytes() == bytes(payload)
    events = [json.loads(x) for x in (tmp_path / 'journal').read_bytes().splitlines()]
    assert [x['event'] for x in events] == ['raw_recorded']


def test_complete_screen_persists_64_ordered_vectors_without_patching(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from scripts import recipient_screen_v1 as runner
    torch.set_num_threads(1)

    class FakeModel:
        def __call__(self, input_ids, **kwargs):
            ids = input_ids[0].tolist()
            recipient = ids[3] if ids[1] == ids[9] else ids[1]
            logits = torch.zeros((1, 14, 50257), dtype=torch.float32)
            logits[0, -1, recipient] = 1
            return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)

    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'synthetic'})
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'synthetic': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {})
    monkeypatch.setattr(runner.base, 'state_hash', lambda model: 'state')
    monkeypatch.setattr(runner.base, 'buffer_hash', lambda model: 'buffers')
    monkeypatch.setattr(runner.base, 'runtime', lambda model: {'synthetic': True})
    monkeypatch.setattr(runner.base, 'load_distilgpt2', lambda **kwargs:
                        SimpleNamespace(model=FakeModel(), tokenizer=None, metadata={'synthetic': True}))
    monkeypatch.setattr(runner.screen, 'validate_tokenizer', lambda tokenizer, cases: None)
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', lambda model: None)
    monkeypatch.setattr(runner, 'check_environment', lambda: runner.CPU_ENV)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['recipient_screen_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    runner.main()
    result = json.loads((output / 'result.json').read_bytes())
    assert result['recorded_forwards'] == 64
    assert result['summary']['gate_passed'] is True
    assert (output / 'last_logits.f32').stat().st_size == 64 * 50257 * 4
    events = [json.loads(x)['event'] for x in (output / 'attempt.jsonl').read_bytes().splitlines()]
    assert events == ['runtime_ready'] + ['start', 'raw_recorded', 'complete'] * 64
