import json
import math
from pathlib import Path
import struct

import pytest

from looking_glass import greater_than_screen as screen


def cases():
    path = Path(__file__).resolve().parents[1] / 'experiments/distilgpt2_greater_than_screen_v1.json'
    return json.loads(path.read_bytes())


def test_fixed_cases_reject_candidate_or_token_mutation():
    assert len(screen.validate_cases(cases())) == 48
    data = cases()
    data['cases'][0]['upper_ids'][0] = data['cases'][0]['lower_ids'][0]
    with pytest.raises(ValueError):
        screen.validate_cases(data)
    data = cases()
    data['cases'][0]['input_ids'][-1] += 1
    with pytest.raises(ValueError):
        screen.validate_cases(data)


def test_equal_sized_sets_have_zero_log_ratio_and_correct_full_mass():
    score = screen.score_logits([0.0] * 32, lower=list(range(8)), upper=list(range(8, 16)), equal=16,
                                numeric=list(range(32)))
    assert score['log_mass_ratio'] == pytest.approx(0.0)
    assert score['lower_full_mass'] == pytest.approx(8/32)
    assert score['upper_full_mass'] == pytest.approx(8/32)
    assert score['equal_full_mass'] == pytest.approx(1/32)
    assert score['upper_conditional_fraction'] == pytest.approx(0.5)
    assert score['aligned_01_99_full_mass'] == pytest.approx(31/32)
    assert score['zero_00_next_token_mass'] == pytest.approx(1/32)
    assert score['other_vocabulary_mass'] == pytest.approx(0.0)


def test_upper_shift_and_extreme_equal_logits_are_stable():
    values = [0.0] * 32
    values[8:16] = [2.0] * 8
    score = screen.score_logits(values, lower=list(range(8)), upper=list(range(8, 16)), equal=16)
    assert score['log_mass_ratio'] == pytest.approx(2.0)
    assert score['upper_conditional_fraction'] == pytest.approx(1/(1+math.exp(-2)))
    score = screen.score_logits([2.0**100] * 32, lower=list(range(8)), upper=list(range(8, 16)), equal=16)
    assert score['log_mass_ratio'] == pytest.approx(0.0)
    assert score['upper_full_mass'] == pytest.approx(8/32)
    with pytest.raises(ValueError):
        screen.score_logits([float('nan')] * 32, lower=list(range(8)), upper=list(range(8, 16)), equal=16)


def test_fixed_contrasts_include_every_triple_and_sign():
    fixed = screen.validate_cases(cases())
    values = {'forward': 2.0, 'reverse': -1.0, 'neutral': 0.0}
    metrics = {case['case_id']: values[case['condition']] for case in fixed}
    result = screen.summarize(fixed, metrics)
    assert len(result['triples']) == 16
    assert result['mean_forward_minus_reverse'] == 3.0
    assert result['mean_forward_minus_neutral'] == 2.0
    assert result['mean_reverse_minus_neutral'] == -1.0
    assert result['positive_forward_minus_reverse'] == 16


def test_nonfinite_returned_vector_is_saved_before_validation(tmp_path):
    from scripts.greater_than_screen_v1 import persist_logits
    case = screen.validate_cases(cases())[0]
    raw = bytearray(50257 * 4)
    struct.pack_into('<f', raw, 0, float('nan'))
    with (tmp_path/'raw').open('xb') as stream, (tmp_path/'journal').open('xb') as journal:
        with pytest.raises(ValueError):
            persist_logits(stream, journal, case, bytes(raw))
    assert (tmp_path/'raw').read_bytes() == bytes(raw)
    assert [json.loads(line)['event'] for line in (tmp_path/'journal').read_bytes().splitlines()] == ['raw_recorded']


def test_synthetic_screen_persists_all_48_vectors_and_triples(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from scripts import greater_than_screen_v1 as runner
    torch.set_num_threads(1)

    class FakeModel:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 50257), dtype=torch.float32),
                                   past_key_values=None, attentions=None, hidden_states=None)

    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'synthetic'})
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'synthetic': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {})
    monkeypatch.setattr(runner.base, 'state_hash', lambda model: 'state')
    monkeypatch.setattr(runner.base, 'buffer_hash', lambda model: 'buffers')
    monkeypatch.setattr(runner.base, 'runtime', lambda model: {'synthetic': True})
    monkeypatch.setattr(runner.base, 'load_distilgpt2', lambda **kwargs:
                        SimpleNamespace(model=FakeModel(), tokenizer=None, metadata={'synthetic': True}))
    monkeypatch.setattr(runner.screen, 'validate_tokenizer', lambda tokenizer, items: None)
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', lambda model: None)
    monkeypatch.setattr(runner, 'check_environment', lambda: runner.CPU_ENV)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['greater_than_screen_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    runner.main()
    result = json.loads((output / 'result.json').read_bytes())
    assert result['recorded_forwards'] == 48
    assert result['summary']['mean_forward_minus_reverse'] == 0.0
    assert len(result['summary']['triples']) == 16
    assert (output / 'last_logits.f32').stat().st_size == 48 * 50257 * 4
    events = [json.loads(x)['event'] for x in (output / 'attempt.jsonl').read_bytes().splitlines()]
    assert events == ['runtime_ready'] + ['start', 'raw_recorded', 'complete'] * 48
