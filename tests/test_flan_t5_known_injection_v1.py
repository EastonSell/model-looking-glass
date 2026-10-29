"""Fabricated-only tests of the fixed T5 injection and evidence schedule."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'looking_glass/flan_t5_trace.py').is_file():
    sys.path.insert(0, str(ROOT))
    from tools import flan_t5_known_injection_v1 as study
else:
    sys.path.insert(0, str(ROOT / 'teams/flan_t5_sync_merged_check'))
    spec = importlib.util.spec_from_file_location('known_injection_tested', Path(__file__).with_name('run_known_injection.py'))
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)


def tiny_model(layers=8):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = T5Config(vocab_size=96, d_model=16, d_kv=8, d_ff=32, num_layers=layers,
                      num_decoder_layers=layers, num_heads=2, dropout_rate=0.0,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    config._attn_implementation = 'eager'
    return T5ForConditionalGeneration(config).eval()


def case():
    return {'case_id': 'fabricated-01', 'split': 'discovery', 'input_ids': [9, 8, 7, 1]}


def test_synthetic_two_stack_schedule_and_landing(tmp_path):
    result = study.execute(tiny_model(), [case()], [0, 3], tmp_path / 'run', vocab=96, hidden=16)
    assert result['forwards'] == 10
    assert len(result['trials']) == 2
    for trial in result['trials']:
        assert trial['first_change_index'] == trial['target_index']
        assert trial['same_site_restore_matches_clean'] and trial['self_matches_injected']
        assert trial['injection_l2'] > 0.0
    events = [json.loads(line) for line in (tmp_path / 'run/attempt.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'forward_start' for e in events) == 10
    assert sum(e['event'] == 'forward_complete' for e in events) == 10
    for event in (e for e in events if e['event'] == 'forward_start'):
        assert set(event['inputs']) == {'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'}
        assert all(v['dtype'] == 'torch.int64' and v['shape'][0] == 1 for v in event['inputs'].values())
    assert events[-1]['event'] == 'complete'


def test_rejects_wrong_model_layer_count_before_output_or_forward(tmp_path, monkeypatch):
    def unexpected_forward(*args, **kwargs):
        raise AssertionError('a model forward must not run')

    model=tiny_model(layers=6)
    monkeypatch.setattr(model,'forward',unexpected_forward)
    root=tmp_path/'wrong-model-dimensions'
    with pytest.raises(ValueError, match='unexpected model dimensions'):
        study.execute(model,[case()],[0,3],root,vocab=96,hidden=16)
    assert not root.exists()


def test_landing_rejects_other_position_or_wrong_source():
    before = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    value = torch.full((8,), 7.0)
    after = before.clone()
    after[0, 1] = value
    assert study._landed(before, after, value)
    after[0, 2, 0] += 1
    assert not study._landed(before, after, value)
    after = before.clone()
    assert not study._landed(before, after, value)


def test_exhausted_budget_preserves_started_incomplete_record(tmp_path):
    root = tmp_path / 'too-small'
    with pytest.raises(ValueError, match='incomplete clean trace or evidence'):
        study.execute(tiny_model(), [case()], [0, 3], root, vocab=96, hidden=16, raw_limit=4)
    records = [json.loads(line) for line in (root / 'attempt.jsonl').read_text().splitlines()]
    assert records[-1]['event'] == 'forward_incomplete'
    assert records[-1]['return_observed'] is True
    assert records[-2]['event'] == 'forward_observation' and records[-2]['unavailable']
    assert not (root / 'result.json').exists()


def test_fixed_random_baselines_from_public_identifiers():
    groups = [('discovery-' + f'{i:02}', 'discovery') for i in range(1, 5)]
    groups += [('heldout-' + f'{i:02}', 'heldout') for i in range(1, 5)]
    across = within = 0
    for case_id, _ in groups:
        for stack, target in (('encoder', 2), ('decoder', 8)):
            a, b = study.random_guesses(case_id, stack)
            across += a == target
            within += b == target
    assert (across, within) == (1, 3)


@pytest.mark.parametrize('fault', ['wrong_shape', 'nonfinite'])
def test_returned_invalid_logits_are_retained_with_failure(tmp_path, monkeypatch, fault):
    original = study.trace_t5_forward

    def bad_trace(*args, **kwargs):
        output, records = original(*args, **kwargs)
        output.logits = output.logits.clone()
        if fault == 'wrong_shape':
            output.logits = output.logits[..., :-1].contiguous()
        else:
            output.logits[0, 0, 0] = float('nan')
        return output, records

    monkeypatch.setattr(study, 'trace_t5_forward', bad_trace)
    root = tmp_path / fault
    with pytest.raises(ValueError):
        study.execute(tiny_model(), [case()], [0, 3], root, vocab=96, hidden=16)
    events = [json.loads(line) for line in (root / 'attempt.jsonl').read_text().splitlines()]
    observed = next(e for e in events if e['event'] == 'forward_observation')
    assert 'logits' in observed['refs'] and len(observed['refs']) == 13
    assert events[-1]['event'] == 'forward_incomplete' and events[-1]['return_observed'] is True
    assert not (root / 'result.json').exists()


def test_persistent_input_mutation_is_rejected_after_evidence(tmp_path, monkeypatch):
    original = study.trace_t5_forward

    def mutate(*args, **kwargs):
        output, records = original(*args, **kwargs)
        kwargs['input_ids'][0, 0] += 1
        return output, records

    monkeypatch.setattr(study, 'trace_t5_forward', mutate)
    root = tmp_path / 'mutated'
    with pytest.raises(ValueError, match='input IDs or masks changed'):
        study.execute(tiny_model(), [case()], [0, 3], root, vocab=96, hidden=16)
    events = [json.loads(line) for line in (root / 'attempt.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'forward_incomplete' and events[-1]['return_observed'] is True
    assert any(e['event'] == 'forward_observation' and 'logits' in e['refs'] for e in events)
    assert not (root / 'result.json').exists()


def test_postflight_failure_cannot_publish_complete_result(tmp_path):
    root = tmp_path / 'postflight-fails'

    def bad_postflight():
        raise RuntimeError('fabricated asset failure')

    with pytest.raises(RuntimeError, match='fabricated asset failure'):
        study.execute(tiny_model(), [case()], [0, 3], root, vocab=96, hidden=16, postflight=bad_postflight)
    events = [json.loads(line) for line in (root / 'attempt.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'forward_complete' for e in events) == 10
    assert events[-1]['event'] == 'postflight_incomplete'
    assert not (root / 'result.json').exists()


def test_short_tensor_write_is_rejected_and_partial_file_retained(tmp_path, monkeypatch):
    original_open = Path.open

    class Partial:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, value):
            return self.stream.write(value[:len(value) // 2])

        def flush(self):
            self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def short_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return Partial(stream) if path.suffix == '.f32le' and args and args[0] == 'xb' else stream

    monkeypatch.setattr(Path, 'open', short_open)
    evidence = study.Evidence(tmp_path / 'short')
    try:
        with pytest.raises(OSError, match='short evidence write'):
            evidence.put(torch.ones(4, dtype=torch.float32))
    finally:
        evidence.close()
    payloads = list((tmp_path / 'short/tensors').iterdir())
    assert len(payloads) == 1 and payloads[0].stat().st_size == 8


def test_failure_logging_does_not_mask_cancellation(tmp_path, monkeypatch):
    original_event = study.Evidence.event

    def event(evidence, value):
        if value['event'] == 'forward_incomplete':
            raise OSError('fabricated logging failure')
        return original_event(evidence, value)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('fabricated cancellation')

    monkeypatch.setattr(study.Evidence, 'event', event)
    monkeypatch.setattr(study, 'trace_t5_forward', interrupted)
    with pytest.raises(KeyboardInterrupt, match='fabricated cancellation') as caught:
        study.execute(tiny_model(), [case()], [0, 3], tmp_path / 'interrupt', vocab=96, hidden=16)
    assert isinstance(caught.value.__cause__, OSError)


def test_close_failure_does_not_mask_cancellation(tmp_path, monkeypatch):
    original_close = study.Evidence.close

    def close(evidence):
        original_close(evidence)
        raise OSError('fabricated close failure')

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('fabricated cancellation')

    monkeypatch.setattr(study.Evidence, 'close', close)
    monkeypatch.setattr(study, 'trace_t5_forward', interrupted)
    with pytest.raises(KeyboardInterrupt, match='fabricated cancellation') as caught:
        study.execute(tiny_model(), [case()], [0, 3], tmp_path / 'interrupt-close', vocab=96, hidden=16)
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize('boundary', ['forward_log', 'postflight_log', 'close'])
def test_later_cancellation_is_not_masked_by_ordinary_failure(tmp_path, monkeypatch, boundary):
    original_event = study.Evidence.event
    original_close = study.Evidence.close

    def event(evidence, value):
        if (boundary == 'forward_log' and value['event'] == 'forward_incomplete'
                or boundary == 'postflight_log' and value['event'] == 'postflight_incomplete'):
            raise KeyboardInterrupt('later cancellation')
        return original_event(evidence, value)

    def close(evidence):
        original_close(evidence)
        if boundary == 'close':
            raise KeyboardInterrupt('later cancellation')

    def bad_forward(*args, **kwargs):
        raise ValueError('first ordinary failure')

    def bad_postflight():
        raise ValueError('first ordinary failure')

    monkeypatch.setattr(study.Evidence, 'event', event)
    monkeypatch.setattr(study.Evidence, 'close', close)
    if boundary != 'postflight_log':
        monkeypatch.setattr(study, 'trace_t5_forward', bad_forward)
    with pytest.raises(KeyboardInterrupt, match='later cancellation') as caught:
        study.execute(tiny_model(), [case()], [0, 3], tmp_path / boundary,
                      vocab=96, hidden=16, postflight=bad_postflight if boundary == 'postflight_log' else None)
    assert isinstance(caught.value.__cause__, ValueError)
    assert not (tmp_path / boundary / 'result.json').exists()
