"""Fabricated-model schedule, landing and arithmetic tests; no pretrained model."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_patch', REPO / 'tools/public_two_attractor_patch_v1.py')
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
CASES = json.loads((REPO / 'experiments/distilgpt2_two_attractor_patch_cases_v1.json').read_bytes())['cases']


class ToyAttention(nn.Module):
    def forward(self, value):
        return (value * 0.1 + torch.cumsum(value, dim=1) * 0.02, None)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ToyAttention()

    def forward(self, value):
        return value + self.attn(value)[0]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(ToyBlock() for _ in range(4))

    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False and bool(torch.all(attention_mask == 1))
        value = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 768).contiguous() / 100000
        for block in self.transformer.h:
            value = block(value)
        logits = torch.zeros((1, 8, 50257), dtype=torch.float32)
        logits[:, :, 318] = value[:, :, 0] * 3
        logits[:, :, 389] = value[:, :, 1] * 2
        logits[:, :, 373] = value[:, :, 2] * 4
        logits[:, :, 547] = value[:, :, 3]
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_hand_calculated_contrast_and_ties():
    def score(value, number):
        a, b = (value, 0.0) if number == 'singular' else (0.0, value)
        return {'values': {'is': a, 'are': b, 'was': a, 'were': b}}
    for number in ('singular', 'plural'):
        clean = {k: score(v, number) for k, v in { (0,0):3, (1,0):2, (0,1):1, (1,1):-1 }.items()}
        actions = {k: score(v, number) for k, v in {'near':2.5, 'beside':2, 'both':1}.items()}
        answer = patch._contrast(clean, actions, number, 'present')
        assert answer['clean_interaction'] == -1
        assert answer['patch_interaction'] == -0.5
        assert answer['patch_minus_clean'] == 0.5
        assert answer['patch_shifts'] == {'near': -0.5, 'beside': -1, 'both': -2}
        actions['both'] = score(0, number)
        answer = patch._contrast(clean, actions, number, 'present')
        assert answer['patched_accuracy']['both'] == {'correct': False, 'tie': True}


def test_real_plan_toy_schedule_and_saved_landing(tmp_path):
    torch.set_num_threads(1)
    model = ToyModel().eval()
    output = tmp_path / 'run'
    result = patch.execute(model, CASES, output, {'fake_model': True})
    records = [json.loads(line) for line in (output / 'attempt.jsonl').read_text().splitlines()]
    starts = [r for r in records if r['event'] == 'forward_start']
    finished = [r for r in records if r['event'] == 'forward_result']
    assert len(starts) == len(finished) == result['forwards'] == 112
    assert sum(r['event'] == 'model_returned' for r in records) == 112
    assert sum(r['kind'] == 'clean' for r in finished) == 32
    assert len(result['units']) == 16 and result['raw_tensor_bytes'] <= patch.RAW_LIMIT
    for record in finished:
        if record['kind'] == 'clean':
            refs = record['vectors'].values()
        else:
            assert all(record['checks'].values())
            refs = record['tensors'].values()
        for ref in refs:
            payload = (output / ref['path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == ref['sha256']
            assert len(payload) == ref['bytes']
    assert records[-1]['event'] == 'complete'
    assert records[-1]['result_sha256'] == hashlib.sha256((output / 'result.json').read_bytes()).hexdigest()
    assert not model.transformer.h[1].attn._forward_hooks
    assert not model.transformer.h[3].attn._forward_hooks


def test_changed_nondonor_token_rejected():
    altered = copy.deepcopy(CASES)
    altered[0]['input_ids'][2] += 1
    with pytest.raises(ValueError, match='alignment'):
        patch._units(altered)


def test_full_logits_must_be_finite_and_match_vocab():
    logits = torch.zeros((1, 8, 50257), dtype=torch.float32)
    logits[0, 0, 1] = float('nan')
    with pytest.raises(ValueError, match='output'):
        patch._scores(SimpleNamespace(logits=logits, past_key_values=None))
    with pytest.raises(ValueError, match='output'):
        patch._scores(SimpleNamespace(logits=torch.zeros((1, 8, 548)), past_key_values=None))


def test_invalid_return_retains_available_evidence(tmp_path):
    class InvalidToy(ToyModel):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output.logits[0, 0, 1] = float('nan')
            return output

    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='output'):
        patch.execute(InvalidToy().eval(), CASES, output, {'fake_model': True})
    records = [json.loads(line) for line in (output / 'attempt.jsonl').read_text().splitlines()]
    assert [r['event'] for r in records] == ['begin', 'forward_start', 'model_returned', 'forward_incomplete']
    incomplete = records[-1]
    assert incomplete['model_returned'] is True
    assert len(incomplete['selected_float32_le_hex']) == 32
    assert set(incomplete['captures']) == {'site_0', 'site_1'}
    assert incomplete['unavailable'] == []
    for ref in incomplete['captures'].values():
        assert hashlib.sha256((output / ref['path']).read_bytes()).hexdigest() == ref['sha256']
    assert not (output / 'result.json').exists()
