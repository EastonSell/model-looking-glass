"""Fabricated end-to-end scheduling and failure checks; no checkpoint."""
import base64
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from scripts import bert_taste_propagation_v1 as study


def fabricated_rows():
    rows = []
    for template in ('after', 'with'):
        for ingredient, word in (('sugar', 3), ('lemon', 4)):
            ids = [0, 1, 2, 9, 10, 5 if template == 'after' else 7,
                   6 if template == 'after' else 8, word, 11, 12]
            rows.append({'id': f'toy-{template}-{ingredient}', 'stem': 'toy', 'template': template,
                         'ingredient': ingredient, 'input_ids': ids, 'attention_mask': [1] * 10})
    return rows


def tiny_model():
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    torch.random.default_generator.manual_seed(20261102)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32, n_layers=6, dim=16,
                              hidden_dim=32, n_heads=2, dropout=0., attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = DistilBertForMaskedLM(config).eval()
    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(value) for value in values]
    return model, Tokenizer()


def test_one_stem_full_schedule_controls_and_saved_bits():
    model, tokenizer = tiny_model()
    events = []
    result = study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                             tokenizer=tokenizer, emit=events.append)
    assert result['completed_forwards'] == 160
    assert sum(e['event'] == 'forward_start' for e in events) == 160
    complete = [e for e in events if e['event'] == 'forward_complete']
    assert len(complete) == 160 and sum(e['event'] == 'dose_plan' for e in events) == 1
    assert len(result['comparisons']) == 16 and len(result['propagation']) == 16
    assert len(result['effects']) == 144
    assert sum(e['mask_logits_float32_le_base64'] is not None for e in complete) == 132
    for event in complete:
        row = next(item for item in fabricated_rows() if item['id'] == event['row_id'])
        assert event['model_entry'] == {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}
        if event['mask_logits_float32_le_base64'] is not None:
            assert len(base64.b64decode(event['mask_logits_float32_le_base64'])) == 64 * 4
        if event['layer'] == 0:
            assert set(event['later_float32_le_base64']) == set(study.observer.LATE_POSITIONS)
        if event['mode'] == 'self':
            assert event['before_float32_le_base64'] == event['after_float32_le_base64']
    for cell in result['comparisons']:
        axis = cell['axes']
        expected = math.fsum((axis['lex']['signed_odd'],
                              -math.fsum((abs(axis['orth1']['odd']), abs(axis['orth2']['odd']))) / 2))
        assert cell['taste_specificity'] == expected
        assert all(0 <= axis[name]['tv_plus'] <= 1 and 0 <= axis[name]['tv_minus'] <= 1
                   for name in ('lex', 'context', 'orth1', 'orth2'))
    json.dumps(result, allow_nan=False)
    assert 'forward' not in model.__dict__ and not study.previous.previous.has_hooks(model)


def test_unavailable_dose_preserves_four_clean_captures(monkeypatch):
    model, tokenizer = tiny_model()
    def invalid(*_args, **_kwargs):
        raise ValueError('fabricated unavailable dose')
    monkeypatch.setattr(study.geometry, 'plan', invalid)
    events = []
    with pytest.raises(ValueError, match='fabricated unavailable dose'):
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                        tokenizer=tokenizer, emit=events.append)
    assert sum(e['event'] == 'forward_start' for e in events) == 4
    assert sum(e['event'] == 'forward_complete' for e in events) == 4
    assert events[-1]['event'] == 'dose_unavailable'


def test_post_return_failure_retains_raw_observation(monkeypatch):
    model, tokenizer = tiny_model()
    real_probe = study.previous.previous.denormal_probe
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('fabricated post-return fault')
        return real_probe()
    monkeypatch.setattr(study.previous.previous, 'denormal_probe', probe)
    events = []
    with pytest.raises(RuntimeError, match='fabricated post-return fault'):
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                        tokenizer=tokenizer, emit=events.append)
    assert [event['event'] for event in events] == ['forward_start', 'forward_returned_invalid']
    assert len(base64.b64decode(events[-1]['mask_logits_float32_le_base64'])) == 64 * 4
    assert set(events[-1]['later_float32_le_base64']) == set(study.observer.LATE_POSITIONS)
    assert 'forward' not in model.__dict__ and not study.previous.previous.has_hooks(model)


@pytest.mark.parametrize('mode,field', [
    ('self', 'logits'), ('self', 'later'),
    ('fresh_clean', 'logits'), ('fresh_clean', 'before'), ('fresh_clean', 'later'),
    ('final_capture', 'logits'), ('final_donor', 'logits'),
])
def test_failed_returned_control_records_raw_bits_without_completion(monkeypatch, mode, field):
    model, tokenizer = tiny_model()
    real_observe = study.observer.observe
    events = []
    def observe(*args, **kwargs):
        observed = real_observe(*args, **kwargs)
        if events[-1]['event'] == 'forward_start' and events[-1]['mode'] == mode:
            if field == 'logits':
                changed = observed.mask_logits.clone()
                changed[0] += 1
                return replace(observed, mask_logits=changed)
            if field == 'before':
                changed = observed.before.clone()
                changed[0] += 1
                return replace(observed, before=changed)
            changed = dict(observed.later)
            changed['late_the'] = changed['late_the'].clone()
            changed['late_the'][0] += 1
            return replace(observed, later=changed)
        return observed
    monkeypatch.setattr(study.observer, 'observe', observe)
    with pytest.raises(RuntimeError, match='changed|differs'):
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                        tokenizer=tokenizer, emit=events.append)
    assert [event['event'] for event in events[-2:]] == ['forward_start', 'forward_returned_invalid']
    assert events[-1]['mode'] == mode
    assert len(base64.b64decode(events[-1]['mask_logits_float32_le_base64'])) == 64 * 4
    assert not study.previous.previous.has_hooks(model) and 'forward' not in model.__dict__


def test_binary64_probe_rejects_flushed_signed_subnormals():
    assert torch.set_flush_denormal(False)
    assert study.metrics.binary64_probe() == 'binary64-subnormals-preserved'
    try:
        assert torch.set_flush_denormal(True)
        with pytest.raises(RuntimeError, match='binary64 subnormal'):
            study.metrics.binary64_probe()
    finally:
        torch.set_flush_denormal(False)
    assert study.metrics.binary64_probe() == 'binary64-subnormals-preserved'


@pytest.mark.parametrize('primary,recording,expected', [
    (KeyboardInterrupt, OSError, KeyboardInterrupt),
    (RuntimeError, KeyboardInterrupt, KeyboardInterrupt),
])
def test_returned_failure_preserves_cancellation(monkeypatch, primary, recording, expected):
    model, tokenizer = tiny_model()
    real_probe = study.previous.previous.denormal_probe
    calls = 0
    def probe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise primary('post-return fault')
        return real_probe()
    def emit(event):
        if event['event'] == 'forward_returned_invalid':
            raise recording('recording fault')
    monkeypatch.setattr(study.previous.previous, 'denormal_probe', probe)
    with pytest.raises(expected) as caught:
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2, tokenizer=tokenizer, emit=emit)
    assert isinstance(caught.value.__cause__, recording if primary is expected else primary)


def test_partial_start_record_is_incomplete(tmp_path, monkeypatch):
    output = tmp_path / 'attempt'
    monkeypatch.setattr(study, 'source_identity', lambda: {'commit': 'fabricated'})
    monkeypatch.setattr(sys, 'argv', ['study', '--cache-dir', str(tmp_path), '--output', str(output)])
    real_write = study.previous.previous._write_json
    def write(path, value):
        if path.name == 'started.json':
            path.write_bytes(b'{"partial":')
            raise OSError('fabricated start fault')
        return real_write(path, value)
    monkeypatch.setattr(study.previous.previous, '_write_json', write)
    with pytest.raises(OSError, match='fabricated start fault'):
        study.main()
    record = json.loads((output / 'incomplete.json').read_text(encoding='utf-8'))
    assert record['status'] == 'incomplete' and record['phase'] == 'started_record'
    assert record['counts'] == {'started': 0, 'completed': 0}


def test_isolated_entry_rejects_nonempty_prefix_and_checkout_shadow(tmp_path):
    root = tmp_path / 'checkout'
    for name in study.SOURCE_PATHS:
        source = study.ROOT / name
        if source.is_file():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (root / 'numbers.py').write_text("raise RuntimeError('untracked source ran')\n", encoding='utf-8')
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_CBWR='AUTO')
    env.pop('ATEN_CPU_CAPABILITY', None)
    args = [sys.executable, '-I', '-B', '-X', 'pycache_prefix=' + str(prefix),
            str(root / 'scripts/bert_taste_propagation_v1.py'), '--help']
    fresh = subprocess.run(args, cwd=root, env=env, capture_output=True, timeout=60)
    assert fresh.returncode == 0, fresh.stderr.decode(errors='replace')
    assert not list(prefix.iterdir())
    (prefix / 'sentinel').write_bytes(b'preexisting')
    stale = subprocess.run(args, cwd=root, env=env, capture_output=True, timeout=60)
    assert stale.returncode != 0 and b'fresh empty bytecode prefix' in stale.stderr
    assert b'untracked source ran' not in stale.stderr
