"""Small random-model checks; no checkpoint or prospective study inputs."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from scripts import bert_taste_template_direction_v1 as study


def fabricated_rows():
    rows = []
    for template in ('after', 'with'):
        for ingredient, word in (('sugar', 3), ('lemon', 4)):
            ids = [0, 1, 2, 9, 10, 5 if template == 'after' else 7,
                   6 if template == 'after' else 8, word, 11, 12]
            rows.append({'id': f'toy-{template}-{ingredient}', 'stem': 'toy', 'template': template,
                         'ingredient': ingredient, 'input_ids': ids, 'attention_mask': [1] * 10})
    return rows


def model_and_tokenizer():
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(4021)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32, n_layers=6, dim=16,
                              hidden_dim=32, n_heads=2, dropout=0., attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = DistilBertForMaskedLM(config).eval()
    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(v) for v in values]
    return model, Tokenizer()


def test_actual_hooks_schedule_controls_and_analysis(monkeypatch):
    model, tokenizer = model_and_tokenizer()
    monkeypatch.setattr(study.previous, 'eligible', lambda *_args: False)
    events = []
    result = study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                             tokenizer=tokenizer, emit=events.append)
    assert result['completed_forwards'] == 56
    assert sum(e['event'] == 'forward_start' for e in events) == 56
    assert sum(e['event'] == 'forward_complete' for e in events) == 56
    assert sum(e['event'] == 'dose_plan' for e in events) == 1
    assert len(result['effects']) == 40 and len(result['comparisons']) == 4
    assert len(result['stem_summaries']) == 1
    assert not any(result['eligibility_descriptive'].values())
    assert result['means']['after'] is not None and result['means']['with'] is not None
    assert all(e['effect'] == 0 for e in result['effects']
               if e['mode'] in ('ingredient_self', 'the_self', 'final_block_donor'))
    for event in events:
        if event['event'] == 'forward_complete':
            row = next(r for r in fabricated_rows() if r['id'] == event['row_id'])
            assert event['model_entry'] == {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}
            if event['dose'] is not None:
                assert event['actual_pre_post_norm'] == event['dose']['delivered_norm']
                assert event['actual_donor_cosine'] == event['dose']['cosine_with_donor_direction']
            if event['mode'] in ('ingredient_self', 'the_self'):
                assert event['before_float32_le_base64'] == event['after_float32_le_base64']
    assert 'forward' not in model.__dict__ and not study.previous.previous.has_hooks(model)
    effect = {(e['template'], e['direction'], e['mode']): e['effect'] for e in result['effects']}
    comparisons = {(c['template'], c['direction']): c for c in result['comparisons']}
    for template in ('after', 'with'):
        for direction, sign in (('sugar_to_lemon', 1), ('lemon_to_sugar', -1)):
            item = comparisons[template, direction]
            expected = sign * (effect[template, direction, 'ingredient_toward']
                               - effect[template, direction, 'the_toward'])
            assert math.isclose(item['signed_matched_position_contrast'], expected, abs_tol=1e-15)
            odd = (effect[template, direction, 'ingredient_toward']
                   - effect[template, direction, 'ingredient_away']) / 2
            assert math.isclose(item['odd']['ingredient'], odd, abs_tol=1e-15)
            orth = (effect[template, direction, 'orth_plus'] - effect[template, direction, 'orth_minus']) / 2
            assert math.isclose(item['donor_odd_minus_abs_orth'], sign * odd - abs(orth), abs_tol=1e-15)
    summary = result['stem_summaries'][0]
    delta_after = (comparisons['after', 'sugar_to_lemon']['signed_matched_position_contrast']
                   - comparisons['after', 'lemon_to_sugar']['signed_matched_position_contrast'])
    delta_with = (comparisons['with', 'sugar_to_lemon']['signed_matched_position_contrast']
                  - comparisons['with', 'lemon_to_sugar']['signed_matched_position_contrast'])
    assert math.isclose(summary['direction_main'], (delta_after + delta_with) / 2, abs_tol=1e-15)
    assert math.isclose(summary['template_direction_interaction'], delta_after - delta_with, abs_tol=1e-15)
    assert math.isclose(summary['direction_specificity_template_difference'],
                        summary['direction_specificity_after'] - summary['direction_specificity_with'], abs_tol=1e-15)


def test_invalid_dose_stops_after_clean_captures(monkeypatch):
    model, tokenizer = model_and_tokenizer()
    def invalid(*_args, **_kwargs):
        raise ValueError('fabricated unavailable dose')
    monkeypatch.setattr(study.geometry, 'plan', invalid)
    events = []
    with pytest.raises(ValueError, match='fabricated unavailable dose'):
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                        tokenizer=tokenizer, emit=events.append)
    assert sum(e['event'] == 'forward_start' for e in events) == 8
    assert sum(e['event'] == 'forward_complete' for e in events) == 8
    assert events[-1]['event'] == 'dose_unavailable'
    assert 'forward' not in model.__dict__ and not study.previous.previous.has_hooks(model)


def test_post_return_failure_retains_raw_observation(monkeypatch):
    model, tokenizer = model_and_tokenizer()
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
    assert [e['event'] for e in events] == ['forward_start', 'forward_returned_invalid']
    invalid = events[-1]
    assert invalid['exception_type'] == 'RuntimeError'
    assert invalid['model_entry'] == {'input_ids': fabricated_rows()[0]['input_ids'],
                                      'attention_mask': [1] * 10}
    assert len(study.base64.b64decode(invalid['mask_logits_float32_le_base64'])) == 64 * 4
    assert len(study.base64.b64decode(invalid['before_float32_le_base64'])) == 16 * 4
    assert 'forward' not in model.__dict__ and not study.previous.previous.has_hooks(model)


@pytest.mark.parametrize('primary,recording,expected', [
    (KeyboardInterrupt, OSError, KeyboardInterrupt),
    (RuntimeError, KeyboardInterrupt, KeyboardInterrupt),
])
def test_returned_failure_preserves_cancellation(monkeypatch, primary, recording, expected):
    model, tokenizer = model_and_tokenizer()
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
            raise recording('failure-record fault')
    monkeypatch.setattr(study.previous.previous, 'denormal_probe', probe)
    with pytest.raises(expected) as caught:
        study.run_cases(model, fabricated_rows(), sweet_id=1, sour_id=2,
                        tokenizer=tokenizer, emit=emit)
    assert isinstance(caught.value.__cause__, recording if primary is expected else primary)


def test_isolated_child_blocks_checkout_import_shadows(tmp_path):
    root = tmp_path / 'checkout'
    for name in study.SOURCE_PATHS:
        source = study.ROOT / name
        if source.is_file():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (root / 'numbers.py').write_text("raise RuntimeError('untracked source ran')\n", encoding='utf-8')
    prefix = tmp_path / 'empty-pyc'
    prefix.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_CBWR='AUTO')
    env.pop('ATEN_CPU_CAPABILITY', None)
    args = [sys.executable, '-I', '-B', '-X', 'pycache_prefix=' + str(prefix),
            str(root / 'scripts/bert_taste_template_direction_v1.py'), '--help']
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert not list(prefix.rglob('*'))
    (root / 'scripts/__init__.py').write_text("raise RuntimeError('untracked initializer ran')\n", encoding='utf-8')
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, timeout=60)
    assert result.returncode != 0 and b'untracked initializer ran' not in result.stderr


def test_isolated_child_rejects_prepopulated_bytecode_prefix(tmp_path):
    root = tmp_path / 'checkout'
    for name in study.SOURCE_PATHS:
        source = study.ROOT / name
        if source.is_file():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    prefix = tmp_path / 'poisoned-pyc'
    prefix.mkdir()
    marker = tmp_path / 'cached-code-executed'
    geometry_source = root / 'scripts/bert_taste_template_geometry_v1.py'
    prepare = '''
import importlib._bootstrap_external as be
import importlib.util
from pathlib import Path
import sys
sys.pycache_prefix = sys.argv[1]
source, marker = Path(sys.argv[2]), Path(sys.argv[3])
stat = source.stat()
payload = "from pathlib import Path\\nPath(" + repr(str(marker)) + ").write_text('ran')\\n"
cached = Path(importlib.util.cache_from_source(str(source)))
cached.parent.mkdir(parents=True, exist_ok=True)
cached.write_bytes(be._code_to_timestamp_pyc(compile(payload, str(source), 'exec'),
                                             int(stat.st_mtime), stat.st_size))
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', prepare, str(prefix),
                    str(geometry_source), str(marker)], check=True, timeout=30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_CBWR='AUTO')
    env.pop('ATEN_CPU_CAPABILITY', None)
    args = [sys.executable, '-I', '-B', '-X', 'pycache_prefix=' + str(prefix)]
    positive = subprocess.run(args + ['-c', 'import sys; sys.path.insert(0, sys.argv[1]); '
                                      'import scripts.bert_taste_template_geometry_v1', str(root)],
                              cwd=root, env=env, capture_output=True, timeout=60)
    assert positive.returncode == 0 and marker.is_file()
    marker.unlink()
    guarded = subprocess.run(args + [str(root / 'scripts/bert_taste_template_direction_v1.py'), '--help'],
                             cwd=root, env=env, capture_output=True, timeout=60)
    assert guarded.returncode != 0
    assert b'fresh empty bytecode prefix' in guarded.stderr
    assert not marker.exists()
