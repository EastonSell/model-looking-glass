import copy
import math
from pathlib import Path

import pytest

from looking_glass import explicit_year_direction as study


def test_fixed_inventory_has_single_token_word_and_year_swaps():
    import json
    path = Path(__file__).resolve().parents[1] / 'experiments/distilgpt2_explicit_year_direction_v1.json'
    data = json.loads(path.read_bytes())
    assert data == study.make_inventory()
    rows = study.validate_inventory(data)
    assert len(rows) == 48 and len({tuple(row['input_ids']) for row in rows}) == 48
    for offset in range(0, len(rows), 6):
        group = rows[offset:offset + 6]
        for i in range(0, 6, 2):
            low, high = group[i:i + 2]
            assert low['donor_case_id'] == high['case_id']
            assert high['donor_case_id'] == low['case_id']
            assert low['middle_ids'] == high['middle_ids']
            assert low['above_ids'] == high['above_ids']
            assert [k for k, (a, b) in enumerate(zip(low['input_ids'], high['input_ids'])) if a != b] == [8]
        for role in ('low', 'high'):
            selected = [r for r in group if r['year_role'] == role]
            assert [r['condition'] for r in selected] == ['after', 'before', 'near']
            assert all(r['local_below_ids'] == selected[0]['local_below_ids'] for r in selected)
            assert all(r['local_above_ids'] == selected[0]['local_above_ids'] for r in selected)
            assert all([k for k, (a, b) in enumerate(zip(selected[0]['input_ids'], r['input_ids'])) if a != b]
                       == [4] for r in selected[1:])
    mutated = copy.deepcopy(data)
    mutated['cases'][0]['input_ids'][8] = False
    with pytest.raises(ValueError):
        study.validate_inventory(mutated)


def test_real_offline_tokenizer_validates_all_contextual_continuations():
    import hashlib
    from tokenizers import Tokenizer
    from transformers import GPT2TokenizerFast
    cache = (Path(__file__).resolve().parents[3] / 'hf-cache/hub/models--distilbert--distilgpt2/snapshots'
             / study.MODEL_REVISION / 'tokenizer.json')
    if not cache.exists():
        pytest.skip('pinned offline tokenizer unavailable')
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == study.TOKENIZER_SHA256
    rows = study.make_inventory()['cases']
    study.validate_tokenizer(Tokenizer.from_file(str(cache)), rows)
    study.validate_tokenizer(GPT2TokenizerFast(tokenizer_file=str(cache)), rows)


def test_score_uses_stable_band_order_and_full_vocabulary_mass():
    row = study.make_inventory()['cases'][0]
    logits = [0.0] * 50257
    for token in row['local_above_ids']:
        logits[token] = 2.0
    result = study.score_logits(logits, row)
    assert result['local_score'] == pytest.approx(2.0)
    assert sum(result[k] for k in ('aligned_01_99_full_mass', 'zero_00_next_token_mass',
                                    'other_vocabulary_mass')) == pytest.approx(1.0)
    assert result['top_token_id'] == min(row['local_above_ids'])
    logits[0] = math.inf
    with pytest.raises(ValueError):
        study.score_logits(logits, row)


def _metrics(rows, clean_pass):
    out = {}
    for row in rows:
        c, role = row['condition'], row['year_role']
        m = ({'after': 1.0, 'before': -1.0, 'near': 0.0}[c]
             if role == 'low' and clean_pass else 0.0)
        w = {'after': 1.0, 'before': -1.0, 'near': 0.0}[c] if clean_pass else 0.0
        out[row['case_id']] = {'band_score': m, 'local_score': w}
    return out


def test_strict_clean_gate_uses_both_anchors_and_both_metrics():
    rows = study.make_inventory()['cases']
    metrics = _metrics(rows, True)
    summary = study.summarize_clean(rows, metrics)
    assert summary['comparator_gate_passed'] and summary['comparator_gate_count'] == 8
    assert summary['pairs'][0]['local_wording_differences']['after_minus_before']['high'] == 2.0
    assert summary['pairs'][0]['fixed_band_wording_differences']['after_minus_before'] == 2.0
    for row in rows:
        if row['pair_noun_index'] in (0, 1, 2) and row['condition'] == 'before' and row['year_role'] == 'high':
            metrics[row['case_id']]['local_score'] = -0.5
    summary = study.summarize_clean(rows, metrics)
    assert not summary['comparator_gate_passed'] and summary['comparator_gate_count'] == 5
    assert len(summary['pairs']) == 8


def test_generic_pattern_is_scored_even_when_comparator_gate_fails():
    rows = study.make_inventory()['cases']
    metrics = _metrics(rows, False)
    actions = {}
    for row in rows:
        direction = -1.0 if row['year_role'] == 'low' else 1.0
        actions[row['case_id']] = {'self': 0.0, 'year_final': direction,
                                   'year_at_zero': 0.0, 'rotated_final': 0.0}
    result = study.summarize_interventions(rows, metrics, actions)
    assert result['generic_pattern_count'] == 8 and result['generic_pattern_supported']
    assert result['comparator_direction_supported'] is None
    assert result['comparator_control_supported'] is None


def test_eligible_opposite_direction_pattern_and_controls():
    rows = study.make_inventory()['cases']
    metrics = _metrics(rows, True)
    actions = {}
    for row in rows:
        value = metrics[row['case_id']]['band_score']
        direction = -1.0 if row['year_role'] == 'low' else 1.0
        effect = direction * (-1.0 if row['condition'] == 'before' else 1.0)
        actions[row['case_id']] = {'self': value, 'year_final': value + effect,
                                   'year_at_zero': value, 'rotated_final': value}
    result = study.summarize_interventions(rows, metrics, actions)
    assert result['comparator_direction_supported'] and result['comparator_control_supported']
    assert result['mean_after_minus_before_by_action']['year_final'] == 2.0
    assert result['generic_pattern_count'] == 0 and not result['generic_pattern_supported']
    actions[rows[0]['case_id']]['self'] += 0.25
    with pytest.raises(ValueError):
        study.summarize_interventions(rows, metrics, actions)
