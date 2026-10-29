import importlib
import math

import pytest


def test_case_grid_uses_reciprocal_single_token_year_and_label_donors():
    study = importlib.import_module('looking_glass.year_label_control')
    cases = study.make_inventory()['cases']
    assert len(cases) == 48
    assert len({tuple(case['input_ids']) for case in cases}) == 48
    first = cases[0]
    assert first['text'] == 'The event tagged alpha took place after the year 1709 in the year 17'
    assert first['input_ids'] == [464, 1785, 30509, 17130, 1718, 1295, 706, 262, 614, 1596, 2931, 287, 262, 614, 1596]
    by_id = {case['case_id']: case for case in cases}
    for case in cases:
        year = by_id[case['year_donor_case_id']]
        label = by_id[case['label_donor_case_id']]
        assert year['year_donor_case_id'] == case['case_id']
        assert label['label_donor_case_id'] == case['case_id']
        assert [i for i, (a, b) in enumerate(zip(case['input_ids'], year['input_ids'])) if a != b] == [10]
        assert [i for i, (a, b) in enumerate(zip(case['input_ids'], label['input_ids'])) if a != b] == [3]
        assert case['middle_ids'] == year['middle_ids'] == label['middle_ids']
        assert case['above_ids'] == year['above_ids'] == label['above_ids']


def test_tampered_donor_or_candidate_band_is_rejected():
    study = importlib.import_module('looking_glass.year_label_control')
    data = study.make_inventory()
    data['cases'][0]['label_donor_case_id'] = data['cases'][0]['case_id']
    with pytest.raises(ValueError):
        study.validate_inventory(data)
    data = study.make_inventory()
    data['cases'][0]['above_ids'][0] = data['cases'][0]['middle_ids'][0]
    with pytest.raises(ValueError):
        study.validate_inventory(data)


def test_individual_absolute_label_effects_cannot_cancel_into_a_pass():
    study = importlib.import_module('looking_glass.year_label_control')
    rows = study.make_inventory()['cases']
    clean = {row['case_id']: {'band_score': 0.0, 'local_score': 0.0} for row in rows}
    actions = {}
    for row in rows:
        year = -0.6 if row['year_role'] == 'low' else 0.6
        actions[row['case_id']] = {'self': 0.0, 'year_final': year, 'label_final': 0.0,
                                   'year_at_zero': 0.0, 'rotated_year_final': 0.0}
    assert study.summarize_interventions(rows, clean, actions)['pattern_count'] == 4
    for row in rows:
        if row['family_index'] in (0, 1):
            actions[row['case_id']]['label_final'] = 0.55 if row['year_role'] == 'low' else -0.55
    result = study.summarize_interventions(rows, clean, actions)
    assert result['pattern_count'] == 2 and result['pattern_supported'] is False
    assert [family['pass'] for family in result['families']] == [False, False, True, True]
    assert all(target['margins']['label_final'] == pytest.approx(0.05)
               for family in result['families'][:2] for target in family['targets'])


def test_full_vocabulary_scores_use_fixed_bands_and_normalized_mass():
    study = importlib.import_module('looking_glass.year_label_control')
    row = study.make_inventory()['cases'][0]
    logits = [0.0] * 50257
    for index in row['middle_ids']:
        logits[index] = 1.0
    score = study.score_logits(logits, row)
    total = 50254 + 3 * math.e
    assert score['band_score'] == pytest.approx(1.0)
    assert score['local_score'] == pytest.approx(math.log((2 + math.e) / 3))
    assert score['aligned_01_99_full_mass'] == pytest.approx((96 + 3 * math.e) / total)
    assert score['zero_00_next_token_mass'] == pytest.approx(1 / total)
    assert score['other_vocabulary_mass'] == pytest.approx(50157 / total)
    assert score['top_token_id'] == min(row['middle_ids'])


def test_wrong_prompt_tokenization_is_rejected_before_model_work():
    study = importlib.import_module('looking_glass.year_label_control')
    class WrongTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [study.prior.digit(0)] if text == '00' else []
        def decode(self, ids, skip_special_tokens=False):
            return '00'
    with pytest.raises(ValueError, match='prompt'):
        study.validate_tokenizer(WrongTokenizer(), study.make_inventory()['cases'])


def test_zero_token_must_decode_to_literal_zero_zero():
    study = importlib.import_module('looking_glass.year_label_control')
    rows = study.make_inventory()['cases']
    prompts = {row['text']: row['input_ids'] for row in rows}

    class WrongZeroDecoder:
        def encode(self, text, add_special_tokens=False):
            if text in prompts:
                return prompts[text]
            if len(text) == 2 and text.isdigit():
                return [study.prior.digit(int(text))]
            if text[:-2] in prompts and text[-2:].isdigit():
                if text[-2:] == '00':
                    return prompts[text[:-2]][:-1] + [999]
                return prompts[text[:-2]] + [study.prior.digit(int(text[-2:]))]
            return []

        def decode(self, ids, skip_special_tokens=False):
            return 'not zero zero'

    with pytest.raises(ValueError, match='00'):
        study.validate_tokenizer(WrongZeroDecoder(), rows)


def test_clean_summary_retains_each_anchor_and_its_low_high_contrast():
    study = importlib.import_module('looking_glass.year_label_control')
    rows = study.make_inventory()['cases']
    metrics = {}
    for row in rows:
        value = float(100 * row['family_index'] + 10 * row['condition_index']
                      + 2 * row['label_index'] + row['anchor_index'])
        metrics[row['case_id']] = {'band_score': value, 'local_score': -value}
    summary = study.summarize_clean(rows, metrics)
    first = summary['families'][0]['conditions']['after']['alpha']
    assert first['low']['band_score'] == 0.0
    assert first['high']['band_score'] == 1.0
    assert first['high']['local_score'] == -1.0
    assert first['low_minus_high'] == -1.0
    assert summary['families'][3]['conditions']['near']['blue']['high']['band_score'] == 323.0
    assert summary['mean_band_score'] == 161.5
    assert summary['mean_low_minus_high'] == -1.0


def test_control_threshold_is_strict_for_each_target():
    study = importlib.import_module('looking_glass.year_label_control')
    rows = study.make_inventory()['cases']
    clean = {row['case_id']: {'band_score': 0.0, 'local_score': 0.0} for row in rows}
    actions = {row['case_id']: {'self': 0.0,
               'year_final': -0.6 if row['year_role'] == 'low' else 0.6,
               'label_final': 0.0, 'year_at_zero': 0.0, 'rotated_year_final': 0.0} for row in rows}
    actions[rows[0]['case_id']]['label_final'] = 0.5
    result = study.summarize_interventions(rows, clean, actions)
    assert result['families'][0]['targets'][0]['checks']['label_final'] is False
    assert result['families'][0]['targets'][0]['year_role'] == 'low'
    assert result['families'][0]['targets'][0]['label'] == 'alpha'
    assert result['mean_year_donorward_by_condition']['after'] == pytest.approx(0.6)
    assert result['mean_effect_by_condition']['after']['year_final'] == 0.0
    assert result['pattern_count'] == 3 and result['pattern_supported'] is True
