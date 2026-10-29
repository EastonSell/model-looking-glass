"""Synthetic saved-logit selection tests; no checkpoint or model is used."""
import copy
import math
import struct

import pytest

from looking_glass import distilbert_color_edit_records_v1 as records
from looking_glass import distilbert_color_preference_screen_v2 as screen


def fixture():
    cases, events = [], []
    for obj in ('banana', 'strawberry', 'lemon'):
        for role in ('discovery', 'calibration-01', 'calibration-02'):
            index = len(cases)
            row = {'id': obj + '-' + role, 'object': obj, 'role': role,
                   'input_ids': [101, index, 103, 102], 'attention_mask': [1, 1, 1, 1],
                   'mask_index': 2, 'factual': 'f', 'counterfactual': 'p', 'wrong_target': 'b',
                   'word_ids': {'f': 1, 'p': 0, 'b': 2}}
            cases.append(row)
            binding = {'action': index, 'phase': 'screen', 'object': obj, 'case_id': row['id'],
                       'condition': 'original', 'input_ids': row['input_ids'],
                       'attention_mask': row['attention_mask'], 'mask_index': row['mask_index']}
            # The unrelated third token is always the full-vocabulary top.
            margin = 0.09999999 if obj == 'banana' else 0.10000001
            raw = struct.pack('<3f', 0.0, margin, 8.0)
            events.extend([{'seq': 2 * index, 'kind': 'start', **binding},
                           {'seq': 2 * index + 1, 'kind': 'complete', **binding,
                            'parameter': None, 'parameter_sha256': None, 'realized_norm': 0.0,
                            'logits': records.encode_raw(raw)}])
    return {'candidate_order': ['banana', 'strawberry', 'lemon'], 'cases': cases}, events


def test_strict_pairwise_selection_does_not_require_the_factual_full_top():
    manifest, events = fixture()
    report = screen.rebuild_screen(manifest, events, vocabulary=3)
    assert report['selected_object'] == 'strawberry'
    assert report['candidate_eligible'] == {'banana': False, 'strawberry': True, 'lemon': True}
    assert all(row['unique_top'] == 2 for row in report['rows'])
    assert report['rows'][0]['factual_minus_purple'] < 0.1
    assert report['rows'][3]['factual_minus_purple'] > 0.1
    assert math.isclose(report['rows'][3]['factual_plus_purple_probability'],
                        report['rows'][3]['probabilities']['0'] + report['rows'][3]['probabilities']['1'])


def test_all_ineligible_and_missing_or_changed_records_are_preserved_or_rejected():
    manifest, events = fixture()
    for event in events[1::2]:
        event['logits'] = records.encode_raw(struct.pack('<3f', 0.0, 0.0, 8.0))
    result = screen.rebuild_screen(manifest, events, vocabulary=3)
    assert result['status'] == 'baseline-ineligible' and result['selected_object'] is None
    with pytest.raises(ValueError):
        screen.rebuild_screen(manifest, events[:-1], vocabulary=3)
    bad = copy.deepcopy(events)
    bad[1]['attention_mask'] = [1, 0, 1, 1]
    with pytest.raises(ValueError):
        screen.rebuild_screen(manifest, bad, vocabulary=3)


def test_pair_mass_can_survive_when_each_rounded_probability_underflows():
    manifest, events = fixture()
    for event in events[1::2]:
        event['logits'] = records.encode_raw(struct.pack('<3f', -745.625, -745.5, 0.0))
    report = screen.rebuild_screen(manifest, events, vocabulary=3)
    row = report['rows'][0]
    assert row['probabilities']['0'] == row['probabilities']['1'] == 0.0
    assert row['factual_plus_purple_probability'] > 0.0
    assert row['eligible']


def test_candidate_order_and_duplicate_ids_are_rejected():
    manifest, events = fixture()
    manifest['candidate_order'].reverse()
    with pytest.raises(ValueError):
        screen.rebuild_screen(manifest, events, vocabulary=3)
