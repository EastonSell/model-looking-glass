import copy

import pytest

from looking_glass import year_label_control as v1
from looking_glass import year_label_control_v2 as v2


def _scores(rows):
    clean = {r['case_id']: {'band_score': 0.0, 'local_score': 0.0} for r in rows}
    actions = {}
    for row in rows:
        year = -0.3 if row['year_role'] == 'low' else 0.3
        actions[row['case_id']] = {'self': 0.0, 'year_final': year, 'label_final': -0.01,
                                   'year_at_zero': 0.02, 'rotated_year_final': -0.03}
    return clean, actions


def test_selection_retains_exact_original_indices_and_donors():
    original = v1.make_inventory()['cases']
    selected = v2.select_cases(original)
    assert selected == original[12:] and len(selected) == 36
    assert {r['family_index'] for r in selected} == {1, 2, 3}
    assert all(r['year_donor_case_id'] in {x['case_id'] for x in selected} for r in selected)
    changed = copy.deepcopy(original)
    changed[12]['rotation'] = 0
    with pytest.raises(ValueError):
        v2.select_cases(changed)


def test_summary_preserves_strict_per_target_and_three_family_denominator():
    rows = v2.select_cases(v1.make_inventory()['cases'])
    clean, actions = _scores(rows)
    summary = v2.summarize_interventions(rows, clean, actions)
    assert [f['family_index'] for f in summary['families']] == [1, 2, 3]
    assert summary['pattern_count'] == 3 and summary['pattern_supported'] is True
    actions[rows[0]['case_id']]['year_final'] = -0.1
    summary = v2.summarize_interventions(rows, clean, actions)
    assert [f['pass'] for f in summary['families']] == [False, True, True]
    assert summary['pattern_count'] == 2 and summary['pattern_supported'] is False


def test_summaries_reject_missing_cases_or_changed_self():
    rows = v2.select_cases(v1.make_inventory()['cases'])
    clean, actions = _scores(rows)
    assert [f['family_index'] for f in v2.summarize_clean(rows, clean)['families']] == [1, 2, 3]
    actions[rows[-1]['case_id']]['self'] = 0.01
    with pytest.raises(ValueError):
        v2.summarize_interventions(rows, clean, actions)
    del clean[rows[0]['case_id']]
    with pytest.raises(ValueError):
        v2.summarize_clean(rows, clean)
