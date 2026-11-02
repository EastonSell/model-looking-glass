import math

import pytest

from looking_glass.causal_metrics import InterventionCase, summarize_interventions


def test_signed_pairs_and_all_excluded_cases_are_retained():
    cases = [
        InterventionCase('positive', 'supported', 'eligible', 2.0, 5.0, 3.0),
        InterventionCase('negative', 'supported', 'eligible', 5.0, 3.0, 4.0),
        InterventionCase('tie', 'supported', 'eligible', -3.0, -1.0, -1.0),
        InterventionCase('unsupported', 'unsupported', 'ineligible', 1.0, 4.0, 2.0),
        InterventionCase('unverified', 'unverified', 'eligible', 1.0, 3.0, 2.0),
        InterventionCase('ineligible', 'supported', 'ineligible', 1.0, 6.0, 0.0),
        InterventionCase('unknown', 'supported', 'unverified', 1.0, 2.0, 1.0),
        InterventionCase('missing_reference', 'supported', 'eligible', None, 7.0, 4.0),
        InterventionCase('missing_arm', 'supported', 'eligible', 1.0, None, 2.0),
    ]
    summary = summarize_interventions(cases)
    assert summary['n_cases'] == 9
    assert summary['baseline_status_counts'] == {'supported': 7, 'unsupported': 1, 'unverified': 1}
    assert summary['eligibility_counts'] == {'eligible': 6, 'ineligible': 2, 'unverified': 1}
    assert summary['missing_readout_counts'] == {'reference': 1, 'candidate': 1, 'control': 0}
    assert summary['paired_cases'] == {
        'included': 3, 'excluded': 6,
        'exclusion_counts': {'baseline_unsupported': 1, 'baseline_unverified': 1,
                             'intervention_ineligible': 1, 'intervention_unverified': 1,
                             'missing_readout': 2},
    }
    rows = {row['case_id']: row for row in summary['cases']}
    assert list(rows) == [case.case_id for case in cases]
    assert (rows['positive']['candidate_effect'], rows['positive']['control_effect'],
            rows['positive']['candidate_minus_control']) == (3.0, 1.0, 2.0)
    assert (rows['negative']['candidate_effect'], rows['negative']['control_effect'],
            rows['negative']['candidate_minus_control']) == (-2.0, -1.0, -1.0)
    assert rows['unsupported']['candidate_effect'] == 3.0
    assert rows['unsupported']['exclusion_reason'] == 'baseline_unsupported'
    assert rows['missing_reference']['candidate_effect'] is None
    assert rows['missing_reference']['candidate_minus_control'] == 3.0
    assert rows['missing_arm']['candidate_effect'] is None
    assert rows['missing_arm']['control_effect'] == 1.0
    assert summary['point'] == {'mean_candidate_effect': 1.0,
                                'mean_control_effect': 2 / 3,
                                'mean_candidate_minus_control': 1 / 3,
                                'candidate_above_control': 1,
                                'candidate_equal_control': 1,
                                'candidate_below_control': 1}


def test_zero_included_denominator_has_undefined_means():
    summary = summarize_interventions([
        InterventionCase('not_eligible', 'supported', 'ineligible', 0, 1, 0),
        InterventionCase('incomplete', 'supported', 'eligible', 0, 1, None),
    ])
    assert summary['paired_cases']['included'] == 0
    assert summary['paired_cases']['excluded'] == 2
    assert summary['point'] == {'mean_candidate_effect': None, 'mean_control_effect': None,
                                'mean_candidate_minus_control': None,
                                'candidate_above_control': 0, 'candidate_equal_control': 0,
                                'candidate_below_control': 0}


def test_exclusion_precedence_and_overlapping_missing_counts():
    summary = summarize_interventions([
        InterventionCase('both', 'unverified', 'ineligible', None, None, None),
        InterventionCase('missing_control', 'supported', 'eligible', 2, 4, None),
    ])
    assert [row['exclusion_reason'] for row in summary['cases']] == ['baseline_unverified', 'missing_readout']
    assert summary['missing_readout_counts'] == {'reference': 1, 'candidate': 1, 'control': 2}
    assert sum(summary['paired_cases']['exclusion_counts'].values()) == 2


@pytest.mark.parametrize('bad', [True, False, float('nan'), float('inf'), -float('inf'), '1'])
def test_rejects_invalid_readouts_even_when_case_would_be_excluded(bad):
    with pytest.raises(ValueError):
        summarize_interventions([InterventionCase('x', 'unsupported', 'eligible', 0, bad, 0)])


def test_rejects_invalid_records_statuses_and_duplicate_ids():
    good = InterventionCase('a', 'supported', 'eligible', 0, 1, 2)
    invalid = [[], [object()], [good, good],
               [InterventionCase('', 'supported', 'eligible', 0, 0, 0)],
               [InterventionCase('a', [], 'eligible', 0, 0, 0)],
               [InterventionCase('a', 'supported', 'unknown', 0, 0, 0)]]
    for cases in invalid:
        with pytest.raises(ValueError):
            summarize_interventions(cases)


def test_rejects_finite_operands_whose_difference_overflows():
    cases = [InterventionCase('effect', 'supported', 'eligible', -1e308, 1e308, 0),
             InterventionCase('contrast', 'unverified', 'unverified', None, 1e308, -1e308)]
    for case in cases:
        with pytest.raises(ValueError, match='finite float range'):
            summarize_interventions([case])


def test_finite_large_mixed_effects_have_stable_mean():
    values = [1e308, 1e308, -1e308, -1e308]
    summary = summarize_interventions(InterventionCase(str(i), 'supported', 'eligible', 0, value, 0)
                                      for i, value in enumerate(values))
    assert summary['paired_cases']['included'] == 4
    assert summary['point']['mean_candidate_effect'] == 0.0
    assert summary['point']['mean_candidate_minus_control'] == 0.0
    assert all(math.isfinite(row['candidate_effect']) for row in summary['cases'])


def test_signed_zero_is_a_tie_not_a_directional_effect():
    summary = summarize_interventions([InterventionCase('zero', 'supported', 'eligible',
                                                        0.0, -0.0, 0.0)])
    assert summary['point']['candidate_above_control'] == 0
    assert summary['point']['candidate_below_control'] == 0
    assert summary['point']['candidate_equal_control'] == 1
