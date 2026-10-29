"""Fabricated-only checks for the prospective sentiment schedule and dose math."""
import math
import struct

import pytest
import torch

from looking_glass.flan_t5_sentiment_numeric import (
    ACTIONS, RECEIVERS, _norm, fixed_schedule, group_decision, plan_group, primary_decision,
    runtime_numeric_probe,
)


def fake_states():
    # Four synthetic state vectors at each position; no tokenizer or checkpoint.
    return {name: torch.tensor([[[float(i), 0.0], [float(i * 2), 0.0],
                                 [float(i * 3), 0.0], [3.0, 4.0]]])
            for i, name in enumerate(RECEIVERS)}


def test_fixed_schedule_has_required_global_controls_and_rotations():
    schedule = fixed_schedule()
    assert len(schedule) == 484
    assert [sum(item['kind'] == k for item in schedule) for k in
            ('clean', 'passthrough', 'self', 'full_swap', 'matched', 'natural', 'repeat')] == [
                32, 32, 96, 32, 256, 32, 4]
    assert all(item['kind'] == 'clean' for item in schedule[:32])
    assert all(item['kind'] in {'passthrough', 'self', 'full_swap'} for item in schedule[32:192])
    assert schedule[0] == {'kind': 'clean', 'unit': 0, 'instruction': 0, 'receiver': 'P1'}
    assert [x['receiver'] for x in schedule[4:8]] == ['N1', 'P2', 'N2', 'P1']
    first = [x for x in schedule if x['kind'] == 'matched' and x['unit'] == 0
             and x['instruction'] == 0 and x['receiver'] == 'P1']
    assert [(x['action'], x['dose']) for x in first] == [
        ('opposite_adjective', 'low'), ('opposite_adjective', 'high'),
        ('same_adjective', 'high'), ('same_adjective', 'low'),
        ('opposite_instruction', 'low'), ('opposite_instruction', 'high'),
        ('opposite_subject', 'high'), ('opposite_subject', 'low')]
    for unit in range(4):
        repeats = [x for x in schedule if x['kind'] == 'repeat' and x['unit'] == unit]
        assert repeats == [{'kind': 'repeat', 'unit': unit,
                            'instruction': 0 if unit % 2 == 0 else 1, 'receiver': 'P1'}]
        assert schedule.index(repeats[0]) > max(i for i, x in enumerate(schedule)
                                                if x['unit'] == unit and x['kind'] == 'natural')


def test_group_plans_16_distances_and_32_matched_values_with_reciprocal_donors():
    runtime_numeric_probe()
    result = plan_group(fake_states(), {'instruction': 0, 'subject': 1, 'adjective': 2})
    assert result['status'] == 'matchable'
    assert len(result['distances']) == 16
    assert len(result['landings']) == 32
    assert result['q'] == 1.0
    for receiver in RECEIVERS:
        for action in ACTIONS:
            assert result['distances'][f'{receiver}:{action}'] > 0
            for level, expected in [('low', .5), ('high', 1.0)]:
                landing = result['landings'][f'{receiver}:{action}:{level}']
                assert landing['target_dose'] == expected
                assert landing['achieved_norm'] == pytest.approx(expected)
                assert landing['cosine'] == pytest.approx(1.0)
                assert len(landing['value']) == 8
    assert result['landings']['P1:opposite_adjective:high']['donor'] == 'N1'
    assert result['landings']['N1:same_adjective:high']['donor'] == 'N2'


def test_zero_distance_makes_group_ineligible_without_discarding_distances():
    states = fake_states()
    states['N1'][0, 0] = states['P1'][0, 0]
    result = plan_group(states, {'instruction': 0, 'subject': 1, 'adjective': 2})
    assert result['status'] == 'dose-ineligible'
    assert result['q'] == 0
    assert len(result['distances']) == 16
    assert result['landings'] == {}


def test_rounded_away_landing_is_recorded_and_stops_group():
    states = fake_states()
    for i, name in enumerate(RECEIVERS):
        states[name][0, 0] = torch.tensor([float(i) * .001, 0.0])
        states[name][0, 2] = torch.tensor([100_000_000.0 + 16.0 * i, 0.0])
    result = plan_group(states, {'instruction': 0, 'subject': 1, 'adjective': 2})
    assert result['q'] > 2e-6
    assert result['status'] == 'dose-unmatchable'
    rounded = result['landings']['P1:opposite_adjective:low']
    assert len(result['landings']) == 32
    assert rounded['achieved_norm'] == 0.0
    assert rounded['cosine'] is None
    assert 'zero' in rounded['failure']
    assert struct.unpack('<f', rounded['value'][:4])[0] == 100_000_000.0


def test_nonfinite_state_is_an_incomplete_error():
    states = fake_states()
    states['P1'][0, 1, 0] = math.inf
    with pytest.raises(ValueError, match='nonfinite'):
        plan_group(states, {'instruction': 0, 'subject': 1, 'adjective': 2})


def test_equal_norm_is_insufficient_when_rounding_misses_direction():
    adjective = [(100_000_000., 100_000_000.), (100_000_008., 100_000_056.),
                 (100_000_080., 100_000_080.), (100_000_136., 100_000_136.)]
    states = {name: torch.tensor([[[8.0 * i, 0.0], [16.0 * i, 0.0],
                                    list(adjective[i]), [0.0, 0.0]]])
              for i, name in enumerate(RECEIVERS)}
    result = plan_group(states, {'instruction': 0, 'subject': 1, 'adjective': 2})
    landing = result['landings']['P1:opposite_adjective:high']
    assert result['q'] == 8.0
    assert landing['achieved_norm'] == 8.0
    assert landing['cosine'] == pytest.approx(7 / math.sqrt(50))
    assert landing['failure'] == 'landing cosine below threshold'
    assert result['status'] == 'dose-unmatchable'


def test_group_and_fixed_four_unit_decision_use_absolute_individual_controls():
    baseline = {'P1': 1., 'N1': -1., 'P2': 1., 'N2': -1.}
    target = {'P1': .75, 'N1': -.75, 'P2': .75, 'N2': -.75}
    mixed = {'P1': .875, 'N1': -1.125, 'P2': 1.125, 'N2': -.875}
    group = group_decision(baseline, {'opposite_adjective': target,
                                      'same_adjective': mixed,
                                      'opposite_instruction': baseline,
                                      'opposite_subject': baseline})
    assert group['A'] == .25
    assert group['control_means']['same_adjective'] == .125
    assert group['B'] == .125
    assert group['passes'] is True
    units = [[group, group], [group, group], [group, group],
             [group, {'status': 'baseline-unsupported'}]]
    final = primary_decision(units, attempt_complete=True)
    assert final == {'status': 'supported', 'passed_units': 3, 'total_units': 4}
    assert primary_decision(units, attempt_complete=False) == {
        'status': 'incomplete', 'passed_units': None, 'total_units': 4}


def test_norm_uses_the_protocols_literal_binary64_power():
    values = tuple(float.fromhex(x) for x in (
        '0x1.8e3cfdp+2', '-0x1.0e5592p+3', '0x1.441b9p+0', '-0x1.0a778bcp+3',
        '0x1.96c3c6ap+1', '-0x1.500d4ap+3', '-0x1.99bf2p+0', '0x1.05299ap+2',
        '0x1.ecd633p+2', '0x1.714a7ep+0', '-0x1.b5a94a4p+2', '0x1.bcffe58p+1',
        '-0x1.fd2e7ep+2', '-0x1.87b5aa6p+2', '-0x1.2e159edcp+2', '0x1.a156f8p-1'))
    assert _norm(values).hex() == '0x1.7c2abfdd72014p+4'
