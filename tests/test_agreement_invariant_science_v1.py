"""Hand-checked invariant-noun statistics; no tokenizer or model."""
import hashlib
import importlib
import json
import math
import re
from collections import Counter
from decimal import Decimal, localcontext

import pytest


def science():
    try:
        return importlib.import_module('looking_glass.agreement_invariant_science_v1')
    except ModuleNotFoundError as error:
        if error.name == 'looking_glass.agreement_invariant_science_v1':
            pytest.fail('invariant-noun science implementation is missing')
        raise


def test_factorial_preserves_signed_contrasts_and_negative_distance_difference():
    contrast = science().factorial
    assert contrast((3.0, 1.0, 2.0, 0.0)) == {
        'k_a': 2.0, 'k_b': 2.0, 's': 2.0, 'd_num': 2.0,
        'd_lex': 1.0, 'delta': 1.0, 'l': 1.0, 'i': 0.0,
        'a_this_minus_b_this': 1.0, 'a_these_minus_b_these': 1.0,
    }
    negative = contrast((0.0, 0.0, 2.0, 2.0))
    assert negative['s'] == 0.0 and negative['d_num'] == 0.0
    assert negative['d_lex'] == 2.0 and negative['delta'] == -2.0


def test_factorial_rejects_nonfinite_or_boolean_values():
    contrast = science().factorial
    for values in ((0.0, 0.0, 0.0, float('nan')), (0.0, 0.0, 0.0, True)):
        with pytest.raises(ValueError):
            contrast(values)


def baseline_cells():
    return [{'frame': frame, 'context': context,
             'margins': (1.0, -1.0, 1.0, -1.0)}
            for frame in range(2) for context in range(3)]


def test_baseline_counts_ties_as_incorrect_and_checks_all_six_cells():
    check = science().assess_baseline
    cells = baseline_cells()
    result = check(cells)
    assert result['overall_correct'] == 24
    assert result['frame_correct'] == [12, 12]
    assert result['noun_correct'] == [12, 12]
    assert result['context_correct'] == [8, 8, 8]
    assert result['all_correct_cells'] == 6
    assert result['adequate'] is True

    cells[0]['margins'] = (0.0, -1.0, 1.0, -1.0)
    result = check(cells)
    assert result['overall_correct'] == 23 and result['ties'] == 1
    assert result['all_correct_cells'] == 5 and result['adequate'] is True

    # Overall, frame, noun and context totals still meet their thresholds,
    # but three distinct cells each contain a tie, so the cell rule fails.
    cells[1]['margins'] = (1.0, 0.0, 1.0, -1.0)
    cells[2]['margins'] = (1.0, -1.0, 0.0, -1.0)
    result = check(cells)
    assert result['overall_correct'] == 21 and result['ties'] == 3
    assert result['all_correct_cells'] == 3 and result['adequate'] is False
    assert result['frame_correct'] == [9, 12]
    assert result['noun_correct'] == [10, 11]
    assert result['context_correct'] == [7, 7, 7]


def test_baseline_rejects_missing_duplicate_or_invalid_cells():
    check = science().assess_baseline
    cells = baseline_cells()
    with pytest.raises(ValueError):
        check(cells[:-1])
    with pytest.raises(ValueError):
        check(cells[:-1] + [cells[0]])
    cells[0]['margins'] = (1.0, -1.0, True, -1.0)
    with pytest.raises(ValueError):
        check(cells)


def criteria_cells():
    return [{'frame': frame, 'context': context,
             'u': (0.125, 0.0, 0.125, 0.0),
             'v1': (0.0, 0.0, 0.0, 0.0),
             'v2': (0.0, 0.0, 0.0, 0.0)}
            for frame in range(2) for context in range(3)]


def test_criteria_keep_raw_results_when_baseline_is_inadequate():
    check = science().assess_criteria
    cells = criteria_cells()
    result = check(cells, True)
    assert result['sensitivity']['u_mean'] == 0.125
    assert result['sensitivity']['noun_k_means'] == [0.125, 0.125]
    assert result['sensitivity']['positive_cells'] == 6
    assert result['number_over_noun']['u_mean'] == 0.125
    assert result['sensitivity']['decision'] == 'pass'
    assert result['number_over_noun']['decision'] == 'pass'
    limited = check(cells, False)
    for criterion in ('sensitivity', 'number_over_noun'):
        assert limited[criterion]['raw_pass'] is True
        assert limited[criterion]['decision'] == 'baseline_limited'


def test_control_comparison_uses_individual_absolute_values_and_is_strict():
    check = science().assess_criteria
    cells = criteria_cells()
    # Alternating signs cancel in a signed mean but not in mean |control|.
    for index, cell in enumerate(cells):
        x = 0.125 if index % 2 == 0 else -0.125
        cell['v1'] = (x, 0.0, x, 0.0)
    result = check(cells, True)
    for criterion in ('sensitivity', 'number_over_noun'):
        assert result[criterion]['v1_abs_mean'] == 0.125
        assert result[criterion]['raw_pass'] is False
        assert result[criterion]['decision'] == 'fail'


def test_criteria_require_four_positive_cells_and_both_positive_frames():
    check = science().assess_criteria
    cells = criteria_cells()
    for cell in cells[:3]:
        cell['u'] = (-0.0625, 0.0, -0.0625, 0.0)
    result = check(cells, True)
    assert result['sensitivity']['positive_cells'] == 3
    assert result['sensitivity']['frame_means'] == [-0.0625, 0.125]
    assert result['sensitivity']['decision'] == 'fail'
    with pytest.raises(ValueError):
        check(cells, 1)


def test_independently_reconstructed_schedule_has_fixed_digest_and_order():
    schedule = science().fixed_schedule()
    assert len(schedule) == 612
    assert len({case_id for case_id, _ in schedule}) == 68
    assert all(case_id.startswith('discovery_') for case_id, _ in schedule[:306])
    assert all(case_id.startswith('evaluation_') for case_id, _ in schedule[306:])
    slots = Counter()
    for start in range(0, len(schedule), 9):
        block = schedule[start:start + 9]
        assert len({case_id for case_id, _ in block}) == 1
        assert [action for _, action in block[:2]] == ['clean', 'noop']
        assert block[-1][1] == 'clean_after'
        assert len({action for _, action in block[2:8]}) == 6
        slots.update((action, index) for index, (_, action) in enumerate(block[2:8]))
    assert all(value in (11, 12) for value in slots.values())
    raw = json.dumps(schedule, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    assert hashlib.sha256(raw).hexdigest() == 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6'


def fabricated_margins():
    schedule = science().fixed_schedule()
    values = {}
    for case_id, action in schedule:
        target = '_target_' in case_id
        baseline = (-1.0 if case_id.endswith('_these') else 1.0) if target else 0.0
        shift = (0.125 if target and case_id.endswith('_this') else
                 0.25 if '_fragment_' in case_id else 0.0)
        if action == 'u_plus':
            baseline += shift
        elif action == 'u_minus':
            baseline -= shift
        values.setdefault(case_id, {})[action] = {
            'present': baseline, 'past': 0.0, 'the_vs_a': 0.0, 'in_vs_on': 0.0}
    return values


def test_full_fixed_scoring_retains_every_case_control_and_cell():
    report = science().score_margins(fabricated_margins())
    assert len(report['cases']) == 68
    assert len(report['fragments']) == 12
    assert len(report['demonstratives']) == 8
    assert report['fragments'][0]['odd']['u']['present'] == 0.25
    assert all(row['context'] == 0 for row in report['demonstratives'])
    for split in ('discovery', 'evaluation'):
        partition = report['partitions'][split]
        assert len(partition['cells']) == 6
        assert partition['baseline']['overall_correct'] == 24
        assert partition['baseline']['adequate'] is True
        assert partition['criteria']['sensitivity']['decision'] == 'pass'
        assert partition['criteria']['number_over_noun']['decision'] == 'pass'
        for cell in partition['cells']:
            present = cell['statistics']['present']['u']
            assert present['s'] == present['delta'] == 0.125
            assert len(cell['case_ids']) == 4
        summary = partition['aggregates']['overall']['present']['u']['s']
        assert summary == {'mean': 0.125, 'positive': 6, 'zero': 0, 'negative': 0}
        assert partition['aggregates']['by_frame']['0']['present']['u']['s']['positive'] == 3
        assert partition['aggregates']['by_context']['2']['present']['u']['s']['positive'] == 2


def test_scoring_rejects_missing_or_malformed_actions_before_results():
    check = science().score_margins
    values = fabricated_margins()
    first = next(iter(values))
    del values[first]['clean_after']
    with pytest.raises(ValueError):
        check(values)
    values = fabricated_margins()
    values[first]['clean']['present'] = float('nan')
    with pytest.raises(ValueError):
        check(values)
    values = fabricated_margins()
    values[first]['clean_after']['present'] += 1.0
    with pytest.raises(ValueError):
        check(values)


def fabricated_case_rows():
    result = []
    schedule = science().fixed_schedule()
    for start in range(0, len(schedule), 9):
        case_id = schedule[start][0]
        target = re.fullmatch(r'(discovery|evaluation)_target_f([01])_c([012])_n([01])_(this|these)', case_id)
        fragment = re.fullmatch(r'(discovery|evaluation)_fragment_f([01])_c([012])', case_id)
        demo = re.fullmatch(r'(discovery|evaluation)_demonstrative_f([01])_(this|these)', case_id)
        if target:
            split, frame, context, noun, det = target.groups()
            role, noun, context = 'target', int(noun), int(context)
        elif fragment:
            split, frame, context = fragment.groups()
            role, noun, context, det = 'fragment', None, int(context), None
        else:
            split, frame, det = demo.groups()
            role, noun, context = 'demonstrative', None, 0
        frame = int(frame)
        split_index = ('discovery', 'evaluation').index(split)
        frame_token = 100 + split_index * 2 + frame
        final = 140 if context == 0 else 150 + split_index * 4 + frame * 2 + context - 1
        determiner = {'this': 428, 'these': 777}.get(det)
        ids = [frame_token, final]
        if role == 'target':
            ids = [frame_token, determiner, 90 + split_index * 2 + noun, final]
        elif role == 'demonstrative':
            ids = [frame_token, determiner, final]
        row = {'case_id': case_id, 'split': split, 'role': role, 'frame_index': frame,
               'context_index': context, 'noun_index': noun, 'determiner': det,
               'text': 'fabricated', 'token_ids': ids, 'final_index': len(ids) - 1}
        if role != 'fragment':
            row['determiner_index'] = 1
        if role == 'target':
            row['subject_index'] = 2
        result.append(row)
    return result


def test_case_alignment_and_fixed_coordinates_are_checked():
    check = science().validate_cases
    cases = fabricated_case_rows()
    assert check(cases, vocabulary=800) is True
    target = next(row for row in cases if row['role'] == 'target')
    target['token_ids'][0] += 20
    with pytest.raises(ValueError):
        check(cases, vocabulary=800)
    cases = fabricated_case_rows()
    cases[0]['case_id'], cases[1]['case_id'] = cases[1]['case_id'], cases[0]['case_id']
    with pytest.raises(ValueError):
        check(cases, vocabulary=800)


def test_case_validator_rejects_wrong_determiner_and_cross_case_prefix():
    check = science().validate_cases
    cases = fabricated_case_rows()
    target = next(row for row in cases if row['role'] == 'target')
    target['token_ids'][target['determiner_index']] = 777 if target['determiner'] == 'this' else 428
    with pytest.raises(ValueError):
        check(cases, vocabulary=800)
    cases = fabricated_case_rows()
    cases[1]['token_ids'] = cases[0]['token_ids'] + [318]
    cases[1]['final_index'] = len(cases[1]['token_ids']) - 1
    with pytest.raises(ValueError):
        check(cases, vocabulary=800)


def test_record_schedule_binds_case_metadata_and_each_arm_slot():
    cases = fabricated_case_rows()
    records = science().expected_records(cases, vocabulary=800)
    pairs = science().fixed_schedule()
    assert len(records) == 612
    for index, row in enumerate(records):
        case = cases[index // 9]
        assert row == {'index': index, 'case_id': pairs[index][0], 'case_index': index // 9,
                       'split': case['split'], 'frame_index': case['frame_index'],
                       'context_index': case['context_index'], 'role': case['role'],
                       'noun_index': case['noun_index'], 'determiner': case['determiner'],
                       'action': pairs[index][1],
                       'slot': (index % 9 - 2 if 2 <= index % 9 < 8 else None)}


def test_stable_kl_resolves_tiny_tails_without_negative_roundoff():
    divergence = science().stable_kl
    with localcontext() as ctx:
        ctx.prec = 450
        p_tail, q_tail = Decimal(-100).exp(), Decimal(-80).exp()
        p = [1 / (1 + p_tail), p_tail / (1 + p_tail)]
        q = [1 / (1 + q_tail), q_tail / (1 + q_tail)]
        expected = float(sum(a * (a / b).ln() for a, b in zip(p, q)))
    actual, underflow = divergence((0.0, -100.0), (0.0, -80.0))
    assert actual > 0.0 and not underflow
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=0.0)


def test_stable_kl_accumulates_underflowed_p_and_reports_final_underflow():
    divergence = science().stable_kl
    # p's second weight underflows in binary64 but q's contribution is representable.
    actual, underflow = divergence((0.0, -800.0), (0.0, -735.0))
    assert actual > 0.0 and not underflow
    assert math.isclose(actual, math.exp(-735.0), rel_tol=0.001, abs_tol=5e-324)
    assert divergence((0.0, -2000.0), (-700.0, 0.0))[0] == pytest.approx(700.0)
    assert divergence((0.0, -2e38), (-2.0, 0.0))[0] == pytest.approx(2.0 + math.log1p(math.exp(-2.0)))
    assert divergence((0.0, -1.0), (10.0, 9.0)) == (0.0, False)
    assert divergence((0.0, -2000.0), (0.0, -1500.0)) == (0.0, True)


def test_stable_kl_rechecks_calling_thread_arithmetic_before_return(monkeypatch):
    module = science()
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('binary64 subnormal arithmetic differs')
    monkeypatch.setattr(module, 'require_binary64_subnormals', guard)
    with pytest.raises(ValueError, match='subnormal'):
        module.stable_kl((0.0, -100.0), (0.0, -80.0))
    assert len(calls) == 2


def test_stable_kl_resolves_f32_minimum_change_among_tied_logits():
    epsilon = math.ldexp(1.0, -149)
    expected = math.ldexp(1.0, -301)
    actual, underflow = science().stable_kl((0.0, 0.0), (0.0, -epsilon))
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert underflow is False
    actual, underflow = science().stable_kl((0.0, 0.0, -1000.0), (0.0, -epsilon, -1020.0))
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert underflow is False


def test_normalizer_retains_aggregate_of_underflowed_weight_changes():
    count = 50000
    clean = (0.0,) + (-745.2,) * count
    arm = (0.0,) + (-744.9,) * count
    with localcontext() as ctx:
        ctx.prec = 500
        p_weight = Decimal.from_float(-745.2).exp()
        q_weight = Decimal.from_float(-744.9).exp()
        expected = float(((1 + count * q_weight) / (1 + count * p_weight)).ln())
    actual = science()._normalizer_shift(clean, arm, 0.0, 0.0, 0.0, 0.0)
    assert expected > 0.0
    assert actual == pytest.approx(expected, rel=0.001, abs=5e-324)


def test_kl_mixed_underflow_and_large_common_offset():
    epsilon = math.ldexp(1.0, -149)
    clean, arm = (0.0, 0.0, -800.0), (0.0, -epsilon, -230.0)
    with localcontext() as ctx:
        ctx.prec = 500
        p = [Decimal.from_float(x).exp() for x in clean]
        q = [Decimal.from_float(x).exp() for x in arm]
        ps, qs = sum(p), sum(q)
        p = [x / ps for x in p]; q = [x / qs for x in q]
        expected = float(sum(x * (x / y).ln() for x, y in zip(p, q)))
    actual, flag = science().stable_kl(clean, arm)
    assert not flag and actual == pytest.approx(expected, rel=1e-11, abs=0.0)
    clean, arm = (0.0, 0.0, -100.0), (0.0, -0.125, -80.0)
    baseline = science().stable_kl(clean, arm)
    offset = float(2**20)
    shifted = science().stable_kl(tuple(x + offset for x in clean), tuple(x + offset for x in arm))
    assert shifted[0] == pytest.approx(baseline[0], rel=1e-12, abs=0.0)
    assert shifted[1] == baseline[1]
