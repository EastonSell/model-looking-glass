"""Fabricated-only tests of held-out gates, tracks and the inherited case loop."""
import pytest
import torch

from looking_glass.flan_t5_dose_response import LEVELS
from looking_glass.flan_t5_equal_dose import Evidence
from looking_glass.flan_t5_heldout_experiment import baseline_gate, run_cases, all_track_decisions


def test_context_reference_is_a_separate_strict_filter():
    gate = baseline_gate([2.0, -2.0], 3.0)
    assert gate['raw_supported'] and not gate['strict_supported'] and gate['gap_supported']
    assert gate['context_reasons'] == ['first-not-above-context']
    gate = baseline_gate([4.0, 2.0], 0.0)
    assert not gate['raw_supported'] and not gate['strict_supported'] and gate['gap_supported']
    assert 'second-margin-not-negative' in gate['raw_reasons']
    with pytest.raises(ValueError):
        baseline_gate([float('nan'), 0.0], 0.0)


def test_semantics_raw_strict_gap_and_dose_precedence_in_full_fake_loop(tmp_path):
    cases = []
    for pair in range(6):
        for template in ('template0', 'template1'):
            case_id = f'pair{pair}-{template}'
            cases.append({'case_id': case_id, 'pair_id': pair, 'template_id': template,
                          'semantic_eligible': case_id != 'pair3-template0',
                          'semantic_reasons': ['ambiguous'] if case_id == 'pair3-template0' else [],
                          'word_position': 0, 'instruction_position': 1, 'context_position': 2,
                          'neutral': {'case': case_id, 'side': None},
                          'source': [{'case': case_id, 'side': side} for side in (0, 1)],
                          'targets': [{'target': 0}, {'target': 1}]})

    def make_state(case_id, side):
        result = torch.zeros((1, 3, 512), dtype=torch.float32)
        if side and not case_id.startswith('pair5-'):
            result[0, :, 0] = torch.tensor([3.0, 6.0, 9.0])
        return result

    evidence, records, active = Evidence(tmp_path / 'attempt'), [], {}

    def fake_forward(source, target, name, action='clean', value=None, token_index=None):
        records.append(name)
        case_id, side = source['case'], source['side']
        if side is None:
            context = 3.0 if case_id.startswith('pair1-') else 0.0
            return context if target['target'] == 0 else 0.0, torch.tensor([0.0]), torch.zeros((1, 3, 512)), None, None
        own = make_state(case_id, side)
        state, before, after = own.clone(), None, None
        if action not in ('clean', 'capture'):
            before = own.clone() if token_index is None else own[0, token_index].clone()
            after = (value.clone() if action == 'patch' else torch.zeros_like(before)
                     if action == 'zero' else before.clone())
            if token_index is None:
                state = after.clone()
            else:
                state[0, token_index] = after
        logit = float(1-side if '-full_swap-' in name else side)
        margin = 2.0 if side == 0 else -2.0
        if case_id.startswith('pair2-') and side == 1:
            margin = 1.0
        if case_id.startswith('pair4-'):
            margin = .2 if side == 0 else -.2
        for a, b in LEVELS:
            if f'-matched_word_{a}of{b}-' in name:
                margin += (-1 if side == 0 else 1) * 1.2 * a / b
        score = margin if target['target'] == 0 else 0.0
        return score, torch.tensor([logit]), state, before, after

    summaries = run_cases({'cases': cases}, [[] for _ in cases], evidence, records, active, fake_forward)
    assert [c['status'] for c in summaries] == [
        'positive', 'positive', 'baseline-unsupported', 'baseline-unsupported',
        'baseline-unsupported', 'baseline-unsupported', 'semantic-ineligible', 'positive',
        'baseline-unsupported', 'baseline-unsupported', 'dose-ineligible', 'dose-ineligible']
    assert [c['raw_status'] for c in summaries[2:6]] == [
        'positive', 'positive', 'baseline-unsupported', 'baseline-unsupported']
    assert [c['descriptive_status'] for c in summaries[2:6]] == ['positive'] * 4
    assert all(c['descriptive_status'] == 'gap-ineligible' for c in summaries[8:10])
    tracks = all_track_decisions(summaries)
    assert tracks['primary']['status'] == tracks['raw_gate']['status'] == 'inconclusive'
    assert tracks['gap_descriptive']['status'] == 'positive'
    assert len(records) == 570
    assert not any(name.startswith('pair3-template0-') for name in records)
    assert len([name for name in records if name.startswith('pair2-template0-')]) == 78
