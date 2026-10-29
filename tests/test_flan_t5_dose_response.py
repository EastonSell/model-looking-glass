import struct

import pytest
import torch

from looking_glass.flan_t5_dose_response import (
    LEVELS, match_response_levels, response_decision, run_cases,
)
from looking_glass.flan_t5_equal_dose import Evidence, aggregate_decision, run_forward


def test_three_exact_fractional_doses_in_both_directions():
    states = [torch.zeros((1, 3, 512), dtype=torch.float32) for _ in range(2)]
    states[1][0, :, 0] = torch.tensor([3.0, 6.0, 9.0])
    positions = {'word': 0, 'instruction': 1, 'punctuation': 2}
    dose, vectors = match_response_levels(states, positions)
    assert dose['status'] == 'matchable' and dose['q'] == 3.0
    assert [x['q'] for x in dose['levels']] == [1.0, 2.0, 3.0]
    assert len(vectors) == len(LEVELS) * len(positions) * 2
    for level in dose['levels']:
        assert all(abs(x-level['q']) <= level['tolerance'] for x in level['achieved'].values())


def test_fractional_delivery_matches_independent_scalar_float32_oracle():
    assert torch.set_flush_denormal(False)
    generator = torch.Generator(device='cpu').manual_seed(731)
    positions = {'word': 0, 'instruction': 1, 'punctuation': 2}
    for _ in range(4):
        states = [torch.randn((1, 3, 512), generator=generator, dtype=torch.float32) for _ in range(2)]
        dose, vectors = match_response_levels(states, positions)
        assert dose['status'] == 'matchable'
        for level in dose['levels']:
            a, b = level['numerator'], level['denominator']
            for site, pos in positions.items():
                ratio = level['q'] / dose['distances'][site]
                for side in range(2):
                    own, donor = states[side][0, pos].tolist(), states[1-side][0, pos].tolist()
                    expected = b''.join(struct.pack('<f', x + ratio * (y-x)) for x, y in zip(own, donor))
                    assert vectors[a, b, site, side].numpy().astype('<f4', copy=False).tobytes() == expected


def test_default_forward_and_explicit_evidence_limits_fail_closed(tmp_path):
    evidence = Evidence(tmp_path / 'bounded', limit=2048)
    with pytest.raises(ValueError, match='capacity'):
        evidence.reserve_forward()
    with pytest.raises(ValueError, match='cap'):
        run_forward(None, evidence, [None] * 324, {}, {}, {}, 'over-old-limit')
    with pytest.raises(ValueError, match='invalid retained-evidence limit'):
        Evidence(tmp_path / 'bad', limit=True)


@pytest.mark.parametrize('bad_attribute', ['logits', 'encoder'])
def test_raising_output_attribute_salvages_other_tensor_and_action_witnesses(tmp_path, bad_attribute):
    class BrokenOutput:
        def __init__(self, logits, state):
            self._logits, self._state = logits, state

        @property
        def logits(self):
            if bad_attribute == 'logits':
                raise RuntimeError('injected output getter')
            return self._logits

        @property
        def encoder_last_hidden_state(self):
            if bad_attribute == 'encoder':
                raise RuntimeError('injected output getter')
            return self._state

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.final_layer_norm = torch.nn.Identity()

        def forward(self, input_ids, decoder_input_ids, **kwargs):
            state = self.encoder.final_layer_norm(input_ids.float().unsqueeze(-1).expand(-1, -1, 512).clone())
            logits = torch.zeros((1, decoder_input_ids.shape[1], 32128), dtype=torch.float32)
            return BrokenOutput(logits, state)

    evidence, records, active = Evidence(tmp_path / 'attempt'), [], {'case_id': 'fabricated'}
    source = {'input_ids': [2, 3, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [7, 1]}
    with pytest.raises(RuntimeError, match='output getter'):
        run_forward(FakeModel().eval(), evidence, records, active, source, target, 'broken',
                    'patch', torch.full((512,), 7.0), 1)
    layout = active['tensor_layout']
    other = 'encoder' if bad_attribute == 'logits' else 'logits'
    assert not layout[bad_attribute]['available'] and layout[bad_attribute]['error_type'] == 'RuntimeError'
    assert layout[other]['available'] and layout[other]['bytes'] > 0
    assert layout['before']['bytes'] == layout['after']['bytes'] == 2048
    assert active['return_observed'] and (evidence.path / active['saved_file']).is_file()
    assert (evidence.path / 'forward-000-observed.json').is_file()


def test_smallest_level_must_be_positive_and_rounding_unmatchable_is_not_negative():
    positions = {'word': 0, 'instruction': 1, 'punctuation': 2}
    states = [torch.zeros((1, 3, 512), dtype=torch.float32) for _ in range(2)]
    states[1][0, :, 0] = 2e-6
    dose, vectors = match_response_levels(states, positions)
    assert dose['status'] == 'dose-ineligible' and vectors == {}
    states[1][0, 0, 0] = 6e-6
    for pos in (1, 2):
        states[0][0, pos, 0] = 100.0
        states[1][0, pos, 0] = 100.00000762939453
    dose, vectors = match_response_levels(states, positions)
    assert dose['status'] == 'dose-unmatchable' and len(vectors) == 18
    assert dose['levels'][0]['achieved']['instruction-0'] == 0


def test_response_decision_distinguishes_reversal_nonmonotonic_weak_and_controls():
    movements = {'word': [[.5, .6], [1.0, 1.1], [2.0, 2.1]],
                 'instruction': [[0, 0], [0, 0], [0, 0]],
                 'punctuation': [[0, 0], [0, 0], [0, 0]]}
    assert response_decision(10.0, movements) == ('positive', [])
    movements['word'][1][0] = -.01
    status, reasons = response_decision(10.0, movements)
    assert status == 'negative' and any(x.startswith('reversed-') for x in reasons)
    assert any(x.startswith('nonmonotone-') for x in reasons)
    assert any(x.startswith('control_comparison-') for x in reasons)
    movements['word'][2][1] = .01
    assert any(x.startswith('weak_endpoint-') for x in response_decision(10.0, movements)[1])
    movements['punctuation'][0][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        response_decision(10.0, movements)


def test_ordering_tolerance_uses_prescribed_subtraction_at_rounding_boundary():
    previous = -4.2070601430730603e-13
    word = -0.00010000000042070602
    assert word >= previous - 1e-4
    assert word + 1e-4 < previous
    movements = {'word': [[previous, 1.0], [word, 2.0], [1.0, 3.0]],
                 'instruction': [[0, 0], [0, 0], [0, 0]],
                 'punctuation': [[0, 0], [0, 0], [0, 0]]}
    _, reasons = response_decision(1.0, movements)
    assert 'nonmonotone-side0-level2' not in reasons


def test_full_fake_case_schedule_includes_terminal_repeat_and_preserves_skips(tmp_path):
    cases = []
    for pair in range(3):
        for template in ('translate', 'question'):
            case_id = f'pair{pair}-{template}'
            cases.append({'case_id': case_id, 'pair_id': pair, 'template_id': template,
                          'word_position': 0, 'instruction_position': 1, 'context_position': 2,
                          'neutral': {'case': case_id, 'side': None},
                          'source': [{'case': case_id, 'side': side} for side in (0, 1)],
                          'targets': [{'target': 0}, {'target': 1}]})

    def make_state(case_id, side):
        result = torch.zeros((1, 3, 512), dtype=torch.float32)
        if side and case_id != 'pair2-question':
            result[0, :, 0] = torch.tensor([3.0, 6.0, 9.0])
        return result

    def exercise(path, fail_repeat):
        evidence, records, active = Evidence(path), [], {}
        def fake_forward(source, target, name, action='clean', value=None, token_index=None):
            records.append(name)
            case_id, side = source['case'], source['side']
            if side is None:
                return 0.0, torch.tensor([0.0]), torch.zeros((1, 3, 512)), None, None
            own, donor = make_state(case_id, side), make_state(case_id, 1-side)
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
            if fail_repeat and '-repeat-' in name:
                logit += 100
            margin = 2.0 if side == 0 else -2.0
            if case_id == 'pair2-translate':
                margin = 2.0
            if case_id.startswith('pair0-'):
                for numerator, denominator in LEVELS:
                    if f'-matched_word_{numerator}of{denominator}-' in name:
                        movement = 1.2 * numerator / denominator
                        margin += -movement if side == 0 else movement
            score = margin if target['target'] == 0 else 0.0
            return score, torch.tensor([logit]), state, before, after
        summaries = run_cases({'cases': cases}, [[] for _ in cases], evidence, records, active, fake_forward)
        return summaries, records

    summaries, records = exercise(tmp_path / 'complete', False)
    assert [r['status'] for r in summaries] == ['positive', 'positive', 'negative', 'negative',
                                                'baseline-unsupported', 'dose-ineligible']
    assert summaries[0]['response_curves']['word'][0] == [0.0, 0.0]
    assert len(summaries[0]['response_curves']['word']) == 4
    assert len(summaries[0]['absolute_response_curves']['word']) == 4
    assert aggregate_decision(summaries) == 'inconclusive'
    assert len(records) == 4 * 78 + 2 * 6
    for name in ('pair0-translate', 'pair0-question', 'pair1-translate', 'pair1-question'):
        case_records = [r for r in records if r.startswith(name + '-')]
        assert len(case_records) == 78 and all('-repeat-' in r for r in case_records[-4:])
    with pytest.raises(ValueError, match='identity logits control failed'):
        exercise(tmp_path / 'failed', True)
