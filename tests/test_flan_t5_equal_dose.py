import json
import math
import struct
from types import SimpleNamespace

import pytest
import torch

from looking_glass.flan_t5_equal_dose import (
    Evidence, aggregate_decision, cell_decision, check_publication_observation, dose_matches, hook_snapshot, matched_replacements,
    model_snapshot, require_clean_hooks, run_cases, run_forward, save_dose_evidence, score_logits,
    validate_landing, vector_l2,
)


def test_small_nonzero_requested_dose_cannot_accept_zero_delivery():
    states = [torch.zeros((1, 3, 512), dtype=torch.float32) for _ in range(2)]
    states[1][0, 0, 0] = 2e-6
    for pos in (1, 2):
        states[0][0, pos, 0] = 100.0
        states[1][0, pos, 0] = 100.00000762939453
    dose, vectors = matched_replacements(states, {'word': 0, 'instruction': 1, 'punctuation': 2})
    assert 1e-6 < dose['q'] < 3e-6
    assert dose['status'] == 'dose-unmatchable'
    for site in ('instruction', 'punctuation'):
        for side in (0, 1):
            assert dose['achieved'][f'{site}-{side}'] == 0
            assert vector_l2(vectors[site, side], states[side][0, 1 if site == 'instruction' else 2]) == 0


def test_binary64_formula_casts_once_to_float32_including_subnormals():
    assert torch.set_flush_denormal(False)
    states = [torch.zeros((1, 3, 512), dtype=torch.float32) for _ in range(2)]
    states[0][0, :, 1] = 2.0 ** -149
    states[1][0, 0, :4] = torch.tensor([2.0, -2.0, 2.0 ** -148, 0.125])
    states[1][0, 1, :4] = torch.tensor([3.0, -1.0, 2.0 ** -147, 0.25])
    states[1][0, 2, :4] = torch.tensor([4.0, -1.5, 2.0 ** -146, 0.5])
    positions = {'word': 0, 'instruction': 1, 'punctuation': 2}
    dose, vectors = matched_replacements(states, positions)
    assert dose['status'] == 'matchable'
    for name, pos in positions.items():
        for side in (0, 1):
            own = states[side][0, pos].tolist()
            donor = states[1-side][0, pos].tolist()
            expected = b''.join(struct.pack('<f', a + dose['fractions'][name] * (b-a))
                                for a, b in zip(own, donor))
            assert vectors[name, side].numpy().astype('<f4', copy=False).tobytes() == expected
            assert 0 < dose['achieved'][f'{name}-{side}']
            assert abs(dose['achieved'][f'{name}-{side}'] - dose['q']) <= dose['tolerance']


def test_individually_q_matched_doses_must_also_match_each_other():
    achieved = {f'{site}-{side}': value for side in (0, 1)
                for site, value in (('word', 99.1), ('instruction', 100.9), ('punctuation', 99.5))}
    assert all(abs(x - 100.0) <= 1.0 for x in achieved.values())
    assert not dose_matches(100.0, 1.0, achieved)


def test_eos_and_unequal_target_lengths_are_scored():
    logits = torch.zeros((1, 3, 32128), dtype=torch.float32)
    logits[0, 0, 7], logits[0, 1, 8], logits[0, 2, 1] = 1.0, 2.0, 3.0
    score, pieces = score_logits(logits, [7, 8, 1])
    reference = [float(torch.log_softmax(logits[0, i], -1)[t]) for i, t in enumerate([7, 8, 1])]
    assert pieces == reference and score == math.fsum(reference)
    short_score, short_pieces = score_logits(logits[:, :2], [7, 1])
    assert len(short_pieces) == 2 and short_score == math.fsum(short_pieces)
    with pytest.raises(ValueError):
        score_logits(logits, [7, 8, 9])


def test_pair_aggregate_preserves_missing_and_mixed_cases():
    def make(statuses):
        return [{'pair_id': i // 2, 'template_id': ('translate', 'question')[i % 2], 'status': s}
                for i, s in enumerate(statuses)]
    assert aggregate_decision(make(['positive'] * 4 + ['dose-unmatchable'] * 2)) == 'positive'
    assert aggregate_decision(make(['positive', 'negative', 'negative', 'negative',
                                    'baseline-unsupported', 'baseline-unsupported'])) == 'negative'
    assert aggregate_decision(make(['positive', 'positive', 'positive', 'negative',
                                    'baseline-unsupported', 'baseline-unsupported'])) == 'inconclusive'
    movements = {'matched_word': [2.0, 2.0], 'matched_instruction': [-1.0, 1.0],
                 'matched_punctuation': [1.0, -1.0]}
    assert cell_decision(5.0, movements)
    movements['matched_word'][0] = -2.0
    assert not cell_decision(5.0, movements)


def test_publication_attestation_binds_commit_and_source_bytes(tmp_path):
    commit = 'a' * 40
    source = {'commit': commit, 'source_sha256': {'source.py': 'b' * 64}}
    value = {'schema': 'mlg-flant5-public-observation-v1', 'commit': commit,
             'commit_url': 'https://github.com/EastonSell/model-looking-glass/commit/' + commit,
             'source_sha256': source['source_sha256'], 'observer': 'reviewer', 'observed_utc': 'test'}
    path = tmp_path / 'observation.json'
    path.write_text(json.dumps(value))
    assert json.loads(check_publication_observation(path, source)) == value
    value['source_sha256'] = {'source.py': 'c' * 64}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='observation'):
        check_publication_observation(path, source)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.final_layer_norm = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('marker', torch.ones(1))
        self.mode = 'normal'

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs):
        if self.mode == 'raise':
            raise RuntimeError('fabricated pre-return failure')
        state = self.encoder.final_layer_norm(input_ids.float().unsqueeze(-1).expand(-1, -1, 512).clone())
        logits = torch.zeros((1, decoder_input_ids.shape[1], 32128), dtype=torch.float32)
        logits[:, :, 7] = state.sum() * 0.001
        if self.mode == 'mutate-input':
            input_ids[0, 0] = 999
        return SimpleNamespace(logits=logits, encoder_last_hidden_state=state, past_key_values=None)


def test_same_forward_witnesses_and_saved_bytes(tmp_path):
    model = FakeModel().eval()
    clean_hooks = hook_snapshot(model)
    require_clean_hooks(clean_hooks)
    evidence, records, active = Evidence(tmp_path / 'evidence'), [], {'case_id': 'fabricated'}
    source = {'input_ids': [2, 3, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [7, 1]}
    _, _, clean, _, _ = run_forward(model, evidence, records, active, source, target, 'clean')
    replacement = clean[0, 1] + 1
    _, _, patched, before, after = run_forward(model, evidence, records, active, source, target,
                                                'patch', 'patch', replacement, 1)
    assert torch.equal(before, clean[0, 1])
    assert torch.equal(after, replacement) and torch.equal(patched[0, 1], replacement)
    assert torch.equal(patched[0, 0], clean[0, 0]) and torch.equal(patched[0, 2], clean[0, 2])
    row = records[-1]
    observed = json.loads((evidence.path / 'forward-001-observed.json').read_text())
    assert row['decoder_ids'] == [0, 7] and row['decoder_mask'] == [1, 1]
    assert [step['name'] for step in row['session_metadata']['steps']] == ['before', 'action', 'after']
    assert observed['saved_sha256'] == row['sha256']
    assert row['tensor_layout']['before']['bytes'] == row['tensor_layout']['after']['bytes'] == 2048
    assert hook_snapshot(model) == clean_hooks
    run_forward(model, evidence, records, active, source, target, 'capture', 'capture')
    assert records[-1]['tensor_layout']['capture']['bytes'] == clean.numel() * 4


def test_same_forward_pre_action_mismatch_rejects():
    own = torch.ones((1, 3, 512), dtype=torch.float32)
    before, after = own[0, 1].clone(), own[0, 1].clone() + 1
    state = own.clone()
    state[0, 1] = after
    validate_landing('matched_word', 'patch', 1, own, own, after, after, state, before, after,
                     {'q': vector_l2(after, before), 'tolerance': 1e-5})
    with pytest.raises(ValueError, match='same-forward'):
        validate_landing('matched_word', 'patch', 1, own, own, after, after, state, before + 1, after,
                         {'q': vector_l2(after, before), 'tolerance': 1e-5})


def test_ineligible_and_unmatchable_dose_vectors_are_saved(tmp_path):
    evidence = Evidence(tmp_path / 'dose')
    positions = {'word': 0, 'instruction': 1, 'punctuation': 2}
    states = [torch.zeros((1, 3, 512), dtype=torch.float32) for _ in range(2)]
    ineligible, vectors = matched_replacements(states, positions)
    save_dose_evidence(evidence, 0, 'ineligible', ineligible, states, positions, vectors)
    row = json.loads((evidence.path / 'dose-00.json').read_text())
    assert row['dose']['status'] == 'dose-ineligible'
    assert len(row['layout']) == 12 and (evidence.path / row['file']).stat().st_size == 12 * 2048
    states[1][0, 0, 0] = 2e-6
    for pos in (1, 2):
        states[0][0, pos, 0] = 100.0
        states[1][0, pos, 0] = 100.00000762939453
    unmatchable, vectors = matched_replacements(states, positions)
    save_dose_evidence(evidence, 1, 'unmatchable', unmatchable, states, positions, vectors)
    row = json.loads((evidence.path / 'dose-01.json').read_text())
    assert row['dose']['status'] == 'dose-unmatchable'
    assert len(row['layout']) == 18 and (evidence.path / row['file']).stat().st_size == 18 * 2048


def test_failure_keeps_bounded_return_and_never_reuses_prior_forward_identity(tmp_path):
    model = FakeModel().eval()
    evidence, records, active = Evidence(tmp_path / 'evidence'), [], {'case_id': 'fabricated'}
    source = {'input_ids': [2, 3, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [7, 1]}
    run_forward(model, evidence, records, active, source, target, 'success')
    model.mode = 'raise'
    with pytest.raises(RuntimeError):
        run_forward(model, evidence, records, active, source, target, 'fails-before-return')
    assert active['forward'] == 'fails-before-return' and active['return_observed'] is False
    assert 'saved_file' not in active and 'saved_sha256' not in active
    assert json.loads((evidence.path / 'forward-001-failed.json').read_text())['forward'] == 'fails-before-return'
    # Use a separate attempt, since a failed attempt is never resumed in production.
    evidence, records, active = Evidence(tmp_path / 'other'), [], {'case_id': 'fabricated'}
    model.mode = 'mutate-input'
    with pytest.raises(ValueError, match='mutated'):
        run_forward(model, evidence, records, active, source, target, 'fails-after-return')
    assert active['return_observed'] and (evidence.path / active['saved_file']).is_file()
    assert (evidence.path / 'forward-000-observed.json').is_file()


@pytest.mark.parametrize('failure_site', ['isfinite', 'score'])
def test_validation_exceptions_preserve_current_return(tmp_path, monkeypatch, failure_site):
    import looking_glass.flan_t5_equal_dose as study
    model = FakeModel().eval()
    evidence, records, active = Evidence(tmp_path / 'failure'), [], {'case_id': 'fabricated'}
    source = {'input_ids': [2, 3, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [7, 1]}
    run_forward(model, evidence, records, active, source, target, 'success')
    def fail(*args, **kwargs):
        raise RuntimeError('injected validation failure')
    with monkeypatch.context() as local:
        local.setattr(torch if failure_site == 'isfinite' else study,
                      'isfinite' if failure_site == 'isfinite' else 'score_logits', fail)
        with pytest.raises(RuntimeError, match='injected'):
            run_forward(model, evidence, records, active, source, target, 'current-forward')
    assert active['forward'] == 'current-forward' and active['return_observed']
    assert active['saved_file'] == 'forward-001.bin'
    assert (evidence.path / active['saved_file']).is_file()
    assert json.loads((evidence.path / 'forward-001-observed.json').read_text())['forward'] == 'current-forward'


def test_plan_metadata_failure_keeps_already_retrieved_witnesses(tmp_path, monkeypatch):
    from looking_glass.intervention_plan import ActivationPlan
    model = FakeModel().eval()
    evidence, records, active = Evidence(tmp_path / 'failure'), [], {'case_id': 'fabricated'}
    source = {'input_ids': [2, 3, 1], 'attention_mask': [1, 1, 1]}
    target = {'input_ids': [7, 1]}
    def fail(_self):
        raise RuntimeError('injected metadata failure')
    with monkeypatch.context() as local:
        local.setattr(ActivationPlan, 'metadata', property(fail))
        with pytest.raises(RuntimeError, match='metadata failure'):
            run_forward(model, evidence, records, active, source, target, 'metadata-failure',
                        'patch', torch.full((512,), 7.0), 1)
    assert active['return_observed'] and active['tensor_layout']['before']['bytes'] == 2048
    assert active['tensor_layout']['after']['bytes'] == 2048
    assert (evidence.path / active['saved_file']).is_file()


def test_model_snapshot_detects_data_and_buffer_mutation():
    model = FakeModel().eval()
    original = model_snapshot(model)
    model.weight.data.add_(1)
    assert model_snapshot(model) != original
    changed = model_snapshot(model)
    model.marker.data.add_(1)
    assert model_snapshot(model) != changed


def test_complete_case_loop_with_fake_forwards_and_control_failure(tmp_path):
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
        if case_id == 'pair2-question':
            result[0, 1:, 0] = 100.0 if side == 0 else 100.00000762939453
            result[0, 0, 0] = 0.0 if side == 0 else 2e-6
        elif side:
            result[0, :, 0] = torch.tensor([2.0, 4.0, 6.0])
        return result

    def exercise(path, fail_control):
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
            swapped = '-full_swap-' in name
            logit = float(1-side if swapped else side)
            if fail_control and '-repeat-' in name:
                logit += 100
            margin = 2.0 if side == 0 else -2.0
            if case_id == 'pair2-translate':
                margin = 2.0  # both directions favor target A: unsupported
            if case_id.startswith('pair0-') and '-matched_word-' in name:
                margin = 0.0  # both directions move 2 nats toward the donor
            score = margin if target['target'] == 0 else 0.0
            return score, torch.tensor([logit]), state, before, after
        return run_cases({'cases': cases}, [[] for _ in cases], evidence, records, active, fake_forward), records

    summaries, records = exercise(tmp_path / 'complete', False)
    assert [row['status'] for row in summaries] == ['positive', 'positive', 'negative', 'negative',
                                                     'baseline-unsupported', 'dose-unmatchable']
    assert len(records) == 4 * 54 + 2 * 6
    assert aggregate_decision(summaries) == 'inconclusive'
    with pytest.raises(ValueError, match='identity logits control failed'):
        exercise(tmp_path / 'failed-control', True)
