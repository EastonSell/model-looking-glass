"""Fabricated cases only: no checkpoint or prospective study inputs are used."""

from collections import Counter
import importlib
import math
import json
from pathlib import Path
import struct

import pytest
import torch


def subject_module():
    try:
        return importlib.import_module('looking_glass.agreement_subject_executor_v1')
    except ModuleNotFoundError as error:
        if error.name != 'looking_glass.agreement_subject_executor_v1':
            raise
        pytest.fail('subject-dependence executor is not implemented')


def fabricated_cases():
    conditions = ('none', 'mass', 'attractor_singular', 'attractor_plural')
    pairs = (('R1', 'I1'), ('R1', 'I2'), ('R2', 'I1'), ('R2', 'I2'))
    subjects = {'R1': (100, 101), 'R2': (102, 103),
                'I1': (104, 105), 'I2': (106, 107)}
    rows = []
    for frame in range(4):
        for condition_index, condition in enumerate(conditions):
            final = (6, 8, 20 + frame * 2, 21 + frame * 2)[condition_index]
            for pair_index, key in enumerate(pairs[frame]):
                for number_index, number in enumerate(('singular', 'plural')):
                    prefix = [1 + frame, 5, subjects[key][number_index]]
                    tokens = prefix + ([6] if condition_index == 0 else [7, final])
                    rows.append({'case_id': f'f{frame}-c{condition_index}-{key}-{number}',
                                 'frame_index': frame, 'condition': condition,
                                 'condition_index': condition_index, 'pair_index': pair_index,
                                 'role': 'target', 'subject_key': key,
                                 'subject_number': number,
                                 'morphology': 'regular' if pair_index == 0 else 'irregular',
                                 'text': f'fabricated-{frame}-{condition}-{key}-{number}',
                                 'token_ids': tokens, 'subject_index': 2,
                                 'final_index': len(tokens) - 1})
            tokens = [1 + frame] + ([6] if condition_index == 0 else [7, final])
            rows.append({'case_id': f'f{frame}-c{condition_index}-fragment',
                         'frame_index': frame, 'condition': condition, 'role': 'context',
                         'condition_index': condition_index, 'pair_index': None,
                         'subject_key': None, 'subject_number': None, 'morphology': None,
                         'text': f'fabricated-{frame}-{condition}-fragment',
                         'token_ids': tokens, 'subject_index': None,
                         'final_index': len(tokens) - 1})
    return rows


def test_fabricated_schedule_balances_all_720_actions():
    study = subject_module()
    rows = fabricated_cases()
    schedule = study.fixed_schedule(rows)
    assert len(schedule) == 720
    by_case = {}
    for index, action in enumerate(schedule):
        assert action['index'] == index
        by_case.setdefault(action['case_id'], []).append(action)
    assert len(by_case) == 80
    for actions in by_case.values():
        assert len(actions) == 9
        assert [a['action'] for a in actions[:2]] == ['clean', 'noop_l3_final']
        assert actions[-1]['action'] == 'clean_after'
        assert {a['action'] for a in actions[2:8]} == {
            f'{name}_{sign}' for name in ('u', 'v1', 'v2') for sign in ('plus', 'minus')}
        assert [a['slot'] for a in actions[2:8]] == list(range(6))

    positions = Counter((a['action'], a['slot']) for a in schedule if a['slot'] is not None)
    assert len(positions) == 36 and set(positions.values()) == {13, 14}
    first_signs = Counter()
    first_subjects = Counter()
    case_order = [schedule[i]['case_id'] for i in range(0, 720, 9)]
    for frame in range(4):
        for condition in range(4):
            block = rows[frame * 20 + condition * 5:frame * 20 + condition * 5 + 5]
            expected = []
            for pair in range(2):
                members = block[pair * 2:pair * 2 + 2]
                if (frame + frame // 2 + pair) % 2:
                    members = list(reversed(members))
                first_subjects[(members[0]['subject_key'], members[0]['subject_number'])] += 1
                expected.extend(item['case_id'] for item in members)
                left, right = (by_case[item['case_id']][2:8] for item in members)
                assert [item['action'] for item in left] == [item['action'] for item in right]
            expected.append(block[-1]['case_id'])
            assert case_order[frame * 20 + condition * 5:frame * 20 + condition * 5 + 5] == expected
            for row in block:
                arms = [item['action'] for item in by_case[row['case_id']][2:8]]
                for direction in ('u', 'v1', 'v2'):
                    sign = 'plus' if arms.index(direction + '_plus') < arms.index(direction + '_minus') else 'minus'
                    group = row['subject_number'] if row['role'] == 'target' else 'context'
                    first_signs[(direction, group, sign)] += 1
                    if row['role'] == 'target':
                        first_signs[(direction, row['morphology'], sign)] += 1
    for direction in ('u', 'v1', 'v2'):
        for group, count in (('singular', 16), ('plural', 16),
                             ('regular', 16), ('irregular', 16), ('context', 8)):
            assert first_signs[(direction, group, 'plus')] == count
            assert first_signs[(direction, group, 'minus')] == count
    for key in ('R1', 'R2', 'I1', 'I2'):
        assert first_subjects[(key, 'singular')] == first_subjects[(key, 'plural')] == 4


def test_schedule_rejects_duplicate_or_misordered_case_identity():
    study = subject_module()
    rows = fabricated_cases()
    rows[1]['case_id'] = rows[0]['case_id']
    with pytest.raises(ValueError):
        study.fixed_schedule(rows)
    rows = fabricated_cases()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        study.fixed_schedule(rows)


@pytest.mark.parametrize('field,value', [('condition_index', 2), ('pair_index', 1)])
def test_schedule_rejects_conflicting_explicit_coordinates(field, value):
    study = subject_module()
    rows = fabricated_cases()
    rows[0][field] = value
    with pytest.raises(ValueError):
        study.fixed_schedule(rows)


def test_journal_accepts_explicit_bounded_four_mib_failure_reservation(tmp_path):
    from looking_glass.agreement_journal import AgreementJournal
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    payload = bytes(12 * 50257 * 4)
    journal.record({'kind': 'fabricated_full_sequence'}, {'logits': ((1, 12, 50257), payload)})
    assert journal.retained_bytes == len(payload)


def test_budget_rejects_small_event_log_before_execution(tmp_path):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024,
                               max_event_log_bytes=256 * 1024)
    with pytest.raises(ValueError, match='reservation'):
        study._check_budget(journal, 600, 8)


def test_prefilled_journal_is_rejected_before_any_forward(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    journal.record({'kind': 'prior_event'}, {'prior': ((1,), struct.pack('<f', 1.0))})
    model = _toy_model()
    def forbid_forward(*_args, **_kwargs):
        raise AssertionError('forward was reached')
    model.forward = forbid_forward
    with pytest.raises(ValueError, match='fresh'):
        study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
    assert model.forward_calls == 0


@pytest.mark.parametrize('amount', [2.0, -2.0])
def test_fixed_delivery_records_signed_projection_and_exact_bits(amount):
    study = subject_module()
    source = struct.pack('<2f', 1.0, -3.0)
    result = study.signed_direction_patch(source, (1.0, 0.0), amount)
    assert result['value'] == struct.pack('<2f', 1.0 + amount, -3.0)
    assert result['requested_norm'] == result['realized_norm'] == 2.0
    assert result['signed_projected_displacement'] == amount
    assert result['projected_relative_error'] == 0.0
    assert result['source_norm'] == math.sqrt(10)
    assert result['changed_elements'] == 1 and result['changed_bit_count'] > 0
    assert result['dose_valid'] is True


def test_projected_tolerance_is_checked_independently_of_l2(monkeypatch):
    study = subject_module()
    source = struct.pack('<2f', 0.0, 0.0)
    x = struct.unpack('<f', struct.pack('<f', 1.99995))[0]
    y = struct.unpack('<f', struct.pack('<f', math.sqrt(4.0 - x * x)))[0]
    assert abs(math.hypot(x, y) - 2.0) / 2.0 < 1e-5
    assert abs(x - 2.0) / 2.0 > 1e-5
    fake = {'value': struct.pack('<2f', x, y), 'requested_norm': 2.0,
            'realized_norm': math.hypot(x, y), 'dose_valid': True}
    monkeypatch.setattr(study, 'direction_patch', lambda *_: dict(fake))
    checked = study.signed_direction_patch(source, (1.0, 0.0), 2.0)
    assert checked['dose_valid'] is False
    assert checked['signed_projected_displacement'] == x
    assert checked['projected_relative_error'] > 1e-5


def _toy_model():
    path = Path(__file__).with_name('test_agreement_patch_runner.py')
    spec = importlib.util.spec_from_file_location('_subject_toy_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        return module.ToyModel().eval()


def _directions():
    return {'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v1': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v2': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}


def _binding():
    return {'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
            'latest_exposure_sha256': '3' * 64}


def test_fabricated_complete_run_retains_all_actions_and_controls(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _toy_model()
        journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                                   max_forward_bytes=4 * 1024 * 1024)
        result = study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
        assert result['status'] == 'complete_candidate'
        assert result['forwards'] == model.forward_calls == 720
        records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in records if row['event']['kind'] == 'forward']
        assert len(forwards) == 720
        assert all('last_logits' in row['tensors'] for row in forwards)
        hooked = [row for row in forwards if row['event']['action'] != 'clean'
                  and row['event']['action'] != 'clean_after']
        assert len(hooked) == 560
        assert all({'whole_before', 'before', 'after', 'whole_after'} <= set(row['tensors'])
                   for row in hooked)
        noops = [row for row in hooked if row['event']['action'] == 'noop_l3_final']
        assert len(noops) == 80
        assert all(row['tensors']['before']['sha256'] == row['tensors']['after']['sha256']
                   and row['tensors']['whole_before']['sha256'] == row['tensors']['whole_after']['sha256']
                   for row in noops)
        arms = [row for row in hooked if row['event']['action'] != 'noop_l3_final']
        assert len(arms) == 480
        assert all(row['event']['dose']['dose_valid'] is True
                   and abs(abs(row['event']['dose']['signed_projected_displacement']) - 2.0) < 2e-5
                   for row in arms)
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'complete_candidate'
    finally:
        torch.set_num_threads(old_threads)


def test_invalid_tokens_or_insufficient_evidence_stop_before_forward(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    rows = fabricated_cases()
    rows[1]['token_ids'] = list(rows[0]['token_ids'])
    model = _toy_model()
    journal = AgreementJournal(tmp_path / 'bad-case', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(ValueError):
        study.run_subject_study(model, journal, rows, _directions(), binding=_binding())
    assert model.forward_calls == 0
    model = _toy_model()
    journal = AgreementJournal(tmp_path / 'small', {'scope': 'fabricated'}, max_total_bytes=1024,
                               max_forward_bytes=1024)
    with pytest.raises(ValueError, match='reservation'):
        study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
    assert model.forward_calls == 0


def test_prefix_overlapping_inputs_stop_before_any_forward(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    rows = fabricated_cases()
    rows[4]['token_ids'] = rows[0]['token_ids'][:2]
    rows[4]['final_index'] = 1
    model = _toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(ValueError, match='prefix'):
        study.run_subject_study(model, journal, rows, _directions(), binding=_binding())
    assert model.forward_calls == 0


def test_noop_disagreement_stops_and_preserves_the_forward(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = _toy_model()
    original = model.forward
    def changed_noop(*args, **kwargs):
        result = original(*args, **kwargs)
        if model.forward_calls == 2:
            result.logits[0, -1, 0] += 1.0
        return result
    model.forward = changed_noop
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(Exception, match='no-op logits differ'):
        study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
    assert model.forward_calls == 2
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    assert [row['event']['action'] for row in records if row['event']['kind'] == 'forward'] == ['clean', 'noop_l3_final']
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'


def test_invalid_projected_dose_is_preserved_before_any_arm_forward(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    original = study.signed_direction_patch
    def reject(*args):
        result = original(*args)
        result['dose_valid'] = False
        return result
    monkeypatch.setattr(study, 'signed_direction_patch', reject)
    model = _toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(Exception, match='dose rule'):
        study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
    assert model.forward_calls == 2
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    rejected = [row for row in records if row['event']['kind'] == 'invalid_delivery']
    assert len(rejected) == 1 and set(rejected[0]['tensors']) == {'source', 'candidate'}
    assert rejected[0]['event']['dose']['signed_projected_displacement'] is not None
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'


def test_nonfinite_full_sequence_is_retained_at_the_twelve_token_bound(tmp_path, monkeypatch):
    from looking_glass.agreement_journal import AgreementJournal
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = _toy_model()
    model.config.vocab_size = 50257
    model.head = torch.nn.Linear(8, 50257, bias=False).eval()
    original = model.forward
    def nonfinite_earlier_position(*args, **kwargs):
        result = original(*args, **kwargs)
        result.logits[0, 0, 0] = float('nan')
        return result
    model.forward = nonfinite_earlier_position
    rows = fabricated_cases()
    for row in rows:
        if row['role'] == 'target':
            tokens = row['token_ids']
            row['token_ids'] = tokens[:-1] + [9] * (12 - len(tokens)) + tokens[-1:]
            row['final_index'] = 11
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(Exception, match='nonfinite sequence logits'):
        study.run_subject_study(model, journal, rows, _directions(), binding=_binding())
    assert model.forward_calls == 1
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    forward = next(row for row in records if row['event']['kind'] == 'forward')
    assert forward['event']['all_sequence_logits_finite'] is False
    assert forward['tensors']['full_sequence_logits']['shape'] == [1, 12, 50257]
    assert forward['tensors']['full_sequence_logits']['bytes'] == 12 * 50257 * 4
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'


def test_full_capture_finiteness_is_checked_from_saved_bits():
    study = subject_module()
    with pytest.raises(ValueError):
        study._finite_captures({'whole_before': struct.pack('<II', 0x3f800000, 0x7fc00000)})


@pytest.mark.parametrize('location', ['failure_cleanup', 'root_recovery', 'hook_inspection'])
def test_fresh_cleanup_cancellation_wins_and_quarantines(tmp_path, monkeypatch, location):
    from looking_glass.agreement_direction_study_v1 import _DirectionRunner
    from looking_glass.agreement_journal import AgreementJournal
    from looking_glass import distilgpt2_intervention_dev
    study = subject_module()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = _toy_model()
    def failed_forward(*_args, **_kwargs):
        raise ValueError('fabricated forward failure')
    model.forward = failed_forward
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt('fresh cancellation')
    if location == 'failure_cleanup':
        monkeypatch.setattr(_DirectionRunner, '_failed_forward', interrupt)
    elif location == 'root_recovery':
        original = _DirectionRunner._root_forward
        def interrupted_root(self, name, value):
            if name == 'ids':
                interrupt()
            return original(self, name, value)
        monkeypatch.setattr(_DirectionRunner, '_root_forward', interrupted_root)
    else:
        monkeypatch.setattr(distilgpt2_intervention_dev, '_hook_state', interrupt)
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run_subject_study(model, journal, fabricated_cases(), _directions(), binding=_binding())
    assert isinstance(caught.value.__cause__, ValueError)
    terminal = json.loads((tmp_path / 'attempt/terminal.json').read_bytes())
    assert terminal['status'] == 'cancelled'
    assert terminal['detail']['cleanup_quarantined'] is True
