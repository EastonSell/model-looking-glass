"""Fabricated-only checks for the prospective agreement-direction study."""
from collections import Counter
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import struct
import sys

import pytest
import torch

from looking_glass.agreement_journal import AgreementJournal


try:
    study = importlib.import_module('looking_glass.agreement_direction_study_v1')
except ModuleNotFoundError as error:
    if error.name != 'looking_glass.agreement_direction_study_v1':
        raise
    study = None


def fabricated_cases():
    targets = [{'case_id': f't{family}-{variant}', 'role': 'target',
                'family_index': family, 'variant': variant,
                'subject_number': ('singular' if variant % 2 == 0 else 'plural')}
               for family in range(8) for variant in range(4)]
    contexts = [{'case_id': f'c{family}', 'role': 'context', 'family_index': family}
                for family in range(8)]
    return targets + contexts


def model_cases():
    rows = fabricated_cases()
    for index, row in enumerate(rows[:32]):
        row.update({'token_ids': [1, 10 + index, 2 + index // 4, 3, 4],
                    'subject_index': 1, 'subject_lemma': 'a' if index % 4 < 2 else 'b'})
    for index, row in enumerate(rows[32:]):
        row['token_ids'] = [5, 6, 100 + index]
    return rows


def toy_model():
    path = Path(__file__).with_name('test_agreement_patch_runner.py')
    spec = importlib.util.spec_from_file_location('_agreement_direction_toy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(67)
        model = module.ToyModel().eval()
    with torch.no_grad():
        weights = model.transformer.embedding.weight
        for token in range(weights.shape[0]):
            weights[token].zero_()
            weights[token, 0] = float((token % 97) * 100)
            weights[token, 1] = float(((token * 3) % 43) * 100)
            weights[token, 2] = float(((token * 7) % 31) * 100)
    return model


def directions():
    return {'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v1': (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            'v2': (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)}


def test_fixed_schedule_has_784_actions_and_balances_arm_slots():
    assert study is not None
    schedule = study.fixed_schedule(fabricated_cases())
    assert len(schedule) == 784
    first = [(item['case_id'], item['action']) for item in schedule[:5]]
    assert first == [('t0-0', 'clean'), ('t0-0', 'clean_repeat'),
                     ('t0-0', 'noop_l3_final'), ('t0-0', 'noop_l3_subject'),
                     ('t0-0', 'noop_l1_final')]
    per_slot = Counter((item['action'], item['slot'], item['subject_number']) for item in schedule
                       if item.get('slot') is not None and item['role'] == 'target')
    assert len(per_slot) == 16 * 16 * 2
    assert all(count == 1 for count in per_slot.values())
    target_zero = [item['action'] for item in schedule if item['case_id'] == 't0-0']
    assert target_zero[5:9] == ['same_diff', 'opp_same', 'opp_diff', 'u_plus']
    target_one = [item['action'] for item in schedule if item['case_id'] == 't0-1']
    assert target_one[5:9] == ['u_minus', 'v1_plus', 'v1_minus', 'v2_plus']
    assert [item['action'] for item in schedule if item['case_id'] == 'c0'] == [
        'clean', 'clean_repeat', 'noop_l3_final', 'u_plus', 'u_minus', 'v1_plus',
        'v1_minus', 'v2_plus', 'v2_minus', 'clean_after']


def test_schedule_rejects_duplicate_or_reordered_case_identities():
    assert study is not None
    cases = fabricated_cases()
    cases[1] = {**cases[1], 'case_id': cases[0]['case_id']}
    with pytest.raises(ValueError):
        study.fixed_schedule(cases)


def test_family_dose_uses_all_six_distances_and_centering_uses_signed_half_gap():
    rows = [struct.pack('<2f', *value) for value in ((0, 0), (4, 0), (0, 6), (4, 6))]
    plan = study.family_geometry(rows, (1.0, 0.0))
    assert plan['doses'] == [2.0, 2.0, 2.0, 2.0]
    assert plan['distances'][0] == [6.0, 4.0, math.sqrt(52)]
    assert plan['distances'][1] == [6.0, 4.0, math.sqrt(52)]
    assert plan['centering'] == [-2.0, 2.0, -2.0, 2.0]
    for row, t in zip(rows, plan['centering']):
        delivered = study.direction_patch(row, (1.0, 0.0), -t)
        assert struct.unpack('<2f', delivered['value'])[0] == 2.0


def test_donor_and_direction_delivery_match_input_dose_and_preserve_zero_bits():
    base = struct.pack('<2f', 1.0, 2.0)
    donor = struct.pack('<2f', 5.0, 5.0)
    delivery = study.donor_patch(base, donor, 2.5)
    assert struct.unpack('<2f', delivery['value']) == (3.0, 3.5)
    assert delivery['requested_norm'] == delivery['realized_norm'] == 2.5
    signed_zero = struct.pack('<II', 0x80000000, 0)
    assert study.donor_patch(signed_zero, signed_zero, 0.0)['value'] == signed_zero
    assert study.direction_patch(signed_zero, (1.0, 0.0), 0.0)['value'] == signed_zero


def test_small_family_dose_and_tiny_center_are_exact_noops():
    rows = [struct.pack('<2f', x, 0.0) for x in (0.0, 2**-20, 2**-19, 3 * 2**-20)]
    plan = study.family_geometry(rows, (1.0, 0.0))
    assert plan['doses'] == [0.0] * 4
    assert plan['centering'] == [0.0] * 4
    assert plan['centering_gaps'] == [-(2**-20), -(2**-20)]
    assert all(study.direction_patch(row, (1.0, 0.0), -t)['value'] == row
               for row, t in zip(rows, plan['centering']))


def test_missed_rounded_dose_returns_candidate_bytes_for_failure_evidence():
    base = struct.pack('<2f', 2**30, 0.0)
    result = study.direction_patch(base, (1.0, 0.0), 1.0)
    assert result['value'] == base
    assert result['requested_norm'] == 1.0 and result['realized_norm'] == 0.0
    assert result['relative_error'] == 1.0 and result['dose_valid'] is False


def test_overflowed_rounded_candidate_remains_available_for_failure_evidence():
    base = struct.pack('<I', 0x7f7fffff)
    result = study.direction_patch(base, (1.0,), 2**104)
    assert result['value'] == struct.pack('<I', 0x7f800000)
    assert result['dose_valid'] is False and result['failure'] == 'nonfinite_rounded_candidate'


def test_fabricated_owner_executes_and_records_all_784_forwards(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = toy_model()
        journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
        result = study.run_direction_study(model, journal, model_cases(), directions())
        assert result['forwards'] == model.forward_calls == 784
        assert result['status'] == 'complete_candidate'
        records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in records if row['event']['kind'] == 'forward']
        assert len(forwards) == 784
        assert all('last_logits' in row['tensors'] for row in forwards)
        first_noop = next(row for row in forwards if row['event']['action'] == 'noop_l3_final')
        assert {'whole_before', 'before', 'after', 'whole_after'} <= set(first_noop['tensors'])
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'complete_candidate'
    finally:
        torch.set_num_threads(old_threads)


def test_failed_repeat_is_saved_before_the_attempt_stops(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = toy_model()
        original = model.forward
        def mismatching_repeat(*args, **kwargs):
            output = original(*args, **kwargs)
            if model.forward_calls == 2:
                output.logits = output.logits.clone()
                output.logits[0, -1, 0] += 1.0
            return output
        model.forward = mismatching_repeat
        journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
        with pytest.raises(study.StudyFailure):
            study.run_direction_study(model, journal, model_cases(), directions())
        records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in records if row['event']['kind'] == 'forward']
        assert [row['event']['action'] for row in forwards] == ['clean', 'clean_repeat']
        assert all('last_logits' in row['tensors'] for row in forwards)
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    finally:
        torch.set_num_threads(old_threads)


def test_zero_dose_arm_requires_clean_logits_and_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = toy_model()
        with torch.no_grad():
            model.transformer.embedding.weight.zero_()
        original = model.forward
        def bad_zero_arm(*args, **kwargs):
            output = original(*args, **kwargs)
            if model.forward_calls == 21:
                output.logits = output.logits.clone()
                output.logits[0, -1, 0] += 1.0
            return output
        model.forward = bad_zero_arm
        journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
        with pytest.raises(study.StudyFailure, match='no-op logits'):
            study.run_direction_study(model, journal, model_cases(), directions())
        rows = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in rows if row['event']['kind'] == 'forward']
        assert model.forward_calls == len(forwards) == 21
        assert forwards[-1]['event']['dose']['requested_norm'] == 0.0
        assert 'last_logits' in forwards[-1]['tensors']
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('limits', [{'max_forward_bytes': 4}, {'max_total_bytes': 100}])
def test_insufficient_journal_capacity_rejects_before_the_first_forward(tmp_path, monkeypatch, limits):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'}, **limits)
    with pytest.raises(study.StudyFailure):
        study.run_direction_study(model, journal, model_cases(), directions())
    assert model.forward_calls == 0
    terminal = json.loads((tmp_path / 'attempt/terminal.json').read_bytes())
    assert terminal['status'] == 'failed' and terminal['detail']['forwards_started'] == 0


def _make_bad_repeat(model):
    original = model.forward
    def bad(*args, **kwargs):
        output = original(*args, **kwargs)
        if model.forward_calls == 2:
            output.logits = output.logits.clone()
            output.logits[0, -1, 0] += 1.0
        return output
    model.forward = bad


def test_new_cancellation_during_failed_terminal_is_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    _make_bad_repeat(model)
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    original = journal.finish
    def interrupted(status, detail):
        if status == 'failed':
            raise KeyboardInterrupt('fresh cancellation')
        return original(status, detail)
    monkeypatch.setattr(journal, 'finish', interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run_direction_study(model, journal, model_cases(), directions())
    assert isinstance(caught.value.__cause__, study.StudyFailure)
    assert not (tmp_path / 'attempt/terminal.json').exists()


def test_new_cancellation_during_failed_cleanup_quarantines_and_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    _make_bad_repeat(model)
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    original = study._DirectionRunner._model_digest
    calls = 0
    def interrupted(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('fresh cancellation')
        return original(self)
    monkeypatch.setattr(study._DirectionRunner, '_model_digest', interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run_direction_study(model, journal, model_cases(), directions())
    assert isinstance(caught.value.__cause__, study.StudyFailure)
    terminal = json.loads((tmp_path / 'attempt/terminal.json').read_bytes())
    assert terminal['status'] == 'cancelled' and terminal['detail']['cleanup_quarantined'] is True


def test_returned_nonfinal_logits_survive_a_post_forward_environment_fault(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    model = toy_model()
    original = model.forward
    def drift(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits = output.logits.clone()
        output.logits[0, 0, 0] = float('nan')
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
        return output
    model.forward = drift
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    with pytest.raises(study.StudyFailure):
        study.run_direction_study(model, journal, model_cases(), directions())
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    rejected = [row for row in records if row['event']['kind'] == 'rejected_forward']
    assert len(rejected) == 1 and 'full_sequence_logits' in rejected[0]['tensors']
    ref = rejected[0]['tensors']['full_sequence_logits']
    raw = (tmp_path / 'attempt/tensors' / (ref['sha256'] + '.f32le')).read_bytes()
    assert (struct.unpack_from('<I', raw)[0] & 0x7f800000) == 0x7f800000


def test_malformed_oversized_logits_save_a_bounded_identified_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    original = model.forward
    def oversized(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits = torch.arange(1_500_000, dtype=torch.float32).reshape(1, 5, 300_000)
        return output
    model.forward = oversized
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    with pytest.raises(study.StudyFailure):
        study.run_direction_study(model, journal, model_cases(), directions())
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    rejected = [row for row in records if row['event']['kind'] == 'rejected_forward']
    assert len(rejected) == 1
    record = rejected[0]
    assert record['event']['logits_observation']['shape'] == [1, 5, 300_000]
    assert record['event']['logits_observation']['saved'] == 'first_batch_first_position_prefix'
    reference = record['tensors']['sequence_logits_prefix']
    assert reference['bytes'] <= 1024 * 1024
    raw = (tmp_path / 'attempt/tensors' / (reference['sha256'] + '.f32le')).read_bytes()
    assert struct.unpack_from('<3f', raw) == (0.0, 1.0, 2.0)
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'


def test_new_cancellation_during_pending_evidence_preserves_the_original_error(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    original = journal.record
    def injected(event, tensors):
        if event['kind'] == 'forward':
            raise ValueError('first publication failed')
        if event['kind'] == 'rejected_forward':
            raise KeyboardInterrupt('fresh cancellation')
        return original(event, tensors)
    monkeypatch.setattr(journal, 'record', injected)
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run_direction_study(model, journal, model_cases(), directions())
    assert isinstance(caught.value.__cause__, ValueError)
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'cancelled'


def test_interrupt_after_committed_plan_salvages_all_available_witnesses(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    original_trace = sys.gettrace()
    source, start = inspect.getsourcelines(study.NaturalAgreementRunner._forward)
    line = next(start + index for index, text in enumerate(source) if 'captures = plan.captures' in text)
    triggered = False
    def tracer(frame, event, arg):
        nonlocal triggered
        if (not triggered and event == 'line' and frame.f_code is study.NaturalAgreementRunner._forward.__code__
                and frame.f_lineno == line):
            triggered = True
            raise KeyboardInterrupt('after committed plan')
        return tracer
    try:
        sys.settrace(tracer)
        with pytest.raises(KeyboardInterrupt):
            study.run_direction_study(model, journal, model_cases(), directions())
    finally:
        sys.settrace(original_trace)
    assert triggered
    records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    rejected = [row for row in records if row['event']['kind'] == 'rejected_forward']
    assert len(rejected) == 1
    assert {'full_sequence_logits', 'whole_before', 'before', 'after', 'whole_after'} <= set(rejected[0]['tensors'])
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'cancelled'
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_failed_binary64_probe_blocks_the_first_forward(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    def bad_rounding():
        raise RuntimeError('unsupported arithmetic')
    monkeypatch.setattr(study, 'binary64_rounding_probe', bad_rounding, raising=False)
    with pytest.raises(RuntimeError, match='unsupported arithmetic'):
        study.run_direction_study(model, journal, model_cases(), directions())
    assert model.forward_calls == 0
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'
    cases = fabricated_cases()
    cases[0], cases[1] = cases[1], cases[0]
    with pytest.raises(ValueError):
        study.fixed_schedule(cases)


def test_run_contract_retains_external_prelaunch_and_exposure_binding(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'}, max_forward_bytes=4)
    binding = {'prelaunch_sha256': 'a' * 64, 'reservation_sha256': 'b' * 64,
               'latest_exposure_sha256': 'c' * 64}
    with pytest.raises(study.StudyFailure):
        study.run_direction_study(model, journal, model_cases(), directions(), binding=binding)
    events = [json.loads(line)['event'] for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    assert events[0]['kind'] == 'run_contract' and events[0]['external_binding'] == binding
    assert model.forward_calls == 0


def test_candidate_failure_evidence_is_reserved_before_delivery(tmp_path):
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'}, max_total_bytes=64)
    study._reserve_candidate_evidence(journal, 8)
    journal.record({'kind': 'fabricated'}, {'data': ((1,), struct.pack('<f', 1.0))})
    with pytest.raises(study.StudyFailure, match='candidate evidence'):
        study._reserve_candidate_evidence(journal, 8)


def test_custody_failure_stops_before_any_forward(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'})
    def no_custody():
        raise RuntimeError('lost custody')
    with pytest.raises(RuntimeError, match='lost custody'):
        study.run_direction_study(model, journal, model_cases(), directions(), custody_check=no_custody)
    assert model.forward_calls == 0
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'
