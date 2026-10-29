"""Synthetic 486-action worker tests; no checkpoint or public cases are used."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import struct

import pytest
import torch

from looking_glass import agreement_substitution_executor_v1 as study
from looking_glass.agreement_journal import AgreementJournal


def fabricated_cases():
    subjects = {'R1': (200, 201), 'R2': (202, 203),
                'I1': (204, 205), 'I2': (206, 207)}
    rows = []
    for frame in range(2):
        for condition_index, condition in enumerate(study.CONDITIONS):
            suffix = [40] if condition_index == 0 else [30, 50 + 2 * frame + condition_index]
            for morphology_index, (morphology, keys) in enumerate((('regular', ('R1', 'R2')),
                                                                   ('irregular', ('I1', 'I2')))):
                for lemma_index, key in enumerate(keys):
                    for number_index, number in enumerate(('singular', 'plural')):
                        tokens = [100 + frame, 20, subjects[key][number_index]] + suffix
                        rows.append({'case_id': f'f{frame}-c{condition_index}-{key}-{number}',
                                     'frame_index': frame, 'condition_index': condition_index,
                                     'condition': condition, 'role': 'target',
                                     'morphology': morphology, 'morphology_index': morphology_index,
                                     'lemma_index': lemma_index, 'subject_key': key,
                                     'subject_number': number, 'text': f'fake-{frame}-{condition}-{key}-{number}',
                                     'token_ids': tokens, 'subject_index': 2, 'final_index': len(tokens) - 1})
            tokens = [100 + frame] + suffix
            rows.append({'case_id': f'f{frame}-c{condition_index}-context',
                         'frame_index': frame, 'condition_index': condition_index,
                         'condition': condition, 'role': 'context', 'morphology': None,
                         'morphology_index': None, 'lemma_index': None, 'subject_key': None,
                         'subject_number': None, 'text': f'fake-{frame}-{condition}-context',
                         'token_ids': tokens, 'subject_index': None, 'final_index': len(tokens) - 1})
    return rows


def toy_module():
    spec = importlib.util.spec_from_file_location('_substitution_toy',
                                                   Path(__file__).with_name('test_agreement_patch_runner.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        return toy_module().ToyModel().eval()


def directions():
    return {'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v1': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v2': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}


def binding():
    return {'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
            'latest_exposure_sha256': '3' * 64}


def test_fabricated_schedule_and_all_arm_slots():
    rows = fabricated_cases()
    schedule = study.fixed_schedule(rows)
    assert len(schedule) == 486
    assert [row['index'] for row in schedule] == list(range(486))
    assert len({row['case_id'] for row in schedule}) == 54
    counts = Counter((row['action'], row['slot']) for row in schedule if row['slot'] is not None)
    assert len(counts) == 36 and set(counts.values()) == {9}
    for start in range(0, len(schedule), 9):
        group = schedule[start:start + 9]
        assert [row['action'] for row in group[:2]] == ['clean', 'noop_l3_final']
        assert group[-1]['action'] == 'clean_after'
        assert {row['action'] for row in group[2:8]} == {
            name + '_' + sign for name in ('u', 'v1', 'v2') for sign in ('plus', 'minus')}
        assert len({row['case_id'] for row in group}) == 1
    assert len(study._validated_cases(rows, 600)) == 54


def test_case_misalignment_and_duplicate_identity_rejected():
    rows = fabricated_cases()
    rows[1]['case_id'] = rows[0]['case_id']
    with pytest.raises(ValueError):
        study.fixed_schedule(rows)
    rows = fabricated_cases()
    rows[1]['token_ids'][0] -= 1
    with pytest.raises(ValueError, match='outside'):
        study._validated_cases(rows, 600)


def test_fabricated_full_run_and_landings(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = toy_model()
        journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                                   max_forward_bytes=4 * 1024 * 1024)
        result = study.run_substitution_study(model, journal, fabricated_cases(), directions(), binding=binding())
        assert result['status'] == 'complete_candidate'
        assert result['forwards'] == model.forward_calls == 486
        records = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in records if row['event']['kind'] == 'forward']
        assert len(forwards) == 486
        assert all('last_logits' in row['tensors'] for row in forwards)
        noops = [row for row in forwards if row['event']['action'] == 'noop_l3_final']
        assert len(noops) == 54
        assert all(row['tensors']['before']['sha256'] == row['tensors']['after']['sha256']
                   and row['tensors']['whole_before']['sha256'] == row['tensors']['whole_after']['sha256']
                   for row in noops)
        arms = [row for row in forwards if row['event']['action'] not in ('clean', 'noop_l3_final', 'clean_after')]
        assert len(arms) == 324
        assert all(row['event']['dose']['dose_valid'] is True for row in arms)
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'complete_candidate'
    finally:
        torch.set_num_threads(old_threads)


def test_prefilled_journal_stops_before_forward(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
    journal = AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                               max_forward_bytes=4 * 1024 * 1024)
    journal.record({'kind': 'prior'}, {'prior': ((1,), struct.pack('<f', 1.0))})
    with pytest.raises(ValueError, match='fresh'):
        study.run_substitution_study(model, journal, fabricated_cases(), directions(), binding=binding())
    assert model.forward_calls == 0


def test_noop_disagreement_is_recorded_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    model = toy_model()
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
        study.run_substitution_study(model, journal, fabricated_cases(), directions(), binding=binding())
    assert model.forward_calls == 2
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'failed'
