"""Fabricated tiny-model checks for the invariant CPU worker."""

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import pytest
import torch


SOURCE = Path(os.environ['INVARIANT_TEST_OWNER_SOURCE']).resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))


def load_package_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schedule_path = HERE / 'agreement_invariant_schedule_v2.py'
worker_path = HERE / 'agreement_invariant_executor_v2.py'
fake_path = HERE / 'test_prepare_invariant_cases_v1.py'
if not schedule_path.is_file():
    schedule_path = SOURCE / 'looking_glass/agreement_invariant_schedule_v2.py'
if not worker_path.is_file():
    worker_path = SOURCE / 'looking_glass/agreement_invariant_executor_v2.py'
if not fake_path.is_file():
    fake_path = SOURCE / 'results/distilgpt2-agreement-invariant-noun-cases-v1/test_prepare_invariant_cases_v1.py'
plan_v1 = load_package_source('looking_glass.agreement_invariant_schedule_v1', SOURCE / 'looking_glass/agreement_invariant_schedule_v1.py')
plan = load_package_source('looking_glass.agreement_invariant_schedule_v2', schedule_path)
study = load_package_source('looking_glass.agreement_invariant_executor_v2', worker_path)
fake = load_package_source('_fake_invariant_cases', fake_path)
toy = load_package_source('_fake_agreement_toy', SOURCE / 'tests/test_agreement_patch_runner.py')
from looking_glass.agreement_journal import AgreementJournal


def fabricated_cases():
    tokenizer = fake.FakeTokenizer()
    rows, _, _ = fake.prep.build(tokenizer, *fake.fake_priors(tokenizer), plan_v1)
    for row in rows:
        if row['split'] == 'discovery':
            row['case_id'] = 'v2_' + row['case_id']
    return rows


def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        result = toy.ToyModel()
        result.transformer.embedding = torch.nn.Embedding(1500, 8)
        result.head = torch.nn.Linear(8, 1500, bias=False)
        result.config.vocab_size = 1500
        return result.eval()


def directions():
    return {'u': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v1': (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            'v2': (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}


def binding():
    return {'prelaunch_sha256': '1' * 64, 'reservation_sha256': '2' * 64,
            'latest_exposure_sha256': '3' * 64}


def journal(tmp_path):
    return AgreementJournal(tmp_path / 'attempt', {'scope': 'fabricated'},
                            max_forward_bytes=4 * 1024 * 1024,
                            max_total_bytes=256 * 1024 * 1024,
                            max_event_log_bytes=32 * 1024 * 1024)


def test_fabricated_schedule_and_geometry():
    rows = fabricated_cases()
    actions = study.fixed_schedule(rows)
    assert len(actions) == 612
    assert [(x['case_id'], x['action']) for x in actions] == plan.actions()
    counts = Counter((x['action'], x['slot']) for x in actions if x['slot'] is not None)
    assert len(counts) == 36 and set(counts.values()) == {11, 12}
    assert len(study._validated_cases(rows, 1500)) == 68
    broken = fabricated_cases()
    target = next(x for x in broken if x['role'] == 'target' and x['determiner'] == 'these')
    target['token_ids'][0] += 100
    with pytest.raises(ValueError):
        study._validated_cases(broken, 1500)


def test_complete_toy_journal(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        toy_model = model()
        result = study.run_invariant_study(toy_model, journal(tmp_path), fabricated_cases(),
                                           directions(), binding=binding())
        assert result['status'] == 'complete_candidate'
        assert result['forwards'] == toy_model.forward_calls == 612
        events = [json.loads(line) for line in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
        forwards = [row for row in events if row['event']['kind'] == 'forward']
        assert len(events) == 613 and len(forwards) == 612
        assert [(x['event']['case_id'], x['event']['action']) for x in forwards] == plan.actions()
        noops = [x for x in forwards if x['event']['action'] == 'noop']
        assert len(noops) == 68
        assert all(x['tensors']['before']['sha256'] == x['tensors']['after']['sha256']
                   and x['tensors']['whole_before']['sha256'] == x['tensors']['whole_after']['sha256']
                   for x in noops)
        arms = [x for x in forwards if x['event']['action'] in plan.BASE]
        assert len(arms) == 408 and all(x['event']['dose']['dose_valid'] is True for x in arms)
        assert all(not module._forward_hooks and not module._forward_pre_hooks for module in toy_model.modules())
        assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] == 'complete_candidate'
    finally:
        torch.set_num_threads(old_threads)


def test_prefilled_journal_fails_before_forward(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    toy_model, evidence = model(), journal(tmp_path)
    evidence.record({'kind': 'prior'}, {'prior': ((1,), struct.pack('<f', 1.0))})
    with pytest.raises(ValueError, match='fresh'):
        study.run_invariant_study(toy_model, evidence, fabricated_cases(), directions(), binding=binding())
    assert toy_model.forward_calls == 0


def test_noop_disagreement_retained_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    toy_model = model()
    original = toy_model.forward
    def changed_noop(*args, **kwargs):
        result = original(*args, **kwargs)
        if toy_model.forward_calls == 2:
            result.logits[0, -1, 0] += 1.0
        return result
    toy_model.forward = changed_noop
    with pytest.raises(Exception, match='no-op logits differ'):
        study.run_invariant_study(toy_model, journal(tmp_path), fabricated_cases(), directions(), binding=binding())
    assert toy_model.forward_calls == 2
    assert json.loads((tmp_path / 'attempt/terminal.json').read_bytes())['status'] != 'complete_candidate'


def test_interrupted_return_retains_output_and_labels_counter(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    original = study._SubjectRunner._root_forward
    interrupted = False
    def interrupt_once(self, name, value):
        nonlocal interrupted
        original(self, name, value)
        if name == 'output' and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(study._SubjectRunner, '_root_forward', interrupt_once)
    toy_model = model()
    with pytest.raises(KeyboardInterrupt):
        study.run_invariant_study(toy_model, journal(tmp_path), fabricated_cases(), directions(), binding=binding())
    records = [json.loads(x) for x in (tmp_path / 'attempt/events.jsonl').read_bytes().splitlines()]
    rejected = next(x for x in records if x['event']['kind'] == 'rejected_forward')
    assert 'full_sequence_logits' in rejected['tensors']
    terminal = json.loads((tmp_path / 'attempt/terminal.json').read_bytes())
    assert terminal['status'] != 'complete_candidate'
    assert terminal['detail']['model_returns'] == 0
    assert terminal['detail']['model_returns_scope'].startswith('normal-path observations only')
