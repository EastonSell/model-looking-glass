"""Small fake-model checks for intervention, order and saved evidence."""
import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import cross_model_agreement_v1 as study


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        basis = torch.arange(768, device=input_ids.device, dtype=torch.float32) / 1000
        hidden = input_ids.float().unsqueeze(-1) * (basis + 1)
        patched = self.block(hidden)[0]
        values = patched.sum(dim=(1, 2))
        logits = values[:, None, None] * torch.arange(1, 17, dtype=torch.float32)[None, None, :]
        return SimpleNamespace(logits=logits.expand(-1, input_ids.shape[1], -1).clone())


def fake_manifest():
    cases = []
    for frame in range(2):
        for attractor in range(2):
            for lemma in range(4):
                for number in range(2):
                    index = len(cases)
                    row = {'input_ids': [index + 1, 1], 'attention_mask': [1, 1],
                           'subject_position': 0, 'output_position': 1}
                    cases.append({'case_id': str(index), 'index': index, 'frame': frame,
                                  'attractor_number': attractor, 'lemma': lemma, 'number': number,
                                  'same_number_donor_lemma': [1, 0, 3, 2][lemma], 'gpt': row})
    return {'cases': cases}


def test_replacement_is_dose_matched_and_deterministic():
    base = np.linspace(-1, 1, 768, dtype=np.float32)
    donor = base + 0.2
    patched, actual = study.replacement(base, donor, 2.0)
    assert actual == pytest.approx(2.0, abs=1e-5)
    assert float(np.linalg.norm((patched - base).astype(np.float64))) == pytest.approx(2.0, abs=1e-5)
    first, first_q = study.random_replacement(base, 2.0, 20261029)
    second, second_q = study.random_replacement(base, 2.0, 20261029)
    assert np.array_equal(first, second) and first_q == second_q
    with pytest.raises(ValueError):
        study.replacement(base, base, 1.0)
    ones = np.ones(768, dtype=np.float32)
    tiny = ones.copy()
    tiny[0] = np.nextafter(tiny[0], np.float32(2.0))
    q = float(np.linalg.norm(tiny.astype(np.float64) - ones.astype(np.float64)))
    with pytest.raises(ValueError, match='dose'):
        study.random_replacement(ones, q, 20261029)
    tiny = ones.copy()
    tiny[:172] = np.nextafter(tiny[:172], np.float32(0.0))
    q = float(np.linalg.norm(tiny.astype(np.float64) - ones.astype(np.float64)))
    with pytest.raises(ValueError, match='dose'):
        study.random_replacement(ones, q, 20261029)


def test_snapshot_verification_rejects_changed_and_extra_assets(tmp_path, monkeypatch):
    file = tmp_path / 'config.json'
    file.write_bytes(b'configuration')
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(study, 'ASSET_SHA', {'gpt': {'config.json': digest}})
    assert study.verify_assets(tmp_path, 'gpt') == {'config.json': digest}
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='asset differs'):
        study.verify_assets(tmp_path, 'gpt')
    file.write_bytes(b'configuration')
    (tmp_path / 'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='additional assets'):
        study.verify_assets(tmp_path, 'gpt')


def test_source_verification_requires_exact_committed_bytes(tmp_path, monkeypatch):
    sources = {'scripts/cross_model_agreement_v1.py': b'run',
               'docs/CROSS_MODEL_AGREEMENT_V1.md': b'protocol',
               'experiments/cross_model_agreement_cases_v1.json': b'cases'}
    for name, value in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    monkeypatch.setattr(study, '__file__', str(tmp_path / 'scripts/cross_model_agreement_v1.py'))
    def fake_git(args, **_kwargs):
        return SimpleNamespace(stdout=sources[args[-1].split(':', 1)[1]])
    monkeypatch.setattr(study.subprocess, 'run', fake_git)
    cases = tmp_path / 'experiments/cross_model_agreement_cases_v1.json'
    assert len(study.verify_source('a' * 40, cases)) == 3
    (tmp_path / 'scripts/cross_model_agreement_v1.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='local source differs'):
        study.verify_source('a' * 40, cases)
    with pytest.raises(ValueError, match='full lowercase Git SHA'):
        study.verify_source('not-a-commit', cases)


def test_hook_replaces_only_selected_state_and_removes_it():
    model = Model()
    row = fake_manifest()['cases'][0]['gpt']
    clean, before, after, finite = study.forward(model, model.block, row, None, torch.device('cpu'))
    assert finite
    assert np.array_equal(before, after)
    changed, before2, after2, finite = study.forward(model, model.block, row, before + 1, torch.device('cpu'))
    assert finite
    assert np.array_equal(before, before2) and np.array_equal(after2, before + 1)
    assert not np.array_equal(clean, changed)
    assert not model.block._forward_hooks


def test_full_fake_schedule_records_donor_controls_and_witnesses(tmp_path):
    model, manifest = Model(), fake_manifest()
    journal = study.Journal(tmp_path, 16)
    study.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
    journal.close()
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(events) == 192
    assert len((tmp_path / 'starts.jsonl').read_text().splitlines()) == 192
    assert [entry['arm'] for entry in events[:32]] == ['clean'] * 32
    assert [entry['arm'] for entry in events[32:37]] == ['self', 'opposite', 'same', 'random', 'clean_after']
    assert events[33]['donor_index'] == 1 and events[34]['donor_index'] == 2
    assert events[33]['actual_l2'] == pytest.approx(events[34]['actual_l2'], abs=1e-5)
    assert (tmp_path / 'logits.f32').stat().st_size == 192 * 16 * 4
    assert (tmp_path / 'witness.f32').stat().st_size == 192 * 2 * 768 * 4
    assert not model.block._forward_hooks


def test_nonselected_nan_preserves_start_and_returned_selected_evidence(tmp_path):
    class NonfiniteModel(Model):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            result.logits[0, 0, 0] = float('nan')
            return result

    model, manifest = NonfiniteModel(), fake_manifest()
    journal = study.Journal(tmp_path, 16)
    with pytest.raises(ValueError, match='nonfinite clean'):
        study.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
    journal.close()
    starts = [json.loads(line) for line in (tmp_path / 'starts.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(starts) == len(events) == 1
    assert starts[0]['arm'] == 'clean' and events[0]['full_logits_finite'] is False
    assert (tmp_path / 'logits.f32').stat().st_size == 16 * 4
    assert (tmp_path / 'witness.f32').stat().st_size == 2 * 768 * 4
    assert not model.block._forward_hooks


def test_interrupt_at_first_returned_write_retains_incomplete_payload(tmp_path, monkeypatch):
    model, manifest = Model(), fake_manifest()
    journal = study.Journal(tmp_path, 16)
    write = journal.write
    calls = []
    def interrupted_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt()
        return write(*args, **kwargs)
    monkeypatch.setattr(journal, 'write', interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        study.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
    journal.close()
    starts = [json.loads(line) for line in (tmp_path / 'starts.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(starts) == len(events) == 1
    assert events[0]['returned_status'] == 'interrupted'
    assert (tmp_path / 'logits.f32').stat().st_size == 16 * 4
    assert (tmp_path / 'witness.f32').stat().st_size == 2 * 768 * 4
    assert not model.block._forward_hooks


def test_interrupt_after_forward_return_still_keeps_returned_payload(tmp_path, monkeypatch):
    model, manifest = Model(), fake_manifest()
    journal = study.Journal(tmp_path, 16)
    real_forward = study.forward
    def interrupted_after_return(*args, **kwargs):
        real_forward(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(study, 'forward', interrupted_after_return)
    with pytest.raises(KeyboardInterrupt):
        study.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
    journal.close()
    assert len((tmp_path / 'starts.jsonl').read_text().splitlines()) == 1
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1
    assert (tmp_path / 'logits.f32').stat().st_size == 16 * 4
    assert (tmp_path / 'witness.f32').stat().st_size == 2 * 768 * 4


def test_model_changing_denormal_mode_fails_after_preserving_return(tmp_path):
    try:
        study.require_subnormals()
        originally_preserved = True
    except RuntimeError:
        originally_preserved = False
    class FlushModel(Model):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if not torch.set_flush_denormal(True):
                raise RuntimeError('platform does not support denormal control')
            return result
    model, manifest = FlushModel(), fake_manifest()
    journal = study.Journal(tmp_path, 16)
    try:
        with pytest.raises(RuntimeError, match='subnormals are flushed'):
            study.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
        journal.close()
        assert len((tmp_path / 'starts.jsonl').read_text().splitlines()) == 1
        assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1
    finally:
        torch.set_flush_denormal(not originally_preserved)
