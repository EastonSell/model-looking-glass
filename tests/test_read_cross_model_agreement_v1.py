"""Check the fixed sign convention and complete-case prediction rule."""
import numpy as np
import pytest
import json
import torch

from scripts import read_cross_model_agreement_v1 as reader
from scripts import cross_model_agreement_v1 as writer
from test_cross_model_agreement_v1 import Model, fake_manifest


def test_singular_and_plural_effects_both_point_toward_opposite_donor():
    targets = {'is': 0, 'are': 1, 'was': 2, 'were': 3}
    singular = {'case_id': 's', 'frame': 0, 'lemma': 0, 'number': 0, 'attractor_number': 0}
    plural = {**singular, 'case_id': 'p', 'number': 1}
    s = {'clean': np.array([2., 0., 1., 0.]), 'opposite': np.array([0., 0., 1., 0.]),
         'same': np.array([1., 0., 1., 0.]), 'random': np.array([1.5, 0., 1., 0.])}
    p = {name: -values for name, values in s.items()}
    for result in (reader.score_case(singular, s, targets), reader.score_case(plural, p, targets)):
        assert result['clean_correct']
        assert result['effects']['opposite']['donorward_effect'] == pytest.approx(2)
        assert result['opposite_minus_same'] == pytest.approx(1)
        assert result['opposite_minus_random'] == pytest.approx(1.5)
        assert result['effects']['opposite']['tv'] > 0
        assert result['effects']['opposite']['kl_clean_to_arm'] > 0


def test_prediction_requires_both_controls_in_same_cases():
    good = {'clean_correct': True, 'opposite_minus_same': 1., 'opposite_minus_random': 1., 'requested_q': 2.}
    rows = [{**good, 'frame': index // 16, 'lemma': (index // 2) % 4} for index in range(32)]
    for row in rows[:9]:
        row['opposite_minus_random'] = 0.
    summary = reader.summarize(rows)
    assert summary['positive_against_both'] == 23
    assert summary['by_frame']['0']['positive_against_both'] == 7
    assert summary['by_lemma']['0']['cases'] == 8
    assert summary['requested_q'] == {'min': 2., 'max': 2., 'mean': 2.}
    assert not summary['descriptive_prediction_passes']
    assert summary['interpretation'] == 'does-not-support-fixed-descriptive-prediction'
    for row in rows[:9]:
        row['clean_correct'] = False
    assert reader.summarize(rows)['interpretation'] == 'baseline-limited'


def test_tiny_tv_and_kl_use_stable_formulas_and_expose_ties():
    tail = reader.distribution(np.array([0., -100.], np.float32), np.array([0., -80.], np.float32))
    assert tail['tv'] == pytest.approx(1.80485138e-35, rel=1e-7)
    assert not tail['numerically_unresolved']
    epsilon = float(np.nextafter(np.float32(0), np.float32(1)))
    close = reader.distribution(np.array([0., 0.], np.float32), np.array([0., epsilon], np.float32))
    assert close['tv'] == pytest.approx(epsilon / 4, rel=1e-12)
    assert close['kl_clean_to_arm'] == pytest.approx(epsilon * epsilon / 8, rel=1e-12)
    assert close['clean_top1_ties'] == 2 and close['arm_top1_ties'] == 1
    assert close['representative_top1_id_changed'] and close['unique_top1_changed'] is None


def test_exact_comparison_distinguishes_signed_zero():
    assert not reader.same_bits(np.array([0.], np.float32), np.array([-0.], np.float32))


def test_many_underflowed_tails_are_explicitly_unresolved():
    clean = np.full(50001, -800., np.float32)
    changed = np.full(50001, -750., np.float32)
    clean[0] = changed[0] = 0.
    result = reader.distribution(clean, changed)
    assert result['numerically_unresolved']
    assert result['tv'] is None and result['kl_clean_to_arm'] is None


def test_reader_fails_closed_if_thread_flushes_subnormals():
    try:
        reader.require_subnormals()
        originally_preserved = True
    except RuntimeError:
        originally_preserved = False
    try:
        if not torch.set_flush_denormal(True):
            pytest.skip('platform cannot control denormal flushing')
        with pytest.raises(RuntimeError, match='subnormals are flushed'):
            reader.distribution(np.array([0., 0.], np.float32), np.array([0., 1.], np.float32))
    finally:
        torch.set_flush_denormal(not originally_preserved)


def test_replay_checks_fake_run_raw_bytes_and_detects_tampering(tmp_path, monkeypatch):
    model, manifest = Model(), fake_manifest()
    journal = writer.Journal(tmp_path, 16)
    writer.run(model, model.block, manifest, 'gpt', torch.device('cpu'), journal)
    journal.close()
    metadata = {'model': 'gpt', 'case_sha256': reader.CASE_SHA, 'pid': 1234, 'source_commit': 'a' * 40}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    fields = {'starts.jsonl': 'starts_sha256', 'events.jsonl': 'events_sha256',
              'memory.jsonl': 'memory_sha256', 'retention-errors.jsonl': 'retention_errors_sha256',
              'logits.f32': 'logits_sha256', 'witness.f32': 'witness_sha256'}
    complete = {'status': 'complete', 'events': 192, 'metadata_sha256': reader.digest(tmp_path / 'metadata.json')}
    complete.update({field: reader.digest(tmp_path / name) for name, field in fields.items()})
    (tmp_path / 'complete.json').write_text(json.dumps(complete))
    observation = {'schema': 'cross-model-direct-child-exit-v1', 'status': 'complete', 'exit_code': 0,
                   'interrupted': None, 'child_pid': 1234, 'source_commit': 'a' * 40,
                   'attempt_name': tmp_path.name, 'complete_sha256': reader.digest(tmp_path / 'complete.json'),
                   'metadata_sha256': reader.digest(tmp_path / 'metadata.json')}
    monkeypatch.setitem(reader.VOCAB, 'gpt', 16)
    monkeypatch.setitem(reader.TARGETS, 'gpt', {'is': 0, 'are': 1, 'was': 2, 'were': 3})
    report = reader.replay(tmp_path, manifest, observation)
    assert report['summary']['cases'] == len(report['cases']) == 32
    (tmp_path / 'failure.txt').write_text('post-completion failure')
    with pytest.raises(ValueError, match='not complete'):
        reader.replay(tmp_path, manifest, observation)
    (tmp_path / 'failure.txt').unlink()
    with pytest.raises(ValueError, match='exit was not observed'):
        reader.replay(tmp_path, manifest, {**observation, 'exit_code': 1})
    with (tmp_path / 'witness.f32').open('ab') as stream:
        stream.write(b'\x00')
    with pytest.raises(ValueError, match='digest mismatch'):
        reader.replay(tmp_path, manifest, observation)
